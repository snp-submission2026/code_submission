from src.config import DATA_PATH, DEVICE, RESULT_PATH, ROOT_PATH,TRAIN_DATA_PATH
from src.data_loader import load_dataset
from src.preprocess import split_data, parse_facet_column, build_facet_keys, prepare_facet_vectors,get_clean_text
from src.preprocess_facet import compute_multilingual_facet_dict, compute_facet_dict
from src.dataset import PostDataset
from src.model.bert_with_facets_film import BertWithFacets_FiLM
from src.utils.training_utils import focal_loss
from src.utils.metrics import evaluate_predictions
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from torch.amp import autocast, GradScaler
import torch, numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import json
import torch.nn as nn
import re


def clean_user(text):
    text = re.sub(r"<\s*user\s*>", "", str(text), flags=re.IGNORECASE).strip()
    text = re.sub(r"@", "", str(text), flags=re.IGNORECASE).strip()
    return text

# Load and preprocess
df, le_label, le_lang = load_dataset(TRAIN_DATA_PATH)
df['text'] = df['text'].apply(get_clean_text)
df['text'] = df['text'].apply(clean_user)
print("⚙️ Generating facets...")
df['facets'] = [json.dumps(compute_multilingual_facet_dict(t, lang)) for t, lang in tqdm(zip(df["text"], df["language"]))]

df["facets"] = df["facets"].apply(parse_facet_column)
train_df, val_df = split_data(df)
facet_keys = build_facet_keys(train_df)
train_df, val_df, numeric_keys = prepare_facet_vectors(train_df, val_df, facet_keys)


cls_counts = torch.tensor(
    [(train_df["label_id"] == i).sum() for i in range(len(le_label.classes_))],
    dtype=torch.float32
)
weights_ce = (cls_counts.sum() / (len(cls_counts) * cls_counts))
criterion = nn.CrossEntropyLoss(weight=weights_ce.to(DEVICE))

# ------------------- BALANCED SAMPLER -------------------
labels_tensor = torch.tensor(train_df["label_id"].values)
class_sample_count = torch.tensor(
    [(labels_tensor == i).sum() for i in range(len(le_label.classes_))],
    dtype=torch.float32
)
weights_per_class = 1.0 / class_sample_count
sample_weights = weights_per_class[labels_tensor]

sampler = WeightedRandomSampler(
    weights=sample_weights.double(),
    num_samples=len(sample_weights),
    replacement=True
)

# ------------------- TOKENIZER -------------------
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")

# Use your filtered numeric_keys list (NOT FACET_KEYS with strings)
facet_keys = numeric_keys

train_ds = PostDataset(train_df, facet_keys, tokenizer, max_len=256)
val_ds   = PostDataset(val_df,   facet_keys, tokenizer, max_len=256)

train_loader = DataLoader(train_ds, batch_size=32, sampler=sampler)
val_loader   = DataLoader(val_ds, batch_size=32, shuffle=False)

# Model
model = BertWithFacets_FiLM(
    facet_dim=len(facet_keys),
    hidden_dim=512,
    num_classes=len(le_label.classes_),
    num_languages=len(le_lang.classes_),
    dropout_rate=0.35
).to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.05)

num_epochs = 12
num_steps = num_epochs * len(train_loader)
warmup_steps = int(0.06 * num_steps)
scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, num_steps)
scaler = GradScaler("cuda")

# ------------------- REGULARIZATION & TRACKING -------------------
GATE_REG = 3e-3
best_macro_f1 = -1.0
best_val_loss = float("inf")
best_state = None
patience, patience_counter = 3, 0

train_loss_arr = []
val_loss_arr = []
val_acc_arr = []
macro_f1_arr = []
gate_weight_arr = []

# ------------------- TRAINING LOOP -------------------
for epoch in range(1, num_epochs + 1):
    model.train()
    total_loss = 0.0
    total_batches = 0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}"):
        input_ids = batch["input_ids"].to(DEVICE, non_blocking=True)
        attention_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
        facets = batch["facets"].to(DEVICE, dtype=torch.float32, non_blocking=True)
        labels = batch["labels"].to(DEVICE, non_blocking=True)
        language = batch["language"].to(DEVICE, non_blocking=True) # Move language tensor to CUDA

        optimizer.zero_grad(set_to_none=True)
        with autocast("cuda"):
            logits, gate_values = model(input_ids, attention_mask, facets, language, return_gate=True)
            loss_cls = focal_loss(logits, labels, weight=weights_ce.to(DEVICE))
            gate_reg_loss = GATE_REG * torch.mean(torch.abs(gate_values - 0.5))
            loss = loss_cls + gate_reg_loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scheduler.step()
        scaler.update()

        total_loss += loss.item()
        total_batches += 1

    train_loss = total_loss / total_batches

    # ------------------- VALIDATION -------------------
    model.eval()
    val_loss = 0.0
    all_preds, all_labels = [], []
    gate_means = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(DEVICE, non_blocking=True)
            attention_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
            facets = batch["facets"].to(DEVICE, dtype=torch.float32, non_blocking=True)
            labels = batch["labels"].to(DEVICE, non_blocking=True)
            language = batch["language"].to(DEVICE, non_blocking=True) # Move language tensor to CUDA

            with autocast("cuda"):
                logits, gate_values = model(input_ids, attention_mask, facets, language, return_gate=True)
                loss = criterion(logits, labels)

            val_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            gate_means.append(gate_values.mean().item())

    val_loss /= len(val_loader)
    val_acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    avg_gate = np.mean(gate_means)
    gate_var = np.var(gate_means)

    # ------------------- LOGGING -------------------
    print(f"\nEpoch {epoch} - Train Loss: {train_loss:.4f}")
    print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Macro-F1: {macro_f1:.4f}")
    print(f"Average gate weight: {avg_gate:.4f} | Gate variance: {gate_var:.6f}")
    # #print("\nClassification Report:")
    # print(classification_report(all_labels, all_preds, target_names=list(le.classes_)))

    train_loss_arr.append(train_loss)
    val_loss_arr.append(val_loss)
    val_acc_arr.append(val_acc)
    macro_f1_arr.append(macro_f1)
    gate_weight_arr.append(avg_gate)


    # ------------------- CHECKPOINT -------------------
    if macro_f1 > best_macro_f1:
        best_macro_f1 = macro_f1
        best_state = model.state_dict()
        patience_counter = 0
        torch.save(best_state, "models/best_model.pt")
        print("✓ Saved new best model checkpoint")
    else:
        patience_counter += 1
        print(f"No improvement ({patience_counter}/{patience})")
        if patience_counter >= patience:
            print("⚠️ Early stopping triggered.")
            break

# output_json = {
#     "train_loss": train_loss_arr.tolist(),
#     "val_loss": val_loss_arr.tolist(),
#     "val_accuracy": val_acc_arr.tolist(),
#     "macro_f1": macro_f1_arr.tolist(),
#     "gate_weights": gate_weight_arr.tolist()
# }

# with open(RESULT_PATH / "all_matrix.json", "w") as f:
#     json.dump(output_json, f, indent=4)

# ------------------- RESTORE & FINAL EVAL -------------------
if best_state:
    model.load_state_dict(best_state)
    print("\n✅ Loaded best model for final evaluation.")

print("\nFinal Classification Report:")
print(classification_report(all_labels, all_preds, target_names=list(le_label.classes_)))
print("\nConfusion Matrix:")
print(confusion_matrix(all_labels, all_preds))
