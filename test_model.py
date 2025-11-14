from src.config import DATA_PATH, MODEL_PATH, RESULT_PATH, DEVICE, TEST_DATA_PATH, ROOT_PATH
from src.data_loader import load_test_dataset
from src.preprocess import split_data, parse_facet_column, build_facet_keys, prepare_facet_vectors_single,get_clean_text
from src.preprocess_facet import compute_multilingual_facet_dict
from src.dataset import PostDataset
from src.model.bert_with_facets_film import BertWithFacets_FiLM
from src.utils.metrics import evaluate_predictions
from sklearn.metrics import accuracy_score, f1_score, hamming_loss
from src.utils.plot_utils import plot_confusion_matrix
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import torch
import numpy as np
import json
from tqdm import tqdm
import pandas as pd
from sklearn.preprocessing import label_binarize

# Load dataset
val_df, _, _ = load_test_dataset(TEST_DATA_PATH)

res_df = val_df.copy()
val_df['text'] = val_df['text'].apply(get_clean_text)
print("⚙️ Generating facets...")
val_df['facets'] = [json.dumps(compute_multilingual_facet_dict(t, lang)) for t, lang in tqdm(zip(val_df["text"], val_df["language"]))]
val_df["facets"] = val_df["facets"].apply(parse_facet_column)
#print(val_df.head())

facet_keys = build_facet_keys(val_df)
val_df, numeric_keys = prepare_facet_vectors_single(val_df, facet_keys)

# Dataset & Dataloader
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")
val_ds = PostDataset(val_df, numeric_keys, tokenizer, max_len=128)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

# Load model
model = BertWithFacets_FiLM(
    facet_dim=len(numeric_keys),
    hidden_dim=512,
    num_classes=5,
    num_languages=5,
    dropout_rate=0.35
).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# Evaluate
all_preds, all_labels, all_probs, all_langs = [], [], [], []

label_names = ["Hate", "Neither", "Offensive", "Threat", "Toxic"]
lang_names = ["Arabic", "Bangla", "English", "Spanish", "Hindi"]

preditc_labes = []


with torch.no_grad():
    for batch in val_loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        facets = batch["facets"].to(DEVICE)
        language = batch["language"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        logits, _ = model(
            input_ids,
            attention_mask,
            facets,
            language,
            return_gate=True
        )
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        # all_preds.extend(preds.cpu().numpy())
        # all_labels.extend(batch["labels"].numpy())

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_langs.extend(language.cpu().numpy())

        print("Batch Predictions:", preds.cpu().numpy())

        l_name = [label_names[i] for i in preds.cpu().numpy()]
        preditc_labes.extend(l_name)
        print(preditc_labes)



res_df['predicted_label'] = preditc_labes
res_df = res_df.drop(columns=["language", "label_id", "language_id"])
res_df.to_csv(ROOT_PATH / 'data' / "test_predictions_with_model.csv", index=False)
report, cm = evaluate_predictions(np.array(all_labels), np.array(all_preds), label_names)

val_acc = accuracy_score(all_labels, all_preds)
macro_f1 = f1_score(all_labels, all_preds, average="macro")

print(f"Val Acc: {val_acc:.4f} | Macro-F1: {macro_f1:.4f}")

print(report)

# # # Save cm + labels to JSON
# output_json = {
#     "labels": le_label.classes_.tolist(),
#     "confusion_matrix": cm.tolist()
# }

# with open(RESULT_PATH / "confusion_matrix.json", "w") as f:
#     json.dump(output_json, f, indent=4)

# print(f"\n✅ Confusion matrix saved to: {RESULT_PATH / 'confusion_matrix.json'}")


# # Convert to numpy arrays
# all_preds = np.array(all_preds)
# all_labels = np.array(all_labels)
# all_probs = np.array(all_probs)
# all_langs = np.array(all_langs)

# # Suppose your label encoder defines:

# results = []

# for lang_id, lang_name in enumerate(lang_names):
#     mask = (all_langs == lang_id)
#     y_true = all_labels[mask]
#     y_pred = all_preds[mask]
#     y_prob = all_probs[mask]

#     if len(y_true) == 0:
#         continue

#     # --- Binary (single label) metrics ---
#     macro_f1 = f1_score(y_true, y_pred, average="macro")
#     acc = accuracy_score(y_true, y_pred)

#     # --- Convert to multilabel (one-hot) for AUC, subset acc, hamming loss ---
#     y_true_bin = label_binarize(y_true, classes=np.arange(len(label_names)))
#     y_pred_bin = label_binarize(y_pred, classes=np.arange(len(label_names)))

#     macro_f1_multi = f1_score(y_true_bin, y_pred_bin, average="macro", zero_division=0)
#     subset_acc = accuracy_score(y_true_bin, y_pred_bin)
#     hamming = hamming_loss(y_true_bin, y_pred_bin)

#     results.append({
#         "Language": lang_name,
#         "Macro-F1 (Binary)": macro_f1,
#         "Acc (Binary)": acc,
#         "Macro-F1 (Multi)": macro_f1_multi,
#         "Subset Acc": subset_acc,
#         "Hamming Loss": hamming
#     })

# # Convert to dataframe
# df_results = pd.DataFrame(results)
# print(df_results.round(2))

# # Save to LaTeX
# print("\nLaTeX table:\n")
# print(df_results.to_latex(index=False, float_format="%.2f"))

