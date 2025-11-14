import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class PhoneticEncoder(nn.Module):
    def __init__(self, model_name="bert-base-multilingual-cased"):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.embedding_dim = self.model.config.hidden_size

    def encode(self, texts):
        toks = self.tokenizer(texts, padding=True, truncation=True, max_length=64, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**toks)
            return out.last_hidden_state.mean(dim=1)

