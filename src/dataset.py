import torch
import json
import numpy as np
from torch.utils.data import Dataset

class PostDataset(Dataset):
    def __init__(self, df, facet_keys, tokenizer, max_len=128):
        self.texts = df["text"].tolist()
        self.labels = df["label_id"].tolist()
        self.language = df["language_id"].tolist()
        self.facet_keys = facet_keys
        self.tokenizer = tokenizer
        self.max_len = max_len

        self.fdicts = []
        for x in df["facets"]:
            if isinstance(x, str):
                try:
                    self.fdicts.append(json.loads(x))
                except Exception:
                    self.fdicts.append({})
            else:
                self.fdicts.append(x or {})

    def __len__(self):
        return len(self.texts)

    def _vec(self, d):
        return np.array([float(d.get(k, 0.0)) for k in self.facet_keys], dtype=np.float32)

    def __getitem__(self, idx):
        text = self.texts[idx]
        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        facet_vec = torch.tensor(self._vec(self.fdicts[idx]), dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        language = torch.tensor(self.language[idx], dtype=torch.long)

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "facets": facet_vec,
            "language": language,
            "labels": label
        }
