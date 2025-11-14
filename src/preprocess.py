import json, ast
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import re

def get_clean_text(text: str) -> str:
    """
    Remove extra spaces and unwanted characters from text.
    """
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
    text = re.sub(r'[^\S\r\n]+', ' ', text)  # Remove non-breaking spaces
    text = re.sub(r'[*/]', '', text)
    return text

def parse_facet_column(x):
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except Exception as e:
            return {}
    elif isinstance(x, dict):
        return x
    else:
        return {}

def build_facet_keys(df, key_col="facets"):
    def parse(x):
        if isinstance(x, str):
            try:
                return json.loads(x)
            except Exception:
                return {}
        return x or {}
    dicts = [parse(x) for x in df[key_col].tolist()]
    keys = set()
    for d in dicts:
        keys.update(d.keys())
    return sorted(keys)

def prepare_facet_vectors(train_df, val_df, keys):
    numeric_keys = [
        k for k in keys
        if isinstance(train_df["facets"].iloc[0].get(k, 0.0), (int, float, np.integer, np.floating))
    ]
    train_fac = np.stack([[row.get(k, 0.0) for k in numeric_keys] for row in train_df["facets"]]).astype("float32")
    val_fac   = np.stack([[row.get(k, 0.0) for k in numeric_keys] for row in val_df["facets"]]).astype("float32")

    scaler = StandardScaler()
    scaler.fit(train_fac)

    train_scaled = scaler.transform(train_fac)
    val_scaled   = scaler.transform(val_fac)

    train_df["facets"] = [dict(zip(numeric_keys, vec)) for vec in train_scaled]
    val_df["facets"]   = [dict(zip(numeric_keys, vec)) for vec in val_scaled]
    return train_df, val_df, numeric_keys

def prepare_facet_vectors_single(df, keys):
    numeric_keys = [
        k for k in keys
        if isinstance(df["facets"].iloc[0].get(k, 0.0), (int, float, np.integer, np.floating))
    ]

    val_fac   = np.stack([[row.get(k, 0.0) for k in numeric_keys] for row in df["facets"]]).astype("float32")

    scaler = StandardScaler()
    scaler.fit(val_fac)

    val_scaled   = scaler.transform(val_fac)

    df["facets"]   = [dict(zip(numeric_keys, vec)) for vec in val_scaled]

    return df, numeric_keys


def split_data(df):
    return train_test_split(df, test_size=0.10, stratify=df["label_id"], random_state=42)
