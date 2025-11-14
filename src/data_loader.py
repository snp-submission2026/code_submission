import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_dataset(csv_path):
    df = pd.read_csv(csv_path)

    # Encode context_label
    le_label = LabelEncoder()
    df["label_id"] = le_label.fit_transform(df["context_label"])

    # Encode language
    le_lang = LabelEncoder()
    df["language_id"] = le_lang.fit_transform(df["language"])

    return df, le_label, le_lang
def load_test_dataset(csv_path) -> (pd.DataFrame, LabelEncoder, LabelEncoder):
    df = pd.read_csv(csv_path)
    le_label = LabelEncoder()
    df["label_id"] = le_label.fit_transform(df["context_label"])

    # Encode language
    le_lang = LabelEncoder()
    df["language_id"] = le_lang.fit_transform(df["language"])

    return df, le_label, le_lang