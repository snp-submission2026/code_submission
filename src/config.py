import torch
from pathlib import Path

ROOT_PATH = Path(__file__).resolve().parent.parent
TRAIN_DATA_PATH = ROOT_PATH / "data" / "train_with_facets.csv"
TEST_DATA_PATH = ROOT_PATH / "data" / "val_with_facets.csv"
DATA_PATH = ROOT_PATH / "data" / "processed_data.csv"
MODEL_PATH = ROOT_PATH / "models" / "best_model.pt"
PLOT_PATH = ROOT_PATH / "plots"
RESULT_PATH = Path("results")
EMOJI_MODEL_PATH = ROOT_PATH / "pretrains" / "emoji2vec.bin"
RESULT_PATH.mkdir(exist_ok=True)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
