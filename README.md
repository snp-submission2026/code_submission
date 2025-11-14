# Context Analysis — Project README

This repository contains code and models for context-aware analysis tasks (preprocessing, training, evaluation). The main code lives under the `src/` folder. Data and model checkpoints are stored in `data/` and `models/` respectively.

## Quick overview

- `src/` : Python package with preprocessing, dataset class, model class and utility modules.
- `data/` : CSV and dataset files used by scripts.
- `models/` : Saved model checkpoints.
- `requirements.txt` : Python dependencies.

## Prerequisites

- Python 3.10+
- pip

If you plan to use GPU features (PyTorch + CUDA), install compatible CUDA-enabled PyTorch separately following the official instructions: https://pytorch.org/get-started/locally/

## Recommended setup (venv)

1. Create and activate a virtual environment (from repo root):

```bash
# create venv
python3 -m venv .venv

# activate (bash)
source .venv/bin/activate
```

2. Upgrade pip and install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

3. (Optional) If you want editable installs of local packages:

```bash
pip install -e .
```

## Running the code
1. To train the model
```bash
python train_model.py
```
2. To evaluate the model
```bash
python test_model.py
```