import torch
import numpy as np
import emoji
from gensim.models import KeyedVectors
from transformers import AutoTokenizer
from typing import List, Dict, Any
import re
from collections import Counter
from src.config import EMOJI_MODEL_PATH

EMBEDDING_DIM = 300
FACET_SLICE_SIZE = 28

try:
    EMOJI_VECTORS = KeyedVectors.load_word2vec_format(EMOJI_MODEL_PATH, binary=True)
except Exception as e:
    print(f"Warning: Could not load Emoji2Vec model. Using a dummy setup. Error: {e}")
    EMOJI_VECTORS = None

PROJECTION_LAYER = torch.nn.Linear(EMBEDDING_DIM, FACET_SLICE_SIZE, bias=False)
PROJECTION_LAYER.eval()

def get_emoji_vector_feature_projected(text: str) -> List[float]:

    """
    Extracts emojis from text, averages their 300-dim Emoji2Vec embeddings, 
    projects the result to FACET_SLICE_SIZE (e.g., 28-dim), and returns a list of floats.
    """
    
    emojis_in_text = [e['emoji'] for e in emoji.emoji_list(text)]
    valid_vectors = []

    if EMOJI_VECTORS is None or not emojis_in_text:
        # Graceful handling: return a list of zeros if model failed or no emojis found
        return [0.0] * FACET_SLICE_SIZE

    for em in emojis_in_text:
        # Gensim's KeyedVectors handles lookup efficiently
        if em in EMOJI_VECTORS.key_to_index:
            # Retrieve the 300-dim NumPy vector
            vec_np = EMOJI_VECTORS[em]
            valid_vectors.append(torch.tensor(vec_np, dtype=torch.float32))

    if not valid_vectors:
        return [0.0] * FACET_SLICE_SIZE

    avg_vector = torch.mean(torch.stack(valid_vectors), dim=0)

    avg_vector = avg_vector.unsqueeze(0)

    with torch.no_grad():
        projected_vector = PROJECTION_LAYER(avg_vector).squeeze(0)

    return projected_vector.tolist()