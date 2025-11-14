import re, json, emoji, torch, nltk, spacy
from collections import Counter
from typing import Dict, Any
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
from src.process_emoji import FACET_SLICE_SIZE
from src.process_emoji import get_emoji_vector_feature_projected
from src.process_script_entropy import calculate_script_entropy

# Download resources if missing
try:
    _ = nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon")

vader = SentimentIntensityAnalyzer()
nlp_multi = spacy.load("xx_ent_wiki_sm")

URL_RE = re.compile(r'https?://\S+|www\.\S+')
MENTION_RE = re.compile(r'@\w+')
HASHTAG_RE = re.compile(r'#\w+')

ENTITY_LABELS = [
    "PERSON","NORP","FAC","ORG","GPE","LOC","PRODUCT","EVENT","WORK_OF_ART",
    "LAW","LANGUAGE","DATE","TIME","PERCENT","MONEY","QUANTITY","ORDINAL","CARDINAL"
]


try:
    # Use the same model name as your BertWithFacets_FiLM
    BERT_TOKENIZER = AutoTokenizer.from_pretrained("xlm-roberta-large")
except Exception as e:
    print(f"Warning: Could not load BERT tokenizer. OOV calculation disabled. Error: {e}")
    BERT_TOKENIZER = None

def calculate_oov_rate(text: str, bert_tokenizer: AutoTokenizer) -> float:
    """
    Calculates the Out-Of-Vocabulary (OOV) rate based on XLM-R's tokenization.
    OOV tokens are identified by the tokenizer's special [UNK] token (or equivalent).
    """
    
    if bert_tokenizer is None or not text.strip():
        return 0.0

    # Tokenize the text without special tokens (like [CLS], [SEP])
    # The return_offsets_mapping=True is often useful for clean indexing, but not strictly required here.
    
    tokens = bert_tokenizer.tokenize(text, add_special_tokens=False)
    
    if not tokens:
        return 0.0
    
    # The XLM-R tokenizer uses <unk> for unknown tokens
    unk_token = bert_tokenizer.unk_token
    
    # Count the number of unknown/OOV tokens
    oov_count = sum(1 for token in tokens if token == unk_token)
    
    # Calculate the ratio
    oov_rate = oov_count / len(tokens)
    
    return round(oov_rate, 6)

# ---- Multilingual Negation ----
NEGATION_WORDS = {
    "not", "no", "never", "none", "neither", "nobody", "nothing",         # English
    "नहीं", "मत", "ना", "न", "nahi", "na", "mat",                         # Hindi
    "না", "নয়", "নে", "noy", "nei", "na",                                # Bangla
    "لا", "لم", "لن", "ليس", "ما", "مش", "مو", "la", "ma", "mesh", "mush",# Arabic
    "no", "nunca", "jamás", "ninguno", "nada", "nadie"                    # Spanish
}

def count_negations(text: str) -> int:
    toks = re.findall(r"\w+", text.lower())
    return sum(tok in NEGATION_WORDS for tok in toks)

# Toxic lexicon
EN_TOXIC = {"hate","kill","bastard","bitch","whore","slut","nigga","faggot","moron","idiot","stupid","terrorist","garbage","loser"}
HI_TOXIC = {"हरामी","गधा","साला","कुत्ता","कमीना","चूतिया","bhosdike","randi","gandu","madarchod","behenchod","pagal"}
BN_TOXIC = {"বোকা","হারামজাদা","চোদন","কুত্তা","বেশ্যা","মাগী","bokachoda","magi","randhi","fukni","harami"}
AR_TOXIC = {"كلب","حمار","ابن","زاني","كافر","نجس","وسخ","غبي","قذر","harami","ibn","kalb","sharmoota"}
ES_TOXIC = {"puta","maricón","zorra","perra","cabrón","idiota","estúpido","negro","sudaca","basura","escoria","rata"}

TOXICITY_LEXICON = set.union(EN_TOXIC, HI_TOXIC, BN_TOXIC, AR_TOXIC, ES_TOXIC)

def _compile_lexicon_regex(lexicon):
    parts = sorted((re.escape(w) for w in lexicon if w.strip()), key=len, reverse=True)
    return re.compile(r"(?i)(?<!\w)(" + r"|".join(parts) + r")(?!\w)") if parts else re.compile(r"$^")

def compute_multilingual_facet_dict(text: str, lang: str) -> Dict[str, Any]:
    if not text or not str(text).strip():
        base = {k: 0.0 for k in [
            "word_count","avg_word_len","vader_sentiment",
            "scramble_ratio","non_ascii_ratio","punct_ratio","ner_count",
            "entity_density","hashtag_count","mention_count","url_count",
            "allcaps_ratio","elong_ratio","negation_count","tox_count",
            "tox_rate","emoji_count","emoji_density", "oov_rate", "script_entropy"
        ]}

        for i in range(FACET_SLICE_SIZE): 
            base[f'emoji_dim_{i+1}'] = 0.0

        for lab in ENTITY_LABELS:
            base[f'ent_{lab}_count'] = 0
            base[f'ent_{lab}_density'] = 0.0
        base["lang_detected"] = "unknown"
        return base


    # Tokenize
    doc = nlp_multi(text)
    toks = [t for t in doc if not t.is_space]
    word_toks = [t for t in toks if t.is_alpha]
    wc = len(word_toks)
    avg_len = (sum(len(t.text) for t in word_toks) / wc) if wc else 0.0

    vader_sent = 0.0
    if lang == "en" or re.search(r"[A-Za-z]", text):
        vader_sent = vader.polarity_scores(text)["compound"]

    emoji_vector_proj = get_emoji_vector_feature_projected(text)

    oov_rate = calculate_oov_rate(text, BERT_TOKENIZER)

    script_entropy = calculate_script_entropy(text)


    non_alpha_tokens = sum(1 for t in toks if not t.is_alpha)
    scramble_ratio = (non_alpha_tokens / len(toks)) if toks else 0.0
    non_ascii_chars = sum(1 for ch in text if ord(ch) > 127)
    non_ascii_ratio = (non_ascii_chars / len(text)) if text else 0.0
    punct_ratio = (sum(1 for t in toks if t.is_punct) / len(toks)) if toks else 0.0

    hashtag_count = len(HASHTAG_RE.findall(text))
    mention_count = len(MENTION_RE.findall(text))
    url_count = len(URL_RE.findall(text))

    allcaps_tokens = sum(1 for t in word_toks if t.text.isupper() and len(t) > 1)
    allcaps_ratio = (allcaps_tokens / wc) if wc else 0.0
    elong_ratio = (sum(1 for t in word_toks if re.search(r"(.)\1\1", t.text.lower())) / wc) if wc else 0.0
    negation_count = count_negations(text)

    emojis = [c for c in text if c in emoji.EMOJI_DATA]
    emoji_count = len(emojis)
    emoji_density = emoji_count / len(text) if text else 0.0

    ent_counter = Counter(ent.label_ for ent in doc.ents)
    ner_count = sum(ent_counter.values())
    token_count = len(toks) or 1
    entity_density = ner_count / token_count

    tox_count = 0; tox_rate = 0.0
    if TOXICITY_LEXICON:
        rx = _compile_lexicon_regex(TOXICITY_LEXICON)
        tox_count = len(rx.findall(text))
        tox_rate = (tox_count / wc) if wc else 0.0

    facets = {
        "word_count": wc,
        "avg_word_len": round(avg_len,3),
        "vader_sentiment": round(vader_sent,3),
        "scramble_ratio": round(scramble_ratio,6),
        "non_ascii_ratio": round(non_ascii_ratio,6),
        "punct_ratio": round(punct_ratio,6),
        "ner_count": int(ner_count),
        "entity_density": round(entity_density,6),
        "hashtag_count": int(hashtag_count),
        "mention_count": int(mention_count),
        "url_count": int(url_count),
        "allcaps_ratio": round(allcaps_ratio,6),
        "elong_ratio": round(elong_ratio,6),
        "negation_count": int(negation_count),
        "tox_count": int(tox_count),
        "tox_rate": round(tox_rate,6),
        "emoji_count": emoji_count,
        "emoji_density": round(emoji_density,6),
        "oov_rate": round(oov_rate,6),
        "script_entropy": round(script_entropy,6),
    }

    for i, val in enumerate(emoji_vector_proj):
        # Keys are now 'emoji_dim_1', 'emoji_dim_2', ..., 'emoji_dim_28'
        facets[f"emoji_dim_{i+1}"] = round(val, 6)

    # --- Add NER features
    for lab in ENTITY_LABELS:
        count = ent_counter.get(lab, 0)
        facets[f"ent_{lab}_count"] = count
        facets[f"ent_{lab}_density"] = round(count / token_count, 6)

    

    ##--- Add phonetic vector
    # try:
    #     with torch.no_grad():
    #         phonetic_vec = phonetic_encoder.encode([text]).squeeze(0).cpu().numpy()
    #     for i, val in enumerate(phonetic_vec):
    #         facets[f"phonetic_{i}"] = float(val)
    # except Exception as e:
    #     print(f"⚠️ Phonetic encoding failed for text: {text[:50]} -> {e}")
    #     for i in range(phonetic_encoder.embedding_dim):
    #         facets[f"phonetic_{i}"] = 0.0
    return facets


import re
from textblob import TextBlob
def compute_facet_dict(text):
    words = re.findall(r'\w+', text)
    wc = len(words)
    awl = sum(len(w) for w in words) / wc if wc > 0 else 0
    sentiment = TextBlob(text).sentiment.polarity
    scramble = sum(1 for w in words if not w.isalpha()) / wc if wc > 0 else 0
    return {
        'word_count': wc,
        'avg_word_len': round(awl, 2),
        'sentiment': round(sentiment, 2),
        'scramble_ratio': round(scramble, 2)
    }