"""
backend/embedding_model.py
──────────────────────────
Bio_ClinicalBERT loaded ONCE as a module-level singleton.
Every call to get_embedding() reuses the same tokenizer + model,
so FastAPI workers don't each pay the ~400 MB load penalty.

Thread-safe: torch.no_grad() + no mutable state between calls.
"""
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import logging
from functools import lru_cache

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)

_MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
_tokenizer: AutoTokenizer | None = None
_model:     AutoModel     | None = None


def _load_model() -> tuple[AutoTokenizer, AutoModel]:
    """Lazy-load the tokenizer and model exactly once."""
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        logger.info("Loading Bio_ClinicalBERT (first call only)…")
        _tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
        _model     = AutoModel.from_pretrained(_MODEL_NAME)
        _model.eval()                          # inference mode
        torch.set_num_threads(1)               # prevent OMP contention
        logger.info("Bio_ClinicalBERT loaded.")
    return _tokenizer, _model


def get_embedding(text: str) -> np.ndarray:
    """
    Return a float32 numpy vector (shape: [768]) for *text*.

    Truncates to 512 tokens (BERT hard limit).
    Uses mean-pooling over the last hidden state.
    """
    if not text or not text.strip():
        raise ValueError("get_embedding() received empty text.")

    tokenizer, model = _load_model()

    tokens = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
    )

    with torch.no_grad():
        output = model(**tokens)

    # Mean-pool over token dimension → shape [768]
    embedding = output.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embedding.astype("float32")