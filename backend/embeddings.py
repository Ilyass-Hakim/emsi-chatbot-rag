"""Embedding models, loaded once on first use and shared by the whole backend."""

from functools import lru_cache

import torch
from sentence_transformers import SentenceTransformer

import config

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache
def text_model() -> SentenceTransformer:
    """BGE-M3: text to a 1024-d vector (half precision on GPU to save memory)."""
    model = SentenceTransformer(config.TEXT_MODEL, device=DEVICE)
    return model.half() if DEVICE == "cuda" else model


@lru_cache
def clip_text_model() -> SentenceTransformer:
    """Multilingual CLIP text encoder: a question to a 512-d vector in image space."""
    return SentenceTransformer(config.CLIP_TEXT_MODEL, device=DEVICE)


@lru_cache
def clip_image_model() -> SentenceTransformer:
    """CLIP image encoder: an image to a 512-d vector."""
    return SentenceTransformer(config.CLIP_IMAGE_MODEL, device=DEVICE)
