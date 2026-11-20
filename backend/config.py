"""Backend settings. Each value can be overridden with an environment variable."""

import os
from pathlib import Path

# Folders
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"            # PDF documents to ingest
IMAGES_DIR = DATA_DIR / "images"        # images extracted from the PDFs

# Vector database
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.getenv("QDRANT_COLLECTION", "emsi")

# Embedding models (downloaded from Hugging Face on first use)
TEXT_MODEL = os.getenv("TEXT_MODEL", "BAAI/bge-m3")
TEXT_DIM = 1024
CLIP_IMAGE_MODEL = os.getenv("CLIP_IMAGE_MODEL", "clip-ViT-B-32")
CLIP_TEXT_MODEL = os.getenv(
    "CLIP_TEXT_MODEL", "sentence-transformers/clip-ViT-B-32-multilingual-v1"
)
IMAGE_DIM = 512

# Vision-language model served by Ollama
VLM_MODEL = os.getenv("VLM_MODEL", "qwen3-vl:4b")

# Ingestion parameters
CHUNK_TOKENS = 300       # size of a text chunk, in tokens
CHUNK_OVERLAP = 50       # tokens shared by two consecutive chunks
MIN_IMAGE_SIZE = 150     # images smaller than this (in pixels) are skipped
MIN_CHUNK_WORDS = 10     # shorter chunks (titles, page numbers) are skipped

# Retrieval parameters
RRF_K = 2                  # small k: a top rank in one list beats a low rank in two lists
IMAGE_SEARCH_WEIGHT = 0.2  # weight of the question-to-image search in the fusion

# Answer generation
TOP_K = 8                  # excerpts given to the model for each question
MAX_CONTEXT_IMAGES = 1     # retrieved images the model also looks at
CONTEXT_WINDOW = 8192      # tokens the model can read at once (prompt + answer)
