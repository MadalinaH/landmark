"""
FAISS index builder and loader for the Where Is This? pipeline.

Uses IndexFlatIP (flat inner-product index) which computes exact cosine
similarity when all stored vectors are L2-normalised.  Flat indices perform
an exhaustive scan over all N vectors at query time - this is fast enough
for N=149 landmarks on CPU and guarantees no approximation error.

Two separate indexes are maintained:
  faiss_index.bin       - image embeddings (one averaged vector per landmark)
  faiss_text_index.bin  - text embeddings (one description vector per landmark)
"""

import sys
from pathlib import Path

import faiss
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2]))
from config import DATA_DIR, EMBEDDING_DIM

_INDEX_PATH = DATA_DIR / "faiss_index.bin"


def build_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """
    Build a FAISS inner-product index from pre-normalised embeddings.
    Vectors must already be L2-normalised: inner product then equals cosine similarity.
    """
    assert (
        embeddings.ndim == 2 and embeddings.shape[1] == EMBEDDING_DIM
    ), f"Expected shape (N, {EMBEDDING_DIM}), got {embeddings.shape}"
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(embeddings)
    return index


def save_index(index: faiss.IndexFlatIP, path: Path = _INDEX_PATH) -> None:
    faiss.write_index(index, str(path))
    print(f"Saved FAISS index ({index.ntotal} vectors) → {path}")


def load_index(path: Path = _INDEX_PATH) -> faiss.IndexFlatIP:
    if not path.exists():
        raise FileNotFoundError(
            f"FAISS index not found at {path}. "
            "Run `uv run python scripts/build_text_index.py` (primary) "
            "or `uv run python scripts/build_index.py` (image index)."
        )
    return faiss.read_index(str(path))
