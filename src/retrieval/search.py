"""
Image-to-index searcher for the Where Is This? pipeline.

LandmarkSearcher accepts an image file path, encodes it with CLIP, and
retrieves the closest landmark vectors from a pre-built FAISS index.

Two retrieval modes are supported:

  mode="image"  - query image vs. pre-computed image embeddings (IndexFlatIP
                  over data/faiss_index.bin).  Used in the Image search tab.
                  Scores are image-to-image cosine similarities (0.85-0.94
                  for correct matches on the golden set).

  mode="text"   - query image vs. pre-computed text embeddings (IndexFlatIP
                  over data/faiss_text_index.bin).  Cross-modal retrieval:
                  CLIP maps images and their textual descriptions to nearby
                  vectors in the same 512-d space.  Scores are lower (0.28-0.35
                  for correct matches) because of the modality gap.

The Streamlit app uses mode="image" for the image tab because image-to-image
retrieval is empirically more accurate on the golden set.
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2]))
from config import CONFIDENCE_THRESHOLD, DATA_DIR, TOP_K
from src.embeddings.image_encoder import _load_model, embed_single_image
from src.retrieval.index import load_index


@dataclass
class SearchResult:
    """Single retrieval result returned by all searcher classes."""

    name: str
    region: str
    description: str
    lat: float | None
    lon: float | None
    score: float  # cosine similarity in [0, 1]
    low_confidence: bool  # True when score is below the calibrated threshold


def _load_metadata(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"Metadata not found at {path}. "
            "Run `uv run python scripts/build_text_index.py` first."
        )
    return json.loads(path.read_text())


class LandmarkSearcher:
    """
    Encode a query image with CLIP and retrieve the nearest landmarks from a
    pre-built FAISS index.  The mode parameter selects which index to query.
    """

    def __init__(self, mode: str = "text", device: str = "cpu"):
        if mode == "text":
            index_path = DATA_DIR / "faiss_text_index.bin"
            # Text index covers every landmark in landmarks.json - no separate metadata needed
            meta_path = DATA_DIR / "landmarks.json"
        elif mode == "image":
            index_path = DATA_DIR / "faiss_index.bin"
            # Image index may skip landmarks with missing folders, so it has its own metadata
            meta_path = DATA_DIR / "metadata.json"
        else:
            raise ValueError(f"mode must be 'text' or 'image', got {mode!r}")

        self._index = load_index(index_path)
        self._metadata = _load_metadata(meta_path)
        self._model, self._preprocess = _load_model(device)
        self._device = device
        self.mode = mode

    def search(
        self,
        image_path: Path,
        top_k: int = TOP_K,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
    ) -> list[SearchResult]:
        """
        Embed the query image and return the top_k closest landmarks.
        Results whose cosine score is below confidence_threshold are flagged
        as low_confidence so the UI can warn the user.
        """
        emb = embed_single_image(
            image_path, self._model, self._preprocess, self._device
        )
        query = emb.reshape(1, -1).astype(np.float32)

        scores, indices = self._index.search(query, min(top_k, self._index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            lm = self._metadata[idx]
            results.append(
                SearchResult(
                    name=lm["name"],
                    region=lm.get("region", ""),
                    description=lm.get("description", ""),
                    lat=lm.get("lat"),
                    lon=lm.get("lon"),
                    score=float(score),
                    low_confidence=float(score) < confidence_threshold,
                )
            )

        return results
