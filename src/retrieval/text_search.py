"""
Text-to-index searcher for the Where Is This? pipeline.

TextSearcher is the inverse direction of LandmarkSearcher: it accepts a
free-text query, encodes it with CLIP's text encoder, and retrieves the
closest landmark description vectors from the pre-built text FAISS index.

Because CLIP maps images and text into the same 512-d embedding space,
"baroque palace with large gardens" will score highly against the Schönbrunn
description without any explicit keyword overlap.

This module also exports _encode_text, which is reused by HybridSearcher to
avoid loading a second copy of the CLIP model.
"""

import sys
from pathlib import Path

import numpy as np
import open_clip
import torch

sys.path.insert(0, str(Path(__file__).parents[2]))

from config import CLIP_MODEL, CLIP_PRETRAINED, CONFIDENCE_THRESHOLD, DATA_DIR, FAISS_TEXT_INDEX_PATH, TOP_K
from src.retrieval.index import load_index
from src.retrieval.search import SearchResult, _load_metadata


def _encode_text(query: str, model, tokenizer, device: str) -> np.ndarray:
    """
    Encode a single text query to a 512-d L2-normalised vector.
    Shared by TextSearcher and HybridSearcher so only one CLIP model is loaded.
    """
    tokens = tokenizer([query]).to(device)
    with torch.no_grad():
        emb = model.encode_text(tokens)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy()[0].astype(np.float32)


class TextSearcher:
    """
    Encode a free-text query with CLIP and retrieve the nearest landmarks
    from the pre-built text-embedding FAISS index.

    The index contains one 512-d vector per landmark, computed from the
    landmark's Wikipedia description by scripts/build_text_index.py.
    Text-to-text cosine scores are typically 0.28-0.35 for correct matches
    (lower than image-to-image due to the cross-modal gap).
    """

    def __init__(self, device: str = "cpu"):
        self._index = load_index(FAISS_TEXT_INDEX_PATH)
        self._metadata = _load_metadata(DATA_DIR / "landmarks.json")
        # Load CLIP model for encoding text queries at search time
        model, _, _ = open_clip.create_model_and_transforms(
            CLIP_MODEL, pretrained=CLIP_PRETRAINED
        )
        self._tokenizer = open_clip.get_tokenizer(CLIP_MODEL)
        model.eval()
        model.to(device)
        self._model = model
        self._device = device

    def search(
        self,
        query: str,
        top_k: int = TOP_K,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
    ) -> list[SearchResult]:
        emb = _encode_text(query, self._model, self._tokenizer, self._device)
        scores, indices = self._index.search(
            emb.reshape(1, -1), min(top_k, self._index.ntotal)
        )
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
