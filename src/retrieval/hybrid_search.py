"""
Hybrid BM25 + CLIP searcher for the Where Is This? pipeline.

Problem: CLIP's text encoder captures semantic similarity well (e.g. "baroque
palace with large gardens" → Schönbrunn) but struggles to distinguish between
semantically close landmarks when the query contains a specific proper noun
(e.g. "Eiffel" should unambiguously return the Eiffel Tower).

Solution: combine CLIP cosine scores with BM25 keyword scores:

    combined = clip_weight × clip_score + (1 - clip_weight) × bm25_normalised

Default clip_weight=0.7 weights semantics heavily while letting BM25 break
ties when a keyword in the query appears verbatim in a description.

Key design decisions:
  1. Stopwords are stripped from the BM25 query before scoring.  Without this,
     common words like "place" (in "place where gladiators fought") match
     "meeting place" in Parliament House, producing a spurious high BM25 score.
  2. If the filtered BM25 query yields a near-zero max score (no keyword signal),
     the system falls back to CLIP-only to avoid adding noise from a meaningless
     BM25 distribution.
"""

import numpy as np
from rank_bm25 import BM25Okapi

# English stopwords - words that carry no landmark-specific information.
# These are stripped from BM25 queries so only content words trigger keyword boosting.
_STOPWORDS = {
    "a",
    "an",
    "the",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "could",
    "should",
    "may",
    "might",
    "shall",
    "can",
    "need",
    "dare",
    "ought",
    "in",
    "on",
    "at",
    "to",
    "for",
    "of",
    "with",
    "by",
    "from",
    "as",
    "into",
    "through",
    "during",
    "including",
    "until",
    "against",
    "among",
    "throughout",
    "despite",
    "towards",
    "upon",
    "concerning",
    "and",
    "but",
    "or",
    "nor",
    "so",
    "yet",
    "both",
    "either",
    "neither",
    "not",
    "no",
    "nor",
    "just",
    "this",
    "that",
    "these",
    "those",
    "i",
    "me",
    "my",
    "we",
    "our",
    "you",
    "your",
    "he",
    "she",
    "it",
    "its",
    "they",
    "them",
    "their",
    "what",
    "which",
    "who",
    "whom",
    "where",
    "when",
    "why",
    "how",
    "all",
    "each",
    "every",
    "both",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "than",
    "too",
    "very",
    "s",
    "t",
    "there",
    "here",
    "place",
    "located",
    "known",
    "also",
    "one",
    "two",
}

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

from config import CONFIDENCE_THRESHOLD, TOP_K
from src.retrieval.search import SearchResult
from src.retrieval.text_search import TextSearcher, _encode_text


class HybridSearcher:
    """
    BM25 + CLIP hybrid retrieval for free-text queries.

    Initialisation builds a BM25 corpus from all landmark descriptions so that
    keyword queries ("Eiffel Tower", "Colosseum gladiators") get an explicit
    boost on top of the semantic CLIP signal.

    The CLIP model and FAISS index are shared with TextSearcher to avoid loading
    duplicate model weights into memory.
    """

    def __init__(self, device: str = "cpu"):
        # Reuse TextSearcher's already-loaded model and index
        self._ts = TextSearcher(device=device)
        self._metadata = self._ts._metadata
        self._index = self._ts._index
        self._model = self._ts._model
        self._tokenizer = self._ts._tokenizer
        self._device = device

        # Build BM25 corpus: one tokenised document per landmark description
        corpus = [lm.get("description", lm["name"]) for lm in self._metadata]
        self._bm25 = BM25Okapi([doc.lower().split() for doc in corpus])

    def search(
        self,
        query: str,
        top_k: int = TOP_K,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        clip_weight: float = 0.7,
    ) -> list[SearchResult]:
        """
        Return top_k landmarks ranked by the combined CLIP + BM25 score.
        clip_weight controls the blend: 1.0 = pure CLIP, 0.0 = pure BM25.
        """
        n = self._index.ntotal
        bm25_weight = 1.0 - clip_weight

        # CLIP scores (retrieve all N landmarks, not just top_k)
        emb = _encode_text(query, self._model, self._tokenizer, self._device)
        raw_scores, raw_indices = self._index.search(emb.reshape(1, -1), n)

        # Scatter scores back to positional array indexed by metadata position
        clip_scores = np.zeros(n, dtype=np.float32)
        for score, idx in zip(raw_scores[0], raw_indices[0]):
            if idx >= 0:
                clip_scores[idx] = score

        # BM25 scores
        # Strip stopwords before scoring to prevent common words (e.g. "place",
        # "where") from matching unrelated landmark descriptions (e.g. Parliament
        # House: "meeting place of the Parliament").
        bm25_tokens = [w for w in query.lower().split() if w not in _STOPWORDS]
        if bm25_tokens:
            bm25_scores = self._bm25.get_scores(bm25_tokens).astype(np.float32)
        else:
            bm25_scores = np.zeros(n, dtype=np.float32)
        bm25_max = bm25_scores.max()

        # Fall back to CLIP-only when BM25 has no signal (near-zero max score)
        if bm25_max < 1e-6:
            combined = clip_scores
        else:
            bm25_scores = bm25_scores / bm25_max  # normalise to [0, 1]
            combined = clip_weight * clip_scores + bm25_weight * bm25_scores

        top_indices = np.argsort(combined)[::-1][:top_k]

        results = []
        for idx in top_indices:
            lm = self._metadata[int(idx)]
            score = float(combined[idx])
            results.append(
                SearchResult(
                    name=lm["name"],
                    region=lm.get("region", ""),
                    description=lm.get("description", ""),
                    lat=lm.get("lat"),
                    lon=lm.get("lon"),
                    score=score,
                    low_confidence=score < confidence_threshold,
                )
            )
        return results
