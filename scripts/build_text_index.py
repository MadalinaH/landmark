"""
Orchestrator: embed all landmark descriptions → build a FAISS text-embedding index.

This is the primary retrieval index used for cross-modal search
(query image vs. pre-computed text embeddings from Wikipedia descriptions).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from config import DATA_DIR
from src.embeddings.text_encoder import build_text_embeddings, save_text_embeddings
from src.retrieval.index import build_index, save_index


def main() -> None:
    print("=== Step 1: Building text embeddings from descriptions ===")
    embeddings, metadata = build_text_embeddings()
    save_text_embeddings(embeddings, metadata)

    print("\n=== Step 2: Building FAISS text index ===")
    index = build_index(embeddings)
    save_index(index, DATA_DIR / "faiss_text_index.bin")

    print(f"\nDone. Text index contains {index.ntotal} landmarks.")


if __name__ == "__main__":
    main()
