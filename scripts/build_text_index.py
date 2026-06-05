"""
Orchestrator: embed all landmark descriptions → build a FAISS text-embedding index.

This is the primary retrieval index used for cross-modal search
(query image vs. pre-computed text embeddings from Wikipedia descriptions).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from config import BACKBONE, DATA_DIR, FAISS_TEXT_INDEX_PATH
from src.embeddings.text_encoder import build_text_embeddings, save_text_embeddings
from src.retrieval.index import build_index, save_index


def main() -> None:
    print(f"=== Building text index (backbone={BACKBONE}) ===")
    print("Step 1: Building text embeddings from descriptions")
    embeddings, metadata = build_text_embeddings()
    save_text_embeddings(embeddings, metadata)

    print("\nStep 2: Building FAISS text index")
    index = build_index(embeddings)
    save_index(index, FAISS_TEXT_INDEX_PATH)

    print(f"\nDone. Text index ({BACKBONE}) → {FAISS_TEXT_INDEX_PATH}")
    print(f"       Contains {index.ntotal} landmarks.")


if __name__ == "__main__":
    main()
