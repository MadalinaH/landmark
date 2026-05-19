"""Orchestrator: embed all landmark images then build the FAISS index."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.embeddings.image_encoder import build_image_embeddings, save_embeddings
from src.retrieval.index import build_index, save_index


def main() -> None:
    print("=== Step 1: Building image embeddings ===")
    embeddings, metadata = build_image_embeddings()
    save_embeddings(embeddings, metadata)

    print("\n=== Step 2: Building FAISS index ===")
    index = build_index(embeddings)
    save_index(index)

    print(f"\nDone. Index contains {index.ntotal} landmarks.")


if __name__ == "__main__":
    main()
