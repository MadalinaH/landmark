"""
Golden-set evaluation (§3.3).

Runs the 5 golden queries against the text-embedding index and reports
Hits@1, Hits@3, MRR, and per-query confidence scores.

Golden images must be placed in: data/eval/
  schoenbrunn_test.jpg   → Schönbrunn Palace
  eiffel_test.jpg        → Eiffel Tower
  colosseum_test.jpg     → Colosseum
  sagrada_test.jpg       → Sagrada Família
  stephansdom_test.jpg   → Stephansdom
"""

import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from config import DATA_DIR
from src.retrieval.search import LandmarkSearcher

EVAL_DIR = DATA_DIR.parent / "data" / "eval"

GOLDEN_SET = [
    ("schoenbrunn_test.jpg", "Schönbrunn Palace"),
    ("eiffel_test.jpg", "Eiffel Tower"),
    ("colosseum_test.jpg", "Colosseum"),
    ("sagrada_test.jpg", "Sagrada Família"),
    ("stephansdom_test.jpg", "Stephansdom"),
]


def _normalise(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()


def reciprocal_rank(results, expected_name: str) -> float:
    exp = _normalise(expected_name)
    for rank, r in enumerate(results, start=1):
        name = _normalise(r.name)
        if exp in name or name in exp:
            return 1.0 / rank
    return 0.0


def main() -> None:
    eval_dir = DATA_DIR.parent / "data" / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    missing = [f for f, _ in GOLDEN_SET if not (eval_dir / f).exists()]
    if missing:
        print("Missing golden images:")
        for f in missing:
            print(f"  {eval_dir / f}")
        print("\nPlace the golden test images in data/eval/ and re-run.")
        sys.exit(1)

    print("Loading searcher (image-to-image index)…")
    searcher = LandmarkSearcher(mode="image")

    hits1 = hits3 = 0
    mrr_total = 0.0

    print(
        f"\n{'Query':<30} {'Expected':<30} "
        f"{'Top-1 Result':<30} {'Score':>6}  H@1  H@3  RR"
    )
    print("-" * 110)

    for filename, expected in GOLDEN_SET:
        img_path = eval_dir / filename
        results = searcher.search(img_path, top_k=3, confidence_threshold=0.0)

        top1 = results[0] if results else None
        top1_name = top1.name if top1 else "—"
        top1_score = top1.score if top1 else 0.0

        rr = reciprocal_rank(results, expected)
        h1 = int(rr == 1.0)
        h3 = int(rr > 0.0)

        hits1 += h1
        hits3 += h3
        mrr_total += rr

        print(
            f"{filename:<30} {expected:<30} {top1_name:<30} {top1_score:>6.3f}  "
            f"{'✓' if h1 else '✗':>3}  {'✓' if h3 else '✗':>3}  {rr:.3f}"
        )

    n = len(GOLDEN_SET)
    print("-" * 110)
    print(f"\nResults over {n} queries:")
    print(f"  Hits@1 : {hits1}/{n} = {hits1/n:.0%}")
    print(f"  Hits@3 : {hits3}/{n} = {hits3/n:.0%}")
    print(f"  MRR    : {mrr_total/n:.3f}")


if __name__ == "__main__":
    main()
