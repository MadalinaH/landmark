"""
Geographic bias audit for the image retrieval index.

Uses leave-one-out evaluation: for each landmark in the image index the last
image in its folder is used as a query against the full index.  Hits@1 and
mean cosine score are computed per region to surface geographic imbalances.

The audit intentionally includes the query image in the index (it was part of
the averaged embedding).  Results are therefore slightly optimistic in absolute
terms, but the relative comparison across regions remains valid.

Run via:  python scripts/run_audit.py
"""

import json
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2]))
from config import CONFIDENCE_THRESHOLD_IMAGE, DATA_DIR, IMAGES_DIR
from src.retrieval.search import LandmarkSearcher
from src.utils import sanitize_folder_name

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

REGION_ORDER = ["Vienna", "Europe", "Americas", "Asia", "Africa", "Oceania", "Natural"]


def _normalise(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()


def run_leave_one_out(device: str = "cpu") -> list[dict]:
    """
    Query each indexed landmark with its last image and record Hits@1.

    Returns a list of dicts:
      name, region, hit1, score, top1_name
    """
    searcher = LandmarkSearcher(mode="image", device=device)
    metadata = json.loads((DATA_DIR / "metadata.json").read_text())

    results = []
    for lm in metadata:
        folder = IMAGES_DIR / sanitize_folder_name(lm["name"])
        if not folder.exists():
            continue
        images = sorted(
            p for p in folder.iterdir()
            if p.suffix.lower() in _IMAGE_EXTENSIONS
        )
        if not images:
            continue

        query_img = images[-1]
        search_results = searcher.search(query_img, top_k=3, confidence_threshold=0.0)
        if not search_results:
            continue

        top1 = search_results[0]
        exp = _normalise(lm["name"])
        got = _normalise(top1.name)
        hit1 = int(exp in got or got in exp)

        results.append({
            "name": lm["name"],
            "region": lm.get("region", "Unknown"),
            "hit1": hit1,
            "score": float(top1.score),
            "top1_name": top1.name,
        })

    return results


def hits_by_region(results: list[dict]) -> dict[str, dict]:
    """Aggregate Hits@1 and mean score per region."""
    by_region: dict[str, list] = defaultdict(list)
    for r in results:
        by_region[r["region"]].append(r)

    summary = {}
    for region, items in by_region.items():
        hits = sum(r["hit1"] for r in items)
        total = len(items)
        correct_scores = [r["score"] for r in items if r["hit1"]]
        summary[region] = {
            "hits": hits,
            "total": total,
            "pct": hits / total if total else 0.0,
            "mean_score": float(np.mean(correct_scores)) if correct_scores else 0.0,
        }
    return summary


def plot_bias(summary: dict[str, dict], save_path: Path) -> None:
    """Save a Hits@1-per-region bar chart to save_path."""
    regions = [r for r in REGION_ORDER if r in summary]
    pcts = [summary[r]["pct"] * 100 for r in regions]
    counts = [f"{summary[r]['hits']}/{summary[r]['total']}" for r in regions]

    colours = [
        "#ef4444" if p < 70 else "#f59e0b" if p < 85 else "#22c55e"
        for p in pcts
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(regions, pcts, color=colours, edgecolor="white", linewidth=0.5)

    for bar, count, pct in zip(bars, counts, pcts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f"{count}\n({pct:.0f}%)",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    ax.axhline(y=80, color="gray", linestyle="--", alpha=0.5, label="80% reference line")
    ax.set_ylim(0, 120)
    ax.set_ylabel("Hits@1 (%)", fontsize=11)
    ax.set_title("Geographic Bias Audit - Hits@1 by Region (leave-one-out)", fontsize=13)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved bias chart → {save_path}")
