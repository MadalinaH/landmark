"""
Confidence calibration visualisation.

Plots two overlapping score histograms - correct vs incorrect matches - with a
vertical line at the calibrated threshold.  A well-calibrated threshold sits in
the valley between the two distributions: most correct matches score above it
and most incorrect matches score below it.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2]))


def plot_calibration(
    correct_scores: list[float],
    incorrect_scores: list[float],
    threshold: float,
    save_path: Path,
    title: str = "Confidence Calibration",
) -> None:
    """
    Save a score-distribution histogram to save_path.

    correct_scores:   cosine scores where top-1 was the expected landmark
    incorrect_scores: cosine scores where top-1 was wrong
    threshold:        the CONFIDENCE_THRESHOLD_IMAGE value from config.py
    """
    fig, ax = plt.subplots(figsize=(9, 4))

    bins = np.linspace(
        min(correct_scores + incorrect_scores) - 0.02,
        max(correct_scores + incorrect_scores) + 0.02,
        30,
    )

    ax.hist(
        correct_scores, bins=bins, alpha=0.65,
        color="#22c55e", label=f"Correct (n={len(correct_scores)})",
    )
    ax.hist(
        incorrect_scores, bins=bins, alpha=0.65,
        color="#ef4444", label=f"Incorrect (n={len(incorrect_scores)})",
    )
    ax.axvline(
        x=threshold, color="black", linestyle="--", linewidth=1.5,
        label=f"Threshold = {threshold}",
    )

    # Precision / recall at threshold
    tp = sum(s >= threshold for s in correct_scores)
    fp = sum(s >= threshold for s in incorrect_scores)
    fn = sum(s < threshold for s in correct_scores)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    ax.text(
        0.98, 0.95,
        f"Precision@threshold: {precision:.2f}\nRecall@threshold: {recall:.2f}",
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    ax.set_xlabel("Cosine similarity score", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved calibration chart → {save_path}")
