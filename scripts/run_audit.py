"""
Run the geographic bias audit + confidence calibration on the image index.

Outputs saved to evaluation/:
  bias_chart.png         - Hits@1 per region bar chart
  calibration_chart.png  - score distribution for correct vs incorrect matches
  audit_results.json     - raw per-landmark results

Usage:
    python scripts/run_audit.py [--device cuda]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from config import CONFIDENCE_THRESHOLD_IMAGE
from src.evaluation.bias_audit import (
    REGION_ORDER,
    hits_by_region,
    plot_bias,
    run_leave_one_out,
)
from src.evaluation.metrics import plot_calibration

OUTPUT_DIR = Path(__file__).parents[1] / "evaluation"


def main(device: str = "cpu") -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("Running leave-one-out evaluation - this takes ~2 min on CPU, seconds on GPU...")
    results = run_leave_one_out(device=device)

    (OUTPUT_DIR / "audit_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False)
    )
    print(f"Saved raw results → {OUTPUT_DIR / 'audit_results.json'}")

    # --- Bias audit ---
    summary = hits_by_region(results)
    plot_bias(summary, OUTPUT_DIR / "bias_chart.png")

    print(f"\n{'Region':<12}  {'Hits@1':>10}  {'%':>6}  {'Mean score':>12}")
    print("-" * 48)
    for region in REGION_ORDER:
        if region not in summary:
            continue
        s = summary[region]
        print(
            f"{region:<12}  {s['hits']}/{s['total']:>2}  "
            f"  {s['pct']:>5.0%}  {s['mean_score']:>12.3f}"
        )

    total_hits = sum(r["hit1"] for r in results)
    print(f"\nOverall Hits@1: {total_hits}/{len(results)} = {total_hits/len(results):.0%}")

    # --- Calibration curves ---
    correct = [r["score"] for r in results if r["hit1"]]
    incorrect = [r["score"] for r in results if not r["hit1"]]

    if incorrect:
        plot_calibration(
            correct, incorrect,
            threshold=CONFIDENCE_THRESHOLD_IMAGE,
            save_path=OUTPUT_DIR / "calibration_chart.png",
            title="Image Index: Score Distribution - Correct vs Incorrect Matches",
        )
    else:
        print("All queries correct - skipping calibration chart (no incorrect scores).")

    print(f"\nAll outputs saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    args = parser.parse_args()
    print(f"Device: {args.device}")
    main(device=args.device)
