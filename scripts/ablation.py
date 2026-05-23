"""
Answers two questions:
  1. Does fine-tuning help?   → IMAGE ablation (leave-one-out, re-embedded on-the-fly)
  2. Does BM25 help?          → TEXT ablation (CLIP-only vs Hybrid on landmark-name queries)

──────────────────────────────────────────────────────────────────────────
IMAGE ABLATION
  Re-embeds every landmark from scratch with the currently loaded CLIP model,
  builds an in-memory FAISS index, and runs leave-one-out evaluation: the last
  image in each landmark folder is used as the query.  Because both the query
  and the index are encoded with the same model, this gives a fair comparison
  between baseline and fine-tuned weights.

TEXT ABLATION
  Uses each landmark's name as a text query against the pre-built text FAISS
  index, comparing CLIP-only retrieval vs Hybrid (CLIP + BM25).  The index was
  built with the baseline model, so this measures the BM25 contribution
  independently of model weights.

──────────────────────────────────────────────────────────────────────────
Run twice, then compare:
  # Step 1: baseline (no fine-tuned weights)
  python scripts/ablation.py

  # Step 2: fine-tuned weights
  CLIP_WEIGHTS_PATH=data/checkpoints/clip_finetuned_best.pt python scripts/ablation.py

  # Comparison table is printed automatically when both result files exist.
  # To print it without re-running the evaluation:
  python scripts/ablation.py --compare-only

  # GPU (much faster for re-embedding):
  python scripts/ablation.py --device cuda

Outputs:
  evaluation/ablation_baseline.json    - baseline evaluation results
  evaluation/ablation_finetuned.json   - fine-tuned evaluation results
  evaluation/ablation_table.md         - markdown comparison table (when both exist)
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
import warnings
from collections import defaultdict
from pathlib import Path

import faiss
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))

from config import CLIP_WEIGHTS_PATH, DATA_DIR, IMAGES_DIR
from src.embeddings.image_encoder import _load_model, embed_landmark_folder
from src.utils import sanitize_folder_name

OUTPUT_DIR = Path(__file__).parents[1] / "evaluation"
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


# Helpers
def _normalise(s: str) -> str:
    """ASCII-fold and lowercase for fuzzy name matching."""
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()


def _is_match(expected: str, got: str) -> bool:
    """True when one name is a substring of the other (handles 'Colosseum' vs 'Colosseum, Rome')."""
    a, b = _normalise(expected), _normalise(got)
    return a in b or b in a


def bootstrap_ci(
    values: list[float], n_boot: int = 10_000, ci: float = 95.0
) -> tuple[float, float]:
    """Return (lower, upper) bootstrap confidence interval for the mean of values."""
    arr = np.array(values, dtype=float)
    boots = np.array(
        [np.mean(np.random.choice(arr, len(arr), replace=True)) for _ in range(n_boot)]
    )
    lo = (100.0 - ci) / 2.0
    return float(np.percentile(boots, lo)), float(np.percentile(boots, 100.0 - lo))


# Image ablation
def _build_fresh_index(
    metadata: list[dict], model, preprocess, device: str
) -> tuple[faiss.Index, list[dict]]:
    """
    Re-embed all landmarks from disk with the given model.
    Returns (faiss_index, valid_metadata) - landmarks with no images are dropped.
    """
    embeddings: list[np.ndarray] = []
    valid_meta: list[dict] = []

    for lm in metadata:
        folder = IMAGES_DIR / sanitize_folder_name(lm["name"])
        if not folder.exists():
            continue
        emb = embed_landmark_folder(folder, model, preprocess, device)
        if emb is None:
            continue
        embeddings.append(emb.astype(np.float32))
        valid_meta.append(lm)

    if not embeddings:
        raise RuntimeError("No embeddings produced - are images in data/images/?")

    arr = np.stack(embeddings)  # (N, 512)
    index = faiss.IndexFlatIP(arr.shape[1])
    index.add(arr)
    return index, valid_meta


def run_image_ablation(device: str = "cpu") -> list[dict]:
    """
    Leave-one-out image retrieval evaluation.
    Both query and index are encoded with the currently loaded CLIP model,
    so baseline vs fine-tuned comparisons are always apples-to-apples.

    Returns a list of per-landmark dicts:
      name, region, hit1, hit3, rr, score, top1_name, top1_wrong
    """
    metadata = json.loads((DATA_DIR / "metadata.json").read_text())
    model, preprocess = _load_model(device)

    print(
        f"  Re-embedding {len(metadata)} landmarks on-the-fly "
        f"({'fine-tuned' if CLIP_WEIGHTS_PATH else 'baseline'} weights)..."
    )
    index, valid_meta = _build_fresh_index(metadata, model, preprocess, device)
    print(f"  Done - {len(valid_meta)} landmarks in index.")

    results: list[dict] = []

    for lm in valid_meta:
        folder = IMAGES_DIR / sanitize_folder_name(lm["name"])
        images = sorted(
            p for p in folder.iterdir() if p.suffix.lower() in _IMAGE_EXTENSIONS
        )
        if not images:
            continue

        # Last image is the leave-one-out query (consistent with bias_audit.py)
        query_img = images[-1]
        try:
            from src.embeddings.image_encoder import embed_single_image

            query_emb = embed_single_image(query_img, model, preprocess, device)
        except Exception as e:
            warnings.warn(f"Skipping {query_img}: {e}")
            continue

        scores, indices = index.search(
            query_emb.reshape(1, -1).astype(np.float32), min(3, index.ntotal)
        )

        hit1, hit3, rr = 0, 0, 0.0
        top1_name = valid_meta[indices[0][0]]["name"] if indices[0][0] >= 0 else ""
        top1_score = float(scores[0][0]) if indices[0][0] >= 0 else 0.0

        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < 0:
                continue
            if _is_match(lm["name"], valid_meta[idx]["name"]):
                if rank == 0:
                    hit1 = 1
                hit3 = 1
                rr = 1.0 / (rank + 1)
                break  # found the correct match; stop

        results.append(
            {
                "name": lm["name"],
                "region": lm.get("region", "Unknown"),
                "hit1": hit1,
                "hit3": hit3,
                "rr": rr,
                "score": top1_score,
                "top1_name": top1_name,
            }
        )

    return results


def _image_summary(results: list[dict]) -> dict:
    n = len(results)
    hits1 = [r["hit1"] for r in results]
    hits3 = [r["hit3"] for r in results]
    rrs = [r["rr"] for r in results]
    correct_scores = [r["score"] for r in results if r["hit1"]]
    ci_lo, ci_hi = bootstrap_ci(hits1)
    return {
        "n": n,
        "hits1": int(sum(hits1)),
        "hits3": int(sum(hits3)),
        "hits1_pct": float(np.mean(hits1)),
        "hits3_pct": float(np.mean(hits3)),
        "mrr": float(np.mean(rrs)),
        "mean_correct_score": float(np.mean(correct_scores)) if correct_scores else 0.0,
        "ci_95_lo": ci_lo,
        "ci_95_hi": ci_hi,
    }


# Text ablation
def run_text_ablation(device: str = "cpu") -> dict:
    """
    Compare CLIP-only vs Hybrid (CLIP + BM25) text retrieval.

    Query set: all 149 landmark names - each name should retrieve itself.
    Uses the pre-built text FAISS index (landmarks.json descriptions).

    Note: the index was built with the baseline CLIP model, so this comparison
    isolates the BM25 contribution rather than the effect of fine-tuning.
    """
    from src.retrieval.hybrid_search import HybridSearcher
    from src.retrieval.text_search import TextSearcher

    landmarks = json.loads((DATA_DIR / "landmarks.json").read_text())

    print("  Loading TextSearcher (CLIP-only)...")
    ts = TextSearcher(device=device)
    print("  Loading HybridSearcher (CLIP + BM25)...")
    hs = HybridSearcher(device=device)

    clip_results: list[dict] = []
    hybrid_results: list[dict] = []

    for lm in landmarks:
        query = lm["name"]

        clip_hits = ts.search(query, top_k=3, confidence_threshold=0.0)
        hybrid_hits = hs.search(query, top_k=3, confidence_threshold=0.0)

        clip_results.append(_score_text_hits(clip_hits, lm["name"]))
        hybrid_results.append(_score_text_hits(hybrid_hits, lm["name"]))

    return {
        "note": (
            "Queries are landmark names against the pre-built text index. "
            "Compares retrieval strategy only (CLIP-only vs Hybrid BM25); "
            "does not reflect fine-tuned model weights."
        ),
        "clip_only": _text_summary(clip_results),
        "hybrid": _text_summary(hybrid_results),
        "per_landmark": [
            {
                "name": lm["name"],
                "clip_hit1": c["hit1"],
                "hybrid_hit1": h["hit1"],
                "clip_rr": c["rr"],
                "hybrid_rr": h["rr"],
            }
            for lm, c, h in zip(landmarks, clip_results, hybrid_results)
        ],
    }


def _score_text_hits(hits, expected_name: str) -> dict:
    hit1, hit3, rr = 0, 0, 0.0
    for rank, r in enumerate(hits[:3]):
        if _is_match(expected_name, r.name):
            if rank == 0:
                hit1 = 1
            hit3 = 1
            rr = 1.0 / (rank + 1)
            break
    return {"hit1": hit1, "hit3": hit3, "rr": rr}


def _text_summary(results: list[dict]) -> dict:
    n = len(results)
    hits1 = [r["hit1"] for r in results]
    hits3 = [r["hit3"] for r in results]
    rrs = [r["rr"] for r in results]
    ci_lo, ci_hi = bootstrap_ci(hits1)
    return {
        "n": n,
        "hits1": int(sum(hits1)),
        "hits3": int(sum(hits3)),
        "hits1_pct": float(np.mean(hits1)),
        "hits3_pct": float(np.mean(hits3)),
        "mrr": float(np.mean(rrs)),
        "ci_95_lo": ci_lo,
        "ci_95_hi": ci_hi,
    }


# Failure analysis
def print_failures(image_results: list[dict], label: str) -> None:
    failures = [r for r in image_results if not r["hit1"]]
    if not failures:
        print(f"\n  [{label}] No image retrieval failures - perfect Hits@1!")
        return

    by_region: dict[str, list] = defaultdict(list)
    for r in failures:
        by_region[r["region"]].append(r)

    print(f"\n  [{label}] Image retrieval failures ({len(failures)}/{len(image_results)}):")
    for region in sorted(by_region):
        print(f"\n    {region}:")
        for r in by_region[region]:
            marker = "✗"
            print(
                f"      {marker} {r['name']!r:<42} "
                f"→ retrieved {r['top1_name']!r}  (score {r['score']:.3f})"
            )


# Printing and saving tables
def _fmt_pct(hits: int, n: int, pct: float) -> str:
    return f"{hits}/{n} ({pct:.0%})"


def print_image_table(baseline: dict | None, finetuned: dict | None) -> None:
    print("\n" + "═" * 78)
    print("  IMAGE RETRIEVAL - leave-one-out with on-the-fly re-embedding")
    print("═" * 78)
    print(
        f"  {'Condition':<22} {'N':>4}  {'Hits@1':>14}  {'Hits@3':>14}  "
        f"{'MRR':>6}  {'Mean score':>10}  {'95% CI (H@1)':>14}"
    )
    print("  " + "─" * 74)

    def row(label, s):
        ci = f"[{s['ci_95_lo']:.2f}, {s['ci_95_hi']:.2f}]"
        return (
            f"  {label:<22} {s['n']:>4}  "
            f"{_fmt_pct(s['hits1'], s['n'], s['hits1_pct']):>14}  "
            f"{_fmt_pct(s['hits3'], s['n'], s['hits3_pct']):>14}  "
            f"{s['mrr']:.3f}  {s['mean_correct_score']:>10.3f}  {ci:>14}"
        )

    if baseline:
        print(row("Baseline CLIP", baseline))
    if finetuned:
        print(row("Fine-tuned CLIP", finetuned))

    if baseline and finetuned:
        delta_h1 = finetuned["hits1_pct"] - baseline["hits1_pct"]
        delta_h3 = finetuned["hits3_pct"] - baseline["hits3_pct"]
        delta_mrr = finetuned["mrr"] - baseline["mrr"]
        delta_score = finetuned["mean_correct_score"] - baseline["mean_correct_score"]
        print(
            f"\n  Fine-tuning Δ: "
            f"Hits@1 {delta_h1:+.1%},  "
            f"Hits@3 {delta_h3:+.1%},  "
            f"MRR {delta_mrr:+.3f},  "
            f"Mean score {delta_score:+.3f}"
        )


def print_text_table(text_results: dict) -> None:
    clip = text_results["clip_only"]
    hybrid = text_results["hybrid"]
    print("\n" + "═" * 78)
    print("  TEXT RETRIEVAL - landmark name queries, strategy comparison")
    print("═" * 78)
    print(
        f"  {'Strategy':<28} {'N':>4}  {'Hits@1':>14}  {'Hits@3':>14}  "
        f"{'MRR':>6}  {'95% CI (H@1)':>14}"
    )
    print("  " + "─" * 74)

    def row(label, s):
        ci = f"[{s['ci_95_lo']:.2f}, {s['ci_95_hi']:.2f}]"
        return (
            f"  {label:<28} {s['n']:>4}  "
            f"{_fmt_pct(s['hits1'], s['n'], s['hits1_pct']):>14}  "
            f"{_fmt_pct(s['hits3'], s['n'], s['hits3_pct']):>14}  "
            f"{s['mrr']:.3f}  {ci:>14}"
        )

    print(row("CLIP-only", clip))
    print(row("Hybrid (CLIP + BM25)", hybrid))

    delta = hybrid["hits1_pct"] - clip["hits1_pct"]
    delta_mrr = hybrid["mrr"] - clip["mrr"]
    print(f"\n  BM25 Δ: Hits@1 {delta:+.1%},  MRR {delta_mrr:+.3f}")

    # Show cases where the two strategies disagree
    disagreements = [
        r
        for r in text_results["per_landmark"]
        if r["clip_hit1"] != r["hybrid_hit1"]
    ]
    if disagreements:
        print(f"\n  Strategy disagreements ({len(disagreements)} landmarks):")
        for r in disagreements:
            winner = "Hybrid" if r["hybrid_hit1"] else "CLIP-only"
            print(f"    {r['name']!r:<42} → {winner} wins")


def save_markdown_table(
    baseline_img: dict | None,
    finetuned_img: dict | None,
    text_results: dict | None,
    path: Path,
) -> None:
    lines = [
        "# Ablation Study Results\n",
        "## Image Retrieval (leave-one-out, on-the-fly re-embedding)\n",
        "| Condition | N | Hits@1 | Hits@3 | MRR | Mean score | 95% CI |",
        "|-----------|---|--------|--------|-----|------------|--------|",
    ]

    def img_row(label, s):
        ci = f"[{s['ci_95_lo']:.2f}, {s['ci_95_hi']:.2f}]"
        return (
            f"| {label} | {s['n']} "
            f"| {_fmt_pct(s['hits1'], s['n'], s['hits1_pct'])} "
            f"| {_fmt_pct(s['hits3'], s['n'], s['hits3_pct'])} "
            f"| {s['mrr']:.3f} | {s['mean_correct_score']:.3f} | {ci} |"
        )

    if baseline_img:
        lines.append(img_row("Baseline CLIP", baseline_img))
    if finetuned_img:
        lines.append(img_row("Fine-tuned CLIP", finetuned_img))

    if baseline_img and finetuned_img:
        d = finetuned_img["hits1_pct"] - baseline_img["hits1_pct"]
        lines.append(f"\n**Fine-tuning Δ Hits@1: {d:+.1%}**\n")

    if text_results:
        clip = text_results["clip_only"]
        hybrid = text_results["hybrid"]
        lines += [
            "\n## Text Retrieval - Strategy Comparison (landmark name queries)\n",
            "| Strategy | N | Hits@1 | Hits@3 | MRR | 95% CI |",
            "|----------|---|--------|--------|-----|--------|",
            (
                f"| CLIP-only | {clip['n']} "
                f"| {_fmt_pct(clip['hits1'], clip['n'], clip['hits1_pct'])} "
                f"| {_fmt_pct(clip['hits3'], clip['n'], clip['hits3_pct'])} "
                f"| {clip['mrr']:.3f} "
                f"| [{clip['ci_95_lo']:.2f}, {clip['ci_95_hi']:.2f}] |"
            ),
            (
                f"| Hybrid (CLIP + BM25) | {hybrid['n']} "
                f"| {_fmt_pct(hybrid['hits1'], hybrid['n'], hybrid['hits1_pct'])} "
                f"| {_fmt_pct(hybrid['hits3'], hybrid['n'], hybrid['hits3_pct'])} "
                f"| {hybrid['mrr']:.3f} "
                f"| [{hybrid['ci_95_lo']:.2f}, {hybrid['ci_95_hi']:.2f}] |"
            ),
        ]
        d = hybrid["hits1_pct"] - clip["hits1_pct"]
        lines.append(f"\n**BM25 Δ Hits@1: {d:+.1%}**\n")

    path.write_text("\n".join(lines) + "\n")
    print(f"\nSaved markdown table → {path}")


# Main
def main() -> None:
    parser = argparse.ArgumentParser(description="Ablation study for Where Is This?")
    parser.add_argument(
        "--device",
        default="cuda" if __import__("torch").cuda.is_available() else "cpu",
        help="cuda or cpu",
    )
    parser.add_argument(
        "--compare-only",
        action="store_true",
        help="Print comparison table from saved JSON files without re-running evaluation.",
    )
    parser.add_argument(
        "--skip-text",
        action="store_true",
        help="Skip text ablation (faster; useful when you only changed image weights).",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)

    label = "finetuned" if CLIP_WEIGHTS_PATH else "baseline"
    result_path = OUTPUT_DIR / f"ablation_{label}.json"

    # Compare-only mode
    if args.compare_only:
        baseline_path = OUTPUT_DIR / "ablation_baseline.json"
        finetuned_path = OUTPUT_DIR / "ablation_finetuned.json"
        if not baseline_path.exists() and not finetuned_path.exists():
            print("No result files found. Run the ablation first.")
            return
        baseline = json.loads(baseline_path.read_text()) if baseline_path.exists() else None
        finetuned = json.loads(finetuned_path.read_text()) if finetuned_path.exists() else None

        b_img = baseline["image_summary"] if baseline else None
        f_img = finetuned["image_summary"] if finetuned else None
        text = (baseline or finetuned).get("text_ablation")

        print_image_table(b_img, f_img)
        if text:
            print_text_table(text)
        save_markdown_table(b_img, f_img, text, OUTPUT_DIR / "ablation_table.md")
        return

    # Image ablation
    print(f"\n[1/2] Image retrieval ablation ({label} weights, device={args.device})")
    image_results = run_image_ablation(device=args.device)
    image_summary = _image_summary(image_results)

    print(f"\n  Hits@1 : {image_summary['hits1']}/{image_summary['n']} ({image_summary['hits1_pct']:.0%})")
    print(f"  Hits@3 : {image_summary['hits3']}/{image_summary['n']} ({image_summary['hits3_pct']:.0%})")
    print(f"  MRR    : {image_summary['mrr']:.3f}")
    print(f"  95% CI : [{image_summary['ci_95_lo']:.2f}, {image_summary['ci_95_hi']:.2f}]")

    print_failures(image_results, label)

    # Text ablation
    text_results = None
    if not args.skip_text:
        print(f"\n[2/2] Text retrieval ablation (strategy comparison, device={args.device})")
        text_results = run_text_ablation(device=args.device)
        print_text_table(text_results)
    else:
        print("\n[2/2] Text ablation skipped (--skip-text).")

    # Save results
    output = {
        "label": label,
        "clip_weights_path": str(CLIP_WEIGHTS_PATH) if CLIP_WEIGHTS_PATH else None,
        "image_summary": image_summary,
        "image_per_landmark": image_results,
        "text_ablation": text_results,
    }
    result_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\nSaved results → {result_path}")

    # Print comparison table if both conditions have been run
    baseline_path = OUTPUT_DIR / "ablation_baseline.json"
    finetuned_path = OUTPUT_DIR / "ablation_finetuned.json"

    if baseline_path.exists() and finetuned_path.exists():
        baseline = json.loads(baseline_path.read_text())
        finetuned = json.loads(finetuned_path.read_text())

        print_image_table(baseline["image_summary"], finetuned["image_summary"])

        if text_results:
            print_text_table(text_results)

        save_markdown_table(
            baseline["image_summary"],
            finetuned["image_summary"],
            text_results or baseline.get("text_ablation"),
            OUTPUT_DIR / "ablation_table.md",
        )
    else:
        missing = "baseline" if not baseline_path.exists() else "finetuned"
        print(
            f"\nComparison table: waiting for {missing} results. "
            f"Run again with {'no' if missing == 'baseline' else 'CLIP_WEIGHTS_PATH set'} "
            f"fine-tuned weights."
        )


if __name__ == "__main__":
    main()
