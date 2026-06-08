"""
Evaluate CLIP-only vs Hybrid (CLIP + BM25) text retrieval on descriptive queries.

Unlike the standard text ablation that uses exact landmark names as queries, this
script uses 20 free-form descriptive queries - the kind a user would type when they
recognise a place visually but don't know its name.  The queries are designed so that
roughly half contain rare keywords that appear in the Wikipedia description text
(giving BM25 a lexical advantage) and half are purely visual/semantic
(where CLIP's cross-modal embedding should dominate).

This lets us answer: does BM25 help specifically when queries contain content-words
that overlap with the stored descriptions, even without knowing the landmark name?

Usage
-----
    python3 scripts/eval_descriptive_queries.py [--device cuda] [--top-k 3]

Output
------
    evaluation/descriptive_query_results.json
    Prints a per-query breakdown and summary to stdout.
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

# ---------------------------------------------------------------------------
# Ground-truth descriptive queries
# ---------------------------------------------------------------------------

# Each entry: (query_text, ground_truth_landmark_name)
# Queries are split into two groups for analysis:
#   "keyword" - contain rare domain-specific words expected in the description text
#   "visual"  - describe the landmark's visual appearance without rare keywords
#
# Ground truth is checked with the same substring-match logic as the main ablation.

QUERIES: list[tuple[str, str, str]] = [
    # ── keyword-rich: BM25 should have an edge ──────────────────────────────
    (
        "tidal island monastery in Normandy connected to the mainland by a causeway "
        "at high tide",
        "Mont Saint Michel",
        "keyword",
    ),
    (
        "ancient mud brick mosque in Mali with wooden beams protruding from its walls "
        "and a large central minaret",
        "Great Mosque of Djenné",
        "keyword",
    ),
    (
        "rock-hewn churches carved entirely from solid volcanic tuff in the "
        "Ethiopian highlands of Tigray",
        "Rock-Hewn Churches Lalibela",
        "keyword",
    ),
    (
        "monolithic granite obelisks marking the capital of a pre-Christian Aksumite "
        "empire in northern Ethiopia",
        "Aksum Obelisk Ethiopia",
        "keyword",
    ),
    (
        "ancient Zoroastrian royal city in modern-day Iran that served as the "
        "ceremonial capital of the Achaemenid Empire",
        "Persepolis Iran",
        "keyword",
    ),
    (
        "white travertine terraces formed by calcium-carbonate-rich thermal springs "
        "cascading down a hillside in Anatolia",
        "Pamukkale Turkey",
        "keyword",
    ),
    (
        "medieval island prison in San Francisco Bay that once held Al Capone and "
        "other high-security federal inmates",
        "Alcatraz Island",
        "keyword",
    ),
    (
        "Roman city buried under volcanic ash and pumice in 79 AD that preserved "
        "streets and buildings intact beneath the debris",
        "Pompeii",
        "keyword",
    ),
    (
        "ancient granite citadel on a sheer rock fortress in Sri Lanka said to have "
        "been built by King Kashyapa with lion-paw gates at the entrance",
        "Sigiriya Sri Lanka",
        "keyword",
    ),
    (
        "giant gold-leaf-covered Buddhist stupa in Yangon Myanmar considered the "
        "holiest pagoda in the country",
        "Shwedagon Pagoda",
        "keyword",
    ),
    # ── visual / semantic: CLIP's cross-modal embedding should dominate ──────
    (
        "ancient stepped pyramid with feathered serpent carvings at the base of its "
        "staircase in the Yucatán Peninsula",
        "Chichen Itza",
        "visual",
    ),
    (
        "white marble mausoleum flanked by four slender minarets and perfectly "
        "reflected in a long rectangular pool",
        "Taj Mahal",
        "visual",
    ),
    (
        "ancient temple ruins on a rocky hilltop overlooking a Mediterranean city "
        "with columns still standing",
        "Acropolis of Athens",
        "visual",
    ),
    (
        "iron lattice tower built for a world exposition rising from a river-side "
        "park in the capital of France",
        "Eiffel Tower",
        "visual",
    ),
    (
        "ancient rose-red city carved entirely from sandstone cliffs in a Jordanian "
        "desert canyon with a narrow gorge entrance",
        "Petra Jordan",
        "visual",
    ),
    (
        "circular terraced Buddhist monument covered in stupas and Buddha statues on "
        "a hilltop in central Java Indonesia",
        "Borobudur",
        "visual",
    ),
    (
        "giant outdoor Christ statue with outstretched arms on a mountain peak "
        "overlooking a coastal city in South America",
        "Christ the Redeemer Rio",
        "visual",
    ),
    (
        "large glass pyramid entrance set in the courtyard of a classical French "
        "royal palace now used as an art museum",
        "Louvre Museum",
        "visual",
    ),
    (
        "baroque stone bridge lined with religious statues crossing a wide river "
        "in the heart of an old European city",
        "Charles Bridge Prague",
        "visual",
    ),
    (
        "massive sandstone temples with enormous pharaoh statues carved directly "
        "into a cliff face beside a Nile reservoir",
        "Abu Simbel",
        "visual",
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalise(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()


def _is_match(expected: str, got: str) -> bool:
    a, b = _normalise(expected), _normalise(got)
    return a in b or b in a


def _rank_of_ground_truth(results, expected: str) -> int | None:
    """Return 1-based rank of the correct answer, or None if not in top-k."""
    for rank, r in enumerate(results, start=1):
        if _is_match(expected, r.name):
            return rank
    return None


def bootstrap_ci(hits: list[int], n_boot: int = 10_000) -> tuple[float, float]:
    import numpy as np
    arr = np.array(hits, dtype=float)
    boots = np.random.choice(arr, size=(n_boot, len(arr)), replace=True).mean(axis=1)
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(device: str, top_k: int) -> None:
    from src.retrieval.text_search import TextSearcher
    from src.retrieval.hybrid_search import HybridSearcher

    print(f"\nLoading TextSearcher (CLIP-only)…")
    ts = TextSearcher(device=device)
    print(f"Loading HybridSearcher (CLIP + BM25)…")
    hs = HybridSearcher(device=device)

    rows = []
    for query, gt, qtype in QUERIES:
        clip_results = ts.search(query, top_k=top_k)
        hybrid_results = hs.search(query, top_k=top_k)

        clip_rank = _rank_of_ground_truth(clip_results, gt)
        hybrid_rank = _rank_of_ground_truth(hybrid_results, gt)

        clip_top = clip_results[0].name if clip_results else "-"
        hybrid_top = hybrid_results[0].name if hybrid_results else "-"

        rows.append(
            {
                "query": query,
                "ground_truth": gt,
                "query_type": qtype,
                "clip_rank": clip_rank,
                "hybrid_rank": hybrid_rank,
                "clip_top1": clip_top,
                "hybrid_top1": hybrid_top,
                "clip_hit1": clip_rank == 1,
                "hybrid_hit1": hybrid_rank == 1,
                "clip_hit3": clip_rank is not None and clip_rank <= 3,
                "hybrid_hit3": hybrid_rank is not None and hybrid_rank <= 3,
                "clip_rr": 1.0 / clip_rank if clip_rank else 0.0,
                "hybrid_rr": 1.0 / hybrid_rank if hybrid_rank else 0.0,
                "winner": (
                    "hybrid"
                    if (hybrid_rank or 999) < (clip_rank or 999)
                    else ("clip" if (clip_rank or 999) < (hybrid_rank or 999) else "tie")
                ),
            }
        )

    # ── Summary stats ────────────────────────────────────────────────────────
    def stats(subset: list[dict], key_prefix: str) -> dict:
        n = len(subset)
        h1 = sum(r[f"{key_prefix}_hit1"] for r in subset)
        h3 = sum(r[f"{key_prefix}_hit3"] for r in subset)
        mrr = sum(r[f"{key_prefix}_rr"] for r in subset) / n if n else 0.0
        lo, hi = bootstrap_ci([r[f"{key_prefix}_hit1"] for r in subset])
        return {"n": n, "hits1": h1, "hits3": h3, "mrr": mrr, "ci_lo": lo, "ci_hi": hi}

    all_clip = stats(rows, "clip")
    all_hybrid = stats(rows, "hybrid")
    kw_rows = [r for r in rows if r["query_type"] == "keyword"]
    vis_rows = [r for r in rows if r["query_type"] == "visual"]
    kw_clip = stats(kw_rows, "clip")
    kw_hybrid = stats(kw_rows, "hybrid")
    vis_clip = stats(vis_rows, "clip")
    vis_hybrid = stats(vis_rows, "hybrid")

    # ── Print per-query table ────────────────────────────────────────────────
    W = 90
    print("\n" + "═" * W)
    print("  DESCRIPTIVE QUERY EVALUATION - CLIP-only vs Hybrid (CLIP + BM25)")
    print("═" * W)

    for qtype_label, subset in [("KEYWORD-RICH queries", kw_rows), ("VISUAL queries", vis_rows)]:
        print(f"\n  ── {qtype_label} ──")
        print(f"  {'Ground truth':<40} {'CLIP':^15} {'Hybrid':^15} {'Winner':<8}")
        print("  " + "─" * (W - 2))
        for r in subset:
            gt_short = r["ground_truth"][:38]
            clip_col = f"#{r['clip_rank']} {r['clip_top1'][:10]}" if r["clip_rank"] else "miss"
            hyb_col = f"#{r['hybrid_rank']} {r['hybrid_top1'][:10]}" if r["hybrid_rank"] else "miss"
            w = r["winner"].upper() if r["winner"] != "tie" else "-"
            print(f"  {gt_short:<40} {clip_col:^15} {hyb_col:^15} {w:<8}")

    # ── Summary table ────────────────────────────────────────────────────────
    print("\n" + "─" * W)
    print(f"\n  {'Subset':<25} {'Model':<10} {'Hits@1':>8} {'Hits@3':>8} {'MRR':>8}  {'95% CI':>15}")
    print("  " + "─" * (W - 2))

    def _row(label, model, s):
        h1_pct = f"{s['hits1']}/{s['n']} ({100*s['hits1']/s['n']:.0f}%)"
        h3_pct = f"{s['hits3']}/{s['n']} ({100*s['hits3']/s['n']:.0f}%)"
        ci = f"[{s['ci_lo']:.2f}, {s['ci_hi']:.2f}]"
        print(f"  {label:<25} {model:<10} {h1_pct:>8} {h3_pct:>8} {s['mrr']:>8.3f}  {ci:>15}")

    _row("All queries", "CLIP", all_clip)
    _row("All queries", "Hybrid", all_hybrid)
    print()
    _row("Keyword-rich (n=10)", "CLIP", kw_clip)
    _row("Keyword-rich (n=10)", "Hybrid", kw_hybrid)
    print()
    _row("Visual (n=10)", "CLIP", vis_clip)
    _row("Visual (n=10)", "Hybrid", vis_hybrid)

    delta_kw = kw_hybrid["hits1"] - kw_clip["hits1"]
    delta_vis = vis_hybrid["hits1"] - vis_clip["hits1"]
    print(f"\n  BM25 Δ Hits@1 - keyword queries: {delta_kw:+d}   visual queries: {delta_vis:+d}")
    print("-" * W)

    # ── Save ────────────────────────────────────────────────────────────────
    output = {
        "queries": rows,
        "summary": {
            "all": {"clip": all_clip, "hybrid": all_hybrid},
            "keyword": {"clip": kw_clip, "hybrid": kw_hybrid},
            "visual": {"clip": vis_clip, "hybrid": vis_hybrid},
        },
    }
    out_path = Path(__file__).parents[1] / "evaluation" / "descriptive_query_results.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\n  Saved → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate descriptive queries: CLIP vs Hybrid")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    run(args.device, args.top_k)
