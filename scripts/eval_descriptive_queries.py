"""
Evaluate CLIP-only vs Hybrid (CLIP + BM25) text retrieval on descriptive queries.

Unlike the standard text ablation that uses exact landmark names as queries, this
script uses 40 free-form descriptive queries - the kind a user would type when they
recognise a place visually but don't know its name.  The queries are split evenly
(20/20) so that half contain rare keywords that appear in the Wikipedia description
text (giving BM25 a lexical advantage) and half are purely visual/semantic
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
    (
        "octagonal Islamic shrine inside the Al-Aqsa mosque compound on the Temple "
        "Mount in Jerusalem's Old City",
        "Dome of the Rock Jerusalem",
        "keyword",
    ),
    (
        "rock formation off the coast of Bali that is home to an ancient Hindu "
        "pilgrimage temple",
        "Tanah Lot Temple Bali",
        "keyword",
    ),
    (
        "former winter palace of the Dalai Lama overlooking the city of Lhasa in "
        "the Tibet Autonomous Region",
        "Potala Palace Tibet",
        "keyword",
    ),
    (
        "rock-hewn monolithic church dedicated to Saint George in the Amhara region "
        "of Ethiopia, one of eleven such churches",
        "Bete Giyorgis Church Ethiopia",
        "keyword",
    ),
    (
        "second largest mosque in Africa located in a Moroccan coastal city, built "
        "partly over the Atlantic Ocean",
        "Hassan II Mosque Casablanca",
        "keyword",
    ),
    (
        "former prison island in Table Bay off Cape Town that once held Nelson "
        "Mandela for political imprisonment",
        "Robben Island South Africa",
        "keyword",
    ),
    (
        "Vaishnava Hindu and Theravada Buddhist temple complex in Siem Reap, "
        "considered the largest religious monument in the world",
        "Angkor Wat",
        "keyword",
    ),
    (
        "skeletal dome left standing after the 1945 atomic bombing, preserved as a "
        "peace memorial in a Japanese city",
        "Hiroshima Peace Memorial",
        "keyword",
    ),
    (
        "Mughal-era fort serving as the main residence of emperors in Old Delhi, "
        "built from red sandstone",
        "Red Fort Delhi",
        "keyword",
    ),
    (
        "minaret and victory tower built during the Delhi Sultanate, the tallest "
        "brick minaret in the world",
        "Qutb Minar",
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
    (
        "giant red sandstone monolith rising from a flat desert plain in the "
        "Australian outback",
        "Uluru",
        "visual",
    ),
    (
        "snow-capped solitary volcanic peak with a perfectly symmetric cone "
        "silhouette near a lake",
        "Mount Fuji",
        "visual",
    ),
    (
        "golden pavilion temple reflected in a still pond surrounded by manicured "
        "gardens",
        "Kinkaku-ji Temple",
        "visual",
    ),
    (
        "thousands of vermilion-orange gates forming a tunnel-like path up a "
        "forested hillside shrine",
        "Fushimi Inari Shrine",
        "visual",
    ),
    (
        "futuristic resort with three connected towers topped by a boat-shaped "
        "sky deck overlooking a bay",
        "Marina Bay Sands Singapore",
        "visual",
    ),
    (
        "very tall glass and steel skyscraper piercing a desert city skyline at "
        "sunset",
        "Burj Khalifa",
        "visual",
    ),
    (
        "twin steel-and-glass skyscrapers connected by a sky bridge in a Southeast "
        "Asian capital",
        "Petronas Towers",
        "visual",
    ),
    (
        "white sail-shaped shell roof segments forming a performing arts venue on "
        "a harbour foreshore",
        "Sydney Opera House",
        "visual",
    ),
    (
        "wide curtain of waterfalls spanning a river gorge on the border between "
        "two southern African countries",
        "Victoria Falls",
        "visual",
    ),
    (
        "vast open savanna grassland in East Africa where huge herds migrate "
        "across the plains each year",
        "Serengeti National Park",
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
        hybrid_results = hs.search(query, top_k=top_k, fusion="weighted")
        rrf_results = hs.search(query, top_k=top_k, fusion="rrf")

        clip_rank = _rank_of_ground_truth(clip_results, gt)
        hybrid_rank = _rank_of_ground_truth(hybrid_results, gt)
        rrf_rank = _rank_of_ground_truth(rrf_results, gt)

        clip_top = clip_results[0].name if clip_results else "-"
        hybrid_top = hybrid_results[0].name if hybrid_results else "-"
        rrf_top = rrf_results[0].name if rrf_results else "-"

        rows.append(
            {
                "query": query,
                "ground_truth": gt,
                "query_type": qtype,
                "clip_rank": clip_rank,
                "hybrid_rank": hybrid_rank,
                "rrf_rank": rrf_rank,
                "clip_top1": clip_top,
                "hybrid_top1": hybrid_top,
                "rrf_top1": rrf_top,
                "clip_hit1": clip_rank == 1,
                "hybrid_hit1": hybrid_rank == 1,
                "rrf_hit1": rrf_rank == 1,
                "clip_hit3": clip_rank is not None and clip_rank <= 3,
                "hybrid_hit3": hybrid_rank is not None and hybrid_rank <= 3,
                "rrf_hit3": rrf_rank is not None and rrf_rank <= 3,
                "clip_rr": 1.0 / clip_rank if clip_rank else 0.0,
                "hybrid_rr": 1.0 / hybrid_rank if hybrid_rank else 0.0,
                "rrf_rr": 1.0 / rrf_rank if rrf_rank else 0.0,
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
    all_rrf = stats(rows, "rrf")
    kw_rows = [r for r in rows if r["query_type"] == "keyword"]
    vis_rows = [r for r in rows if r["query_type"] == "visual"]
    kw_clip = stats(kw_rows, "clip")
    kw_hybrid = stats(kw_rows, "hybrid")
    kw_rrf = stats(kw_rows, "rrf")
    vis_clip = stats(vis_rows, "clip")
    vis_hybrid = stats(vis_rows, "hybrid")
    vis_rrf = stats(vis_rows, "rrf")

    # ── Print per-query table ────────────────────────────────────────────────
    W = 100
    print("\n" + "═" * W)
    print("  DESCRIPTIVE QUERY EVALUATION - CLIP-only vs Weighted-Hybrid vs RRF-Hybrid")
    print("═" * W)

    for qtype_label, subset in [("KEYWORD-RICH queries", kw_rows), ("VISUAL queries", vis_rows)]:
        print(f"\n  ── {qtype_label} ──")
        print(f"  {'Ground truth':<35} {'CLIP':^14} {'Weighted':^14} {'RRF':^14}")
        print("  " + "─" * (W - 2))
        for r in subset:
            gt_short = r["ground_truth"][:33]
            clip_col = f"#{r['clip_rank']} {r['clip_top1'][:9]}" if r["clip_rank"] else "miss"
            hyb_col = f"#{r['hybrid_rank']} {r['hybrid_top1'][:9]}" if r["hybrid_rank"] else "miss"
            rrf_col = f"#{r['rrf_rank']} {r['rrf_top1'][:9]}" if r["rrf_rank"] else "miss"
            print(f"  {gt_short:<35} {clip_col:^14} {hyb_col:^14} {rrf_col:^14}")

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
    _row("All queries", "Weighted", all_hybrid)
    _row("All queries", "RRF", all_rrf)
    print()
    _row("Keyword-rich (n=20)", "CLIP", kw_clip)
    _row("Keyword-rich (n=20)", "Weighted", kw_hybrid)
    _row("Keyword-rich (n=20)", "RRF", kw_rrf)
    print()
    _row("Visual (n=20)", "CLIP", vis_clip)
    _row("Visual (n=20)", "Weighted", vis_hybrid)
    _row("Visual (n=20)", "RRF", vis_rrf)

    delta_kw_w = kw_hybrid["hits1"] - kw_clip["hits1"]
    delta_vis_w = vis_hybrid["hits1"] - vis_clip["hits1"]
    delta_kw_r = kw_rrf["hits1"] - kw_clip["hits1"]
    delta_vis_r = vis_rrf["hits1"] - vis_clip["hits1"]
    print(f"\n  Weighted Δ Hits@1 - keyword queries: {delta_kw_w:+d}   visual queries: {delta_vis_w:+d}")
    print(f"  RRF      Δ Hits@1 - keyword queries: {delta_kw_r:+d}   visual queries: {delta_vis_r:+d}")
    print("-" * W)

    # ── Save ────────────────────────────────────────────────────────────────
    output = {
        "queries": rows,
        "summary": {
            "all": {"clip": all_clip, "hybrid": all_hybrid, "rrf": all_rrf},
            "keyword": {"clip": kw_clip, "hybrid": kw_hybrid, "rrf": kw_rrf},
            "visual": {"clip": vis_clip, "hybrid": vis_hybrid, "rrf": vis_rrf},
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
