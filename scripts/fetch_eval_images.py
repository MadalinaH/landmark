"""
Download one held-out test image per golden-set landmark from Wikipedia.

Uses the article thumbnail from the Wikipedia REST API - a different photo
from whatever is in data/images/, giving a genuine held-out evaluation set.

Run: uv run python scripts/fetch_eval_images.py
"""

import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parents[1]))
from config import DATA_DIR

EVAL_DIR = DATA_DIR / "eval"
HEADERS = {"User-Agent": "WhereIsThis/1.0 (academic project; hariimadalina@gmail.com)"}
WIKI_SUMMARY = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"

GOLDEN_SET = [
    ("Schönbrunn_Palace", "schoenbrunn_test.jpg"),
    ("Eiffel_Tower", "eiffel_test.jpg"),
    ("Colosseum", "colosseum_test.jpg"),
    ("Sagrada_Família", "sagrada_test.jpg"),
    ("St._Stephen's_Cathedral,_Vienna", "stephansdom_test.jpg"),
]


def fetch_thumbnail(title: str) -> str | None:
    r = requests.get(WIKI_SUMMARY.format(title=title), headers=HEADERS, timeout=10)
    if r.status_code != 200:
        return None
    return r.json().get("thumbnail", {}).get("source")


def main() -> None:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading golden-set images to {EVAL_DIR}\n")

    for title, filename in GOLDEN_SET:
        out = EVAL_DIR / filename
        if out.exists():
            print(f"  skip  {filename} (already exists)")
            continue

        url = fetch_thumbnail(title)
        if not url:
            print(f"  FAIL  {filename} — no thumbnail found for '{title}'")
            continue

        img_data = requests.get(url, headers=HEADERS, timeout=15).content
        out.write_bytes(img_data)
        print(f"  OK    {filename}  ({len(img_data)//1024} KB)  ← {url}")

    print("\nDone. Run: uv run python scripts/evaluate.py")


if __name__ == "__main__":
    main()
