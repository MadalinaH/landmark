"""
Enrich landmarks.json with Wikipedia summaries and coordinates.

For each landmark:
  - Fetches /page/summary/{title} from the Wikipedia REST API
  - Updates description with Wikipedia extract (kept if longer than current)
  - Fills in missing lat/lon from Wikipedia coordinates
  - Writes results back to data/landmarks.json (in-place)
"""

import json
import sys
import time
import urllib.parse
from pathlib import Path

import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parents[1]))
from config import DATA_DIR

WIKI_SUMMARY_API = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
WIKI_SEARCH_API = "https://en.wikipedia.org/w/api.php"
WIKIDATA_API = "https://www.wikidata.org/w/api.php"
HEADERS = {"User-Agent": "WhereIsThis/1.0 (academic project; hariimadalina@gmail.com)"}
DELAY = 0.5  # seconds between requests — Wikipedia rate limit courtesy


def _title_from_url(url: str) -> str | None:
    """Extract article title from a full Wikipedia URL."""
    if not url:
        return None
    parsed = urllib.parse.urlparse(url)
    path = parsed.path  # e.g. /wiki/Sch%C3%B6nbrunn_Palace
    if "/wiki/" not in path:
        return None
    return urllib.parse.unquote(path.split("/wiki/", 1)[1])


def _fetch_summary(title: str) -> dict | None:
    url = WIKI_SUMMARY_API.format(title=urllib.parse.quote(title, safe="(),'"))
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code == 200:
            return r.json()
        if r.status_code == 404:
            return None
        r.raise_for_status()
    except requests.RequestException as e:
        print(f"  Request error for '{title}': {e}", file=sys.stderr)
    return None


def _search_title(query: str) -> str | None:
    """Use the Wikipedia search API to find the best-matching article title."""
    try:
        r = requests.get(
            WIKI_SEARCH_API,
            params={
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srlimit": 1,
                "format": "json",
            },
            headers=HEADERS,
            timeout=10,
        )
        r.raise_for_status()
        results = r.json().get("query", {}).get("search", [])
        if results:
            return results[0]["title"]
    except requests.RequestException as e:
        print(f"  Search error for '{query}': {e}", file=sys.stderr)
    return None


def _fetch_wikidata_coords(wikipedia_title: str) -> tuple[float, float] | None:
    """Look up P625 (coordinate location) from Wikidata via the Wikipedia title."""
    try:
        # Resolve article title → Wikidata entity ID
        r = requests.get(
            WIKIDATA_API,
            params={
                "action": "wbgetentities",
                "sites": "enwiki",
                "titles": wikipedia_title,
                "props": "claims",
                "format": "json",
            },
            headers=HEADERS,
            timeout=10,
        )
        r.raise_for_status()
        entities = r.json().get("entities", {})
        for entity in entities.values():
            claims = entity.get("claims", {})
            p625 = claims.get("P625", [])
            if p625:
                val = p625[0]["mainsnak"]["datavalue"]["value"]
                return val["latitude"], val["longitude"]
    except (requests.RequestException, KeyError, IndexError):
        pass
    return None


def enrich_landmark(lm: dict) -> tuple[dict, str]:
    """Return (updated landmark dict, status string)."""
    # 1. Try URL-derived title first
    title = _title_from_url(lm.get("wikipedia_url", ""))
    data = _fetch_summary(title) if title else None

    # 2. Try raw landmark name
    if data is None:
        data = _fetch_summary(lm["name"])

    # 3. Fall back to Wikipedia search API (handles location-suffixed names)
    if data is None:
        found_title = _search_title(lm["name"])
        if found_title:
            data = _fetch_summary(found_title)

    if data is None:
        return lm, "not_found"

    changed = []

    # Update description if Wikipedia extract is longer
    wiki_extract = data.get("extract", "").strip()
    current_desc = lm.get("description", "").strip()
    if wiki_extract and len(wiki_extract) > len(current_desc):
        lm["description"] = wiki_extract
        changed.append("description")

    # Fill missing coordinates — Wikipedia summary first, Wikidata fallback
    coords = data.get("coordinates")
    if coords and lm.get("lat") is None:
        lm["lat"] = coords.get("lat")
        lm["lon"] = coords.get("lon")
        changed.append("coords")

    if lm.get("lat") is None:
        # Resolve the Wikipedia title used for this lookup
        wiki_title = (
            _title_from_url(lm.get("wikipedia_url", ""))
            or data.get("title")
            or lm["name"]
        )
        wikidata_coords = _fetch_wikidata_coords(wiki_title)
        if wikidata_coords:
            lm["lat"], lm["lon"] = wikidata_coords
            changed.append("coords_wikidata")

    # Store canonical Wikipedia URL if missing
    if not lm.get("wikipedia_url") and data.get("content_urls"):
        lm["wikipedia_url"] = data["content_urls"].get("desktop", {}).get("page", "")

    status = "updated:" + "+".join(changed) if changed else "ok"
    return lm, status


def main() -> None:
    path = DATA_DIR / "landmarks.json"
    landmarks = json.loads(path.read_text(encoding="utf-8"))

    print(f"Fetching Wikipedia summaries for {len(landmarks)} landmarks…")
    stats = {"desc": 0, "coords_wiki": 0, "coords_wikidata": 0, "not_found": 0, "ok": 0}

    enriched = []
    for lm in tqdm(landmarks):
        lm, status = enrich_landmark(lm)
        enriched.append(lm)

        if "description" in status:
            stats["desc"] += 1
        if "coords_wikidata" in status:
            stats["coords_wikidata"] += 1
        elif "coords" in status:
            stats["coords_wiki"] += 1
        if status == "not_found":
            stats["not_found"] += 1
        if status == "ok":
            stats["ok"] += 1

        time.sleep(DELAY)

    path.write_text(
        json.dumps(enriched, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    missing_coords = sum(1 for lm in enriched if lm.get("lat") is None)
    print(f"\nDone. Saved to {path}")
    print(f"  Descriptions updated     : {stats['desc']}")
    print(f"  Coordinates (Wikipedia)  : {stats['coords_wiki']}")
    print(f"  Coordinates (Wikidata)   : {stats['coords_wikidata']}")
    print(f"  Not found on Wikipedia   : {stats['not_found']}")
    print(f"  Already up to date       : {stats['ok']}")
    print(f"  Still missing coordinates: {missing_coords}")


if __name__ == "__main__":
    main()
