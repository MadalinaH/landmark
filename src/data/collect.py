import json
import time
import urllib.parse

import requests

from config import RAW_DIR, USER_AGENT

HEADERS = {
    "User-Agent": USER_AGENT,
    "Referer": "https://commons.wikimedia.org/",
}


def get_wikipedia_summary(landmark_name: str) -> dict:
    """Fetch summary from Wikipedia REST API."""
    encoded = urllib.parse.quote(landmark_name.replace(" ", "_"))
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code == 200:
            data = r.json()
            return {
                "description": data.get("extract", ""),
                "wikipedia_url": data.get("content_urls", {})
                .get("desktop", {})
                .get("page", ""),
            }
    except Exception as e:
        print(f"  Wikipedia error for {landmark_name}: {e}")
    return {"description": "", "wikipedia_url": ""}


def get_wikidata_coords(landmark_name: str) -> dict:
    """Fetch coordinates from Wikidata."""
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "titles": landmark_name,
        "prop": "coordinates",
        "format": "json",
    }
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=10)
        pages = r.json().get("query", {}).get("pages", {})
        for page in pages.values():
            coords = page.get("coordinates", [])
            if coords:
                return {"lat": coords[0]["lat"], "lon": coords[0]["lon"]}
    except Exception as e:
        print(f"  Wikidata error for {landmark_name}: {e}")
    return {"lat": None, "lon": None}


def build_landmarks_json(landmark_list: list[dict]) -> None:
    """
    Given a list of {name, region} dicts, fetch all metadata
    and save to data/landmarks.json.
    """
    output_path = RAW_DIR / "landmarks_raw.json"
    landmarks = []

    for i, item in enumerate(landmark_list):
        name = item["name"]
        region = item["region"]
        print(f"[{i+1}/{len(landmark_list)}] {name}")

        wiki = get_wikipedia_summary(name)
        coords = get_wikidata_coords(name)

        landmark = {
            "id": i,
            "name": name,
            "region": region,
            "description": wiki["description"],
            "wikipedia_url": wiki["wikipedia_url"],
            "lat": coords["lat"],
            "lon": coords["lon"],
        }

        landmarks.append(landmark)
        print(
            f"  desc: {len(wiki['description'])} chars | "
            f"coords: {coords['lat']},{coords['lon']}"
        )

        time.sleep(0.5)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(landmarks, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(landmarks)} landmarks to {output_path}")
