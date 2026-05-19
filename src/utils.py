"""
Shared utilities used across the Where Is This? pipeline.

extract_gps         - read EXIF GPS tags from an image file
sanitize_folder_name - convert a landmark name to a safe filesystem folder name
"""

from __future__ import annotations


def extract_gps(image_path: str) -> tuple[float, float] | None:
    """
    Return (lat, lon) in decimal degrees from a JPEG/PNG's EXIF GPS tags,
    or None if the image contains no location metadata.

    Uses only Pillow — no extra dependencies.  Degrees/minutes/seconds
    rationals (IFDRational objects) are converted to a single float.
    This is relevant for the responsible-AI discussion: most smartphone
    photos embed precise GPS coordinates that users may not realise they
    are sharing when they upload an image.
    """
    from PIL import Image
    from PIL.ExifTags import GPSTAGS, TAGS

    try:
        with Image.open(image_path) as img:
            raw = img._getexif()
    except Exception:
        return None

    if not raw:
        return None

    # Map numeric EXIF tag IDs to human-readable names
    exif = {TAGS.get(k, k): v for k, v in raw.items()}
    gps_info_raw = exif.get("GPSInfo")
    if not gps_info_raw:
        return None

    # Map numeric GPS sub-tag IDs to names (GPSLatitude, GPSLongitudeRef, etc.)
    gps = {GPSTAGS.get(k, k): v for k, v in gps_info_raw.items()}

    def _to_decimal(dms, ref: str) -> float | None:
        """Convert a (degrees, minutes, seconds) tuple + compass ref to decimal."""
        try:
            d, m, s = dms
            val = float(d) + float(m) / 60 + float(s) / 3600
            if ref in ("S", "W"):
                val = -val
            return val
        except Exception:
            return None

    lat = _to_decimal(gps.get("GPSLatitude", ()), gps.get("GPSLatitudeRef", "N"))
    lon = _to_decimal(gps.get("GPSLongitude", ()), gps.get("GPSLongitudeRef", "E"))

    if lat is None or lon is None:
        return None
    return lat, lon


def sanitize_folder_name(name: str) -> str:
    """
    Convert a landmark display name to the folder name used under data/images/.

    Rules: NFKD-normalise unicode (Schönbrunn → Schonbrunn), lowercase,
    spaces/hyphens/slashes → underscore, drop apostrophes and dots.
    Must be applied consistently everywhere a folder path is derived from
    a landmark name - image encoder, thumbnail lookup, and evaluation all
    use this function.
    """
    import unicodedata

    normalized = unicodedata.normalize("NFKD", name)
    ascii_name = normalized.encode("ascii", "ignore").decode("ascii")
    return (
        ascii_name.lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("'", "")
        .replace("-", "_")
        .replace(".", "")
    )
