"""Deterministic hotel query rewrites for local retrieval."""

from __future__ import annotations


_SYNONYMS = {
    "luxury": ["premium", "5 star", "upscale"],
    "beach": ["beachfront", "sea view", "Red Sea", "seaside"],
    "family": ["family friendly", "kids", "aqua park"],
    "budget": ["affordable", "value", "cheap"],
    "romantic": ["couples", "honeymoon", "quiet"],
    "business": ["business hotel", "central", "work trip"],
    "nature": ["eco lodge", "desert", "mountain", "lake", "oasis"],
    "spa": ["wellness", "spa", "relaxation"],
}


def generate_query_rewrites(city: str, hotel_type: str) -> list[str]:
    """Return exactly three deterministic retrieval rewrites."""
    clean_city = " ".join(str(city or "").split())
    clean_type = " ".join(str(hotel_type or "").split())
    lower_type = clean_type.lower()

    matched: list[str] = []
    for key, synonyms in _SYNONYMS.items():
        if key in lower_type:
            matched.extend(synonyms)

    if not matched:
        matched = [clean_type, "best rated", "recommended"]

    while len(matched) < 3:
        matched.append(clean_type or "hotel")

    first_suffix = "" if "hotel" in matched[0].lower() or "resort" in matched[0].lower() else " hotel"
    return [
        f"{matched[0]}{first_suffix} in {clean_city}",
        f"{matched[1]} stay in {clean_city}",
        f"{matched[2]} {clean_type} in {clean_city}",
    ]
