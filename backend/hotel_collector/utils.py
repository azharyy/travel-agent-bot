"""Utility functions for normalizing RollingGo hotel catalog data."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Iterable, Mapping
from typing import Any

from pydantic import BaseModel

from backend.hotel_collector.cities import CITIES


_CITY_OVERRIDES = {
    "sahel": "North Coast",
    "sahel hotels": "North Coast",
    "north coast": "North Coast",
    "north coast hotels": "North Coast",
    "north coast egypt": "North Coast",
    "north coast egypt hotels": "North Coast",
    "sharm el sheikh": "Sharm El-Sheikh",
    "sharm el sheikh hotels": "Sharm El-Sheikh",
    "sharm el-sheikh": "Sharm El-Sheikh",
    "sharm el-sheikh hotels": "Sharm El-Sheikh",
}


def slugify(value: Any) -> str:
    """Convert a value into a stable lowercase slug."""
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower()).strip("-")
    return text or "unknown"


def _collapse_spaces(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _strip_hotels_suffix(value: str) -> str:
    return re.sub(r"\s+hotels\s*$", "", _collapse_spaces(value), flags=re.IGNORECASE)


def _title_city(value: str) -> str:
    known_words = {
        "ain": "Ain",
        "alexandria": "Alexandria",
        "aswan": "Aswan",
        "bay": "Bay",
        "cairo": "Cairo",
        "catherine": "Catherine",
        "dahab": "Dahab",
        "el": "El",
        "fayoum": "Fayoum",
        "gouna": "Gouna",
        "hurghada": "Hurghada",
        "ismailia": "Ismailia",
        "luxor": "Luxor",
        "marsa": "Marsa",
        "matruh": "Matruh",
        "new": "New",
        "nuweiba": "Nuweiba",
        "oasis": "Oasis",
        "port": "Port",
        "said": "Said",
        "saint": "Saint",
        "siwa": "Siwa",
        "sokhna": "Sokhna",
        "soma": "Soma",
        "taba": "Taba",
    }
    return " ".join(known_words.get(part.lower(), part.capitalize()) for part in value.split())


def city_query_to_display_name(query: str) -> str:
    """Convert a supported city search query into a display city name."""
    cleaned = _collapse_spaces(query or "")
    key = cleaned.lower()
    if key in _CITY_OVERRIDES:
        return _CITY_OVERRIDES[key]

    without_suffix = _strip_hotels_suffix(cleaned)
    suffix_key = without_suffix.lower()
    if suffix_key in _CITY_OVERRIDES:
        return _CITY_OVERRIDES[suffix_key]

    return _title_city(without_suffix)


_SUPPORTED_CITY_NAMES = {
    city_query_to_display_name(city).lower(): city_query_to_display_name(city) for city in CITIES
}


def canonicalize_city(city: str) -> str:
    """Normalize user/provider city values to the GuideMe display city set."""
    display_name = city_query_to_display_name(city)
    key = display_name.lower()
    return _SUPPORTED_CITY_NAMES.get(key, display_name)


def _coerce_to_items(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        parts = re.split(r"[,;|]+", value)
        return [part.strip() for part in parts if part.strip()]
    if isinstance(value, Mapping):
        return [str(key).strip() for key, enabled in value.items() if enabled and str(key).strip()]
    if isinstance(value, Iterable):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _normalize_items(value: Any) -> list[str]:
    seen = set()
    normalized = []
    for item in _coerce_to_items(value):
        cleaned = _collapse_spaces(item).lower()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            normalized.append(cleaned)
    return normalized


def normalize_tags(value: Any) -> list[str]:
    """Normalize provider tags to lowercase, de-duplicated strings."""
    return _normalize_items(value)


def normalize_amenities(value: Any) -> list[str]:
    """Normalize provider amenities to lowercase, de-duplicated strings."""
    return _normalize_items(value)


def safe_float(value: Any, default: float | None = None) -> float | None:
    """Parse a float from loose provider values."""
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return default
    if isinstance(value, int | float):
        return float(value)
    match = re.search(r"-?\d+(?:[.,]\d+)?", str(value).replace(",", ""))
    return float(match.group(0)) if match else default


def safe_int(value: Any, default: int | None = None) -> int | None:
    """Parse an integer from loose provider values."""
    parsed = safe_float(value)
    return int(parsed) if parsed is not None else default


def make_hotel_id(
    name: str,
    city: str,
    provider_hotel_id: str | int | None = None,
    provider: str = "rollinggo",
) -> str:
    """Build a stable local hotel identifier."""
    provider_slug = slugify(provider)
    if provider_hotel_id:
        return f"{provider_slug}-{slugify(provider_hotel_id)}"

    canonical_city = canonicalize_city(city)
    seed = f"{provider_slug}|{canonical_city.lower()}|{str(name).lower()}"
    digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:10]
    return f"{provider_slug}-{slugify(canonical_city)}-{slugify(name)[:48]}-{digest}"


def flatten_chroma_metadata(metadata: Mapping[str, Any]) -> dict[str, str | int | float | bool]:
    """Convert nested metadata into Chroma-compatible scalar values."""
    flattened: dict[str, str | int | float | bool] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, str | int | float | bool):
            flattened[str(key)] = value
            continue
        flattened[str(key)] = json.dumps(value, ensure_ascii=True, sort_keys=True)
    return flattened


def _as_dict(hotel: BaseModel | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(hotel, BaseModel):
        return hotel.model_dump()
    return dict(hotel)


def summarize_hotel_for_rag(hotel: BaseModel | Mapping[str, Any]) -> str:
    """Create a static hotel summary suitable for text embeddings."""
    data = _as_dict(hotel)
    parts = [
        str(data.get("name") or "Unknown hotel"),
        f"in {canonicalize_city(str(data.get('city') or ''))}",
    ]

    if data.get("country_code"):
        parts.append(f"country {data['country_code']}")
    if data.get("source"):
        parts.append(f"source {data['source']}")
    if data.get("hotel_type"):
        parts.append(str(data["hotel_type"]))
    if data.get("star_rating") is not None:
        parts.append(f"{data['star_rating']} star")
    if data.get("review_score") is not None:
        parts.append(f"guest score {data['review_score']}")
    if data.get("address"):
        parts.append(f"address: {data['address']}")
    if data.get("amenities"):
        parts.append("amenities: " + ", ".join(normalize_amenities(data["amenities"])))
    if data.get("tags"):
        parts.append("tags: " + ", ".join(normalize_tags(data["tags"])))
    if data.get("summary"):
        parts.append(str(data["summary"]))
    if data.get("description"):
        parts.append(str(data["description"]))

    return ". ".join(part for part in parts if part).strip()
