"""Manual builder for GuideMe's static RollingGo hotel card database."""

from __future__ import annotations

import argparse
import asyncio
import html
import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup
from dotenv import load_dotenv

from backend.hotel_collector.cities import CITIES
from backend.hotel_collector.models import HotelCard
from backend.hotel_collector.rollinggo_mcp_client import (
    SEARCH_HOTELS_TOOL,
    RollingGoMCPClient,
    RollingGoMCPError,
)
from backend.hotel_collector.utils import (
    canonicalize_city,
    city_query_to_display_name,
    make_hotel_id,
    normalize_amenities,
    normalize_tags,
    safe_float,
)


DATA_DIR = Path("backend/data")
HOTELS_PATH = DATA_DIR / "hotels.json"
TAGS_PATH = DATA_DIR / "rollinggo_tags.json"

COASTAL_CITIES = {
    "Alexandria",
    "Hurghada",
    "Sharm El-Sheikh",
    "Marsa Matruh",
    "Port Said",
    "North Coast",
    "Dahab",
    "Taba",
    "Soma Bay",
    "El Gouna",
    "Nuweiba",
    "Ain Sokhna",
}

COASTAL_PROFILES = [
    "general best hotels",
    "luxury beach resort",
    "family resort",
    "budget value hotel",
    "romantic spa resort",
]

URBAN_CULTURAL_PROFILES = [
    "general best hotels",
    "luxury business hotel",
    "family friendly hotel",
    "budget value hotel",
    "boutique cultural stay",
]

PROFILE_TYPES = {
    "general best hotels": "hotel",
    "luxury beach resort": "beach resort",
    "family resort": "family resort",
    "budget value hotel": "budget hotel",
    "romantic spa resort": "spa resort",
    "luxury business hotel": "business hotel",
    "family friendly hotel": "family hotel",
    "boutique cultural stay": "boutique hotel",
}

PRICE_KEYS = {
    "price",
    "rate",
    "rates",
    "amount",
    "nightlyPrice",
    "nightly_price",
    "totalPrice",
    "total_price",
    "currency",
}

WHITESPACE_RE = re.compile(r"\s+")
HTML_TAG_RE = re.compile(r"<[^>]+>")

FOREIGN_LOCATION_MARKERS = {
    "corfu",
    "diwobao",
    "foxwoods",
    "haleiwa",
    "hawaii",
    "hatley",
    "ionian sea",
    "karpovoulos",
    "north stonington",
    "samos island",
    "sainte catherine de hatley",
    "urumqi",
    "wildwood",
}


@dataclass
class BuildSettings:
    rollinggo_mcp_url: str = "http://127.0.0.1:8010/mcp"
    rollinggo_api_key: str = ""
    rollinggo_accept_language: str = "en_US"
    hotel_country_code: str = "EG"
    hotel_db_build_checkin: str = "2026-08-01"
    hotel_db_build_nights: int = 2
    hotel_db_build_adults: int = 2
    hotel_db_top_n: int = 20


@dataclass
class Candidate:
    card: HotelCard
    provider_hotel_id: str
    score: float = 0.0
    star_rating: float = 0.0
    description_length: int = 0
    has_image: bool = False
    source_profiles: set[str] = field(default_factory=set)

    @property
    def rank_score(self) -> float:
        score_weight = self.score * 10
        star_weight = self.star_rating * 5
        description_weight = min(self.description_length, 800) / 100
        image_weight = 6 if self.has_image else 0
        tag_weight = len(self.card.tags) + len(self.card.amenities) * 0.5
        diversity_weight = len(self.source_profiles) * 2
        return score_weight + star_weight + description_weight + image_weight + tag_weight + diversity_weight


def _load_build_settings() -> BuildSettings:
    load_dotenv()
    try:
        from backend.core.config import get_settings

        settings = get_settings()
        return BuildSettings(
            rollinggo_mcp_url=settings.rollinggo_mcp_url,
            rollinggo_api_key=settings.rollinggo_api_key,
            rollinggo_accept_language=settings.rollinggo_accept_language,
            hotel_country_code=settings.hotel_country_code,
            hotel_db_build_checkin=settings.hotel_db_build_checkin,
            hotel_db_build_nights=settings.hotel_db_build_nights,
            hotel_db_build_adults=settings.hotel_db_build_adults,
            hotel_db_top_n=settings.hotel_db_top_n,
        )
    except Exception:
        return BuildSettings(
            rollinggo_mcp_url=os.getenv("ROLLINGGO_MCP_URL", "http://127.0.0.1:8010/mcp"),
            rollinggo_api_key=os.getenv("ROLLINGGO_API_KEY", ""),
            rollinggo_accept_language=os.getenv("ROLLINGGO_ACCEPT_LANGUAGE", "en_US"),
            hotel_country_code=os.getenv("HOTEL_COUNTRY_CODE", "EG"),
            hotel_db_build_checkin=os.getenv("HOTEL_DB_BUILD_CHECKIN", "2026-08-01"),
            hotel_db_build_nights=int(os.getenv("HOTEL_DB_BUILD_NIGHTS", "2")),
            hotel_db_build_adults=int(os.getenv("HOTEL_DB_BUILD_ADULTS", "2")),
            hotel_db_top_n=int(os.getenv("HOTEL_DB_TOP_N", "20")),
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build GuideMe's static hotel card database.")
    parser.add_argument("--city", help='Build one city, for example "Hurghada" or "North Coast".')
    parser.add_argument("--limit-cities", type=int, help="Limit how many configured cities to process.")
    parser.add_argument("--skip-tags", action="store_true", help="Skip getHotelSearchTags cache refresh.")
    parser.add_argument("--force", action="store_true", help="Do not preserve existing hotels.json entries.")
    return parser.parse_args()


def _provider_query_for_display_city(display_city: str) -> str:
    if display_city == "North Coast":
        return "North Coast Egypt hotels"
    if display_city == "Sharm El-Sheikh":
        return "Sharm El Sheikh hotels"
    return f"{display_city} hotels"


def _display_to_query_map() -> dict[str, str]:
    return {city_query_to_display_name(query): query for query in CITIES}


def _resolve_city_queries(city: str | None, limit_cities: int | None) -> list[str]:
    if city:
        display_city = canonicalize_city(city)
        query = _display_to_query_map().get(display_city) or _provider_query_for_display_city(display_city)
        return [query]

    queries = list(CITIES)
    if limit_cities is not None:
        return queries[: max(limit_cities, 0)]
    return queries


def _profiles_for_city(display_city: str) -> list[str]:
    if display_city in COASTAL_CITIES:
        return COASTAL_PROFILES
    return URBAN_CULTURAL_PROFILES


def _search_payload(
    provider_city_query: str,
    profile: str,
    settings: BuildSettings,
) -> dict[str, Any]:
    search_place = city_query_to_display_name(provider_city_query)
    return {
        "originQuery": f"{provider_city_query} {profile}",
        "place": search_place,
        "placeType": "city",
        "countryCode": settings.hotel_country_code,
        "size": settings.hotel_db_top_n,
        "checkInParam": {
            "adultCount": settings.hotel_db_build_adults,
            "checkInDate": settings.hotel_db_build_checkin,
            "stayNights": settings.hotel_db_build_nights,
        },
        "filterOptions": {
            "starRatings": [0.0, 5.0],
        },
        "hotelTags": {
            "preferredTags": [],
            "requiredTags": [],
            "excludedTags": [],
        },
    }


def _pick(data: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        value = data.get(key)
        if value not in (None, ""):
            return value
    return None


def _text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    if "<" in text and ">" in text:
        try:
            text = BeautifulSoup(text, "html.parser").get_text(" ", strip=True)
        except Exception:
            text = HTML_TAG_RE.sub(" ", text)
    text = html.unescape(text)
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text or None


def _truncate(value: Any, limit: int = 700) -> str | None:
    text = _text(value)
    if not text:
        return None
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _label_items(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        for key in ("name", "title", "label", "text", "tagName", "amenityName"):
            label = _text(value.get(key))
            if label:
                return [label]
        return [str(key) for key, enabled in value.items() if enabled and key not in PRICE_KEYS]
    if isinstance(value, list | tuple | set):
        items: list[str] = []
        for item in value:
            items.extend(_label_items(item))
        return items
    return [str(value)]


def _normalize_provider_tags(raw_tags: Any, profile: str) -> list[str]:
    tags = _label_items(raw_tags)
    tags.append(profile)
    hotel_type = PROFILE_TYPES.get(profile)
    if hotel_type:
        tags.append(hotel_type)
    return normalize_tags(tags)


def _normalize_provider_amenities(raw_amenities: Any) -> list[str]:
    return normalize_amenities(_label_items(raw_amenities))


def _image_url(hotel: Mapping[str, Any]) -> str | None:
    image_url = _text(_pick(hotel, "imageUrl", "image_url", "mainImageUrl", "mainPhotoUrl", "photoUrl"))
    if image_url:
        return image_url

    images = _pick(hotel, "images", "photos", "imageUrls")
    if isinstance(images, list) and images:
        first = images[0]
        if isinstance(first, Mapping):
            return _text(_pick(first, "url", "imageUrl", "image_url"))
        return _text(first)
    return None


def _summary(hotel: Mapping[str, Any], name: str, display_city: str) -> str:
    description = _truncate(_pick(hotel, "description", "summary", "shortDescription"))
    if description:
        return description
    return f"{name} in {display_city}."


def _looks_like_foreign_false_positive(hotel: Mapping[str, Any], name: str) -> bool:
    parts = [
        name,
        _text(_pick(hotel, "address", "destinationName", "areaName")),
        _text(_pick(hotel, "description", "summary", "shortDescription")),
    ]
    haystack = " ".join(part for part in parts if part).casefold()
    return any(marker in haystack for marker in FOREIGN_LOCATION_MARKERS)


def _provider_hotel_id(hotel: Mapping[str, Any], name: str, display_city: str) -> str:
    provider_id = _text(_pick(hotel, "hotelId", "hotel_id", "id", "providerHotelId"))
    if provider_id:
        return provider_id
    return make_hotel_id(name=name, city=display_city)


def _normalize_hotel(
    hotel: Mapping[str, Any],
    display_city: str,
    profile: str,
) -> Candidate | None:
    name = _text(_pick(hotel, "nameEn", "name", "hotelName", "title"))
    if not name:
        return None
    if _looks_like_foreign_false_positive(hotel, name):
        return None

    provider_id = _provider_hotel_id(hotel, name, display_city)
    summary = _summary(hotel, name, display_city)
    tags = _normalize_provider_tags(_pick(hotel, "tags", "hotelTags", "labels"), profile)
    amenities = _normalize_provider_amenities(
        _pick(hotel, "hotelAmenities", "amenities", "facilities")
    )
    image_url = _image_url(hotel)

    card = HotelCard(
        id=make_hotel_id(name=name, city=display_city, provider_hotel_id=provider_id),
        source="rollinggo",
        hotel_id=provider_id,
        name=name,
        city=display_city,
        hotel_type=PROFILE_TYPES.get(profile, "hotel"),
        summary=summary,
        tags=tags,
        amenities=amenities,
        image_url=image_url,
        booking_url=_text(_pick(hotel, "bookingUrl", "booking_url", "url")),
    )
    return Candidate(
        card=card,
        provider_hotel_id=provider_id,
        score=safe_float(_pick(hotel, "score", "reviewScore", "rating"), 0.0) or 0.0,
        star_rating=safe_float(_pick(hotel, "starRating", "star_rating", "stars"), 0.0) or 0.0,
        description_length=len(summary or ""),
        has_image=bool(image_url),
        source_profiles={profile},
    )


def _find_hotel_items(value: Any) -> list[Mapping[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, Mapping)]
    if not isinstance(value, Mapping):
        return []

    for key in (
        "hotelInformationList",
        "hotels",
        "items",
        "results",
        "properties",
        "data",
        "result",
        "payload",
    ):
        nested = value.get(key)
        if isinstance(nested, list):
            return [item for item in nested if isinstance(item, Mapping)]
        if isinstance(nested, Mapping):
            found = _find_hotel_items(nested)
            if found:
                return found
    return []


def _merge_unique(left: list[str], right: list[str]) -> list[str]:
    seen = set()
    output = []
    for item in left + right:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            output.append(item)
    return output


def _merge_candidate(existing: Candidate, incoming: Candidate) -> Candidate:
    existing.source_profiles.update(incoming.source_profiles)
    existing.card.tags = _merge_unique(existing.card.tags, incoming.card.tags)
    existing.card.amenities = _merge_unique(existing.card.amenities, incoming.card.amenities)

    if not existing.card.image_url and incoming.card.image_url:
        existing.card.image_url = incoming.card.image_url
        existing.has_image = True
    if not existing.card.booking_url and incoming.card.booking_url:
        existing.card.booking_url = incoming.card.booking_url
    if len(incoming.card.summary or "") > len(existing.card.summary or ""):
        existing.card.summary = incoming.card.summary
        existing.description_length = incoming.description_length
    if incoming.score > existing.score:
        existing.score = incoming.score
    if incoming.star_rating > existing.star_rating:
        existing.star_rating = incoming.star_rating
    if incoming.rank_score > existing.rank_score:
        existing.card.hotel_type = incoming.card.hotel_type
    return existing


def _dedupe_candidates(candidates: list[Candidate]) -> list[Candidate]:
    by_provider_id: dict[str, Candidate] = {}
    for candidate in candidates:
        key = candidate.provider_hotel_id
        if key in by_provider_id:
            by_provider_id[key] = _merge_candidate(by_provider_id[key], candidate)
        else:
            by_provider_id[key] = candidate
    return list(by_provider_id.values())


def _select_diverse(candidates: list[Candidate], profiles: list[str], limit: int) -> list[Candidate]:
    selected: list[Candidate] = []
    selected_ids: set[str] = set()
    sorted_candidates = sorted(candidates, key=lambda item: item.rank_score, reverse=True)

    while len(selected) < limit:
        added = False
        for profile in profiles:
            for candidate in sorted_candidates:
                if candidate.card.id in selected_ids or profile not in candidate.source_profiles:
                    continue
                selected.append(candidate)
                selected_ids.add(candidate.card.id)
                added = True
                break
            if len(selected) >= limit:
                break
        if not added:
            break

    if len(selected) < limit:
        for candidate in sorted_candidates:
            if candidate.card.id in selected_ids:
                continue
            selected.append(candidate)
            selected_ids.add(candidate.card.id)
            if len(selected) >= limit:
                break

    return selected


async def _cache_tags(client: RollingGoMCPClient) -> None:
    try:
        tags = await client.get_hotel_search_tags()
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        TAGS_PATH.write_text(
            json.dumps(tags, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Cached RollingGo search tags to {TAGS_PATH}")
    except Exception as exc:
        print(f"Could not cache RollingGo search tags: {exc}")


async def _preflight_client(client: RollingGoMCPClient) -> bool:
    try:
        tools = await client.list_tools()
    except Exception as exc:
        print("Could not connect to RollingGo MCP before starting the hotel build.")
        print(f"Configured ROLLINGGO_MCP_URL: {client.url}")
        print("Start the external RollingGo MCP server separately, or update ROLLINGGO_MCP_URL in .env.")
        print(f"Details: {exc}")
        return False

    tool_names = [str(tool.get("name") or "") for tool in tools]
    has_search = any(
        name == SEARCH_HOTELS_TOOL or name.endswith(f".{SEARCH_HOTELS_TOOL}") or name.endswith(f"_{SEARCH_HOTELS_TOOL}")
        for name in tool_names
    )
    if not has_search:
        print(f"RollingGo MCP is reachable, but '{SEARCH_HOTELS_TOOL}' was not found.")
        print("Available tools:")
        for name in tool_names:
            print(f"- {name or '<unnamed>'}")
        return False

    print(f"Connected to RollingGo MCP at {client.url}. Found {len(tool_names)} tools.")
    return True


async def _collect_city(
    client: RollingGoMCPClient,
    city_query: str,
    settings: BuildSettings,
) -> tuple[list[HotelCard], dict[str, Any]]:
    display_city = city_query_to_display_name(city_query)
    provider_city_query = _provider_query_for_display_city(display_city)
    profiles = _profiles_for_city(display_city)
    errors: list[str] = []
    raw_candidates: list[Candidate] = []
    collected_before_dedupe = 0

    for profile in profiles:
        payload = _search_payload(provider_city_query, profile, settings)
        try:
            result = await client.search_hotels(payload)
            hotels = _find_hotel_items(result)
            collected_before_dedupe += len(hotels)
            for hotel in hotels:
                candidate = _normalize_hotel(hotel, display_city, profile)
                if candidate:
                    raw_candidates.append(candidate)
        except Exception as exc:
            errors.append(f"{profile}: {exc}")

    deduped = _dedupe_candidates(raw_candidates)
    selected = _select_diverse(deduped, profiles, settings.hotel_db_top_n)
    cards = [candidate.card for candidate in selected]

    stats = {
        "city": display_city,
        "collected_before_dedupe": collected_before_dedupe,
        "after_dedupe": len(deduped),
        "final_saved_count": len(cards),
        "errors": errors,
    }
    return cards, stats


def _load_existing_hotels() -> list[dict[str, Any]]:
    if not HOTELS_PATH.exists():
        return []
    try:
        data = json.loads(HOTELS_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    return []


def _save_hotels(cards: list[HotelCard], processed_cities: set[str], force: bool) -> None:
    existing = [] if force else _load_existing_hotels()
    preserved = [item for item in existing if item.get("city") not in processed_cities]
    new_cards = [card.model_dump() for card in cards]
    by_id: dict[str, dict[str, Any]] = {}
    for item in preserved + new_cards:
        item_id = item.get("id")
        if item_id:
            by_id[str(item_id)] = item

    output = sorted(by_id.values(), key=lambda item: (str(item.get("city", "")), str(item.get("name", ""))))
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    HOTELS_PATH.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved {len(output)} static hotel cards to {HOTELS_PATH}")


async def _run() -> int:
    args = _parse_args()
    settings = _load_build_settings()
    if not settings.rollinggo_mcp_url:
        print("ROLLINGGO_MCP_URL is required.")
        return 1
    if not settings.rollinggo_api_key:
        print("ROLLINGGO_API_KEY is required. Set it in .env before building.")
        return 1

    city_queries = _resolve_city_queries(args.city, args.limit_cities)
    if not city_queries:
        print("No cities selected.")
        return 0

    client = RollingGoMCPClient(
        url=settings.rollinggo_mcp_url,
        api_key=settings.rollinggo_api_key,
        accept_language=settings.rollinggo_accept_language,
    )

    if not await _preflight_client(client):
        return 1

    if not args.skip_tags:
        await _cache_tags(client)

    all_cards: list[HotelCard] = []
    processed_cities: set[str] = set()
    for city_query in city_queries:
        display_city = city_query_to_display_name(city_query)
        processed_cities.add(display_city)
        print(f"\nCollecting {display_city}...")
        try:
            cards, stats = await _collect_city(client, city_query, settings)
            all_cards.extend(cards)
            print(
                "{city}: collected before dedupe={before}, after dedupe={after}, "
                "final saved count={final}, errors={errors}".format(
                    city=stats["city"],
                    before=stats["collected_before_dedupe"],
                    after=stats["after_dedupe"],
                    final=stats["final_saved_count"],
                    errors=len(stats["errors"]),
                )
            )
            for error in stats["errors"]:
                print(f"  - {error}")
        except Exception as exc:
            print(
                f"{display_city}: collected before dedupe=0, after dedupe=0, "
                f"final saved count=0, errors=1"
            )
            print(f"  - city failed: {exc}")
            continue

    _save_hotels(all_cards, processed_cities, args.force)
    return 0


def main() -> int:
    try:
        return asyncio.run(_run())
    except RollingGoMCPError as exc:
        print(f"RollingGo MCP build failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
