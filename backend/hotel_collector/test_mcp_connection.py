"""Manual smoke test for the external RollingGo MCP server."""

from __future__ import annotations

import asyncio
import os
import sys
from collections.abc import Mapping
from typing import Any

from dotenv import load_dotenv

from backend.hotel_collector.rollinggo_mcp_client import (
    RollingGoMCPClient,
    RollingGoMCPError,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _load_runtime_settings() -> tuple[str, str, str]:
    load_dotenv()
    try:
        from backend.core.config import get_settings

        settings = get_settings()
        url = settings.rollinggo_mcp_url
        api_key = settings.rollinggo_api_key
        accept_language = settings.rollinggo_accept_language
    except Exception:
        url = os.getenv("ROLLINGGO_MCP_URL", "http://127.0.0.1:8010/mcp")
        api_key = os.getenv("ROLLINGGO_API_KEY", "")
        accept_language = os.getenv("ROLLINGGO_ACCEPT_LANGUAGE", "en_US")

    return url.strip(), api_key.strip(), accept_language.strip() or "en_US"


def _pick(data: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in data and data[key] not in (None, ""):
            return data[key]
    return None


def _exists(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list | tuple | dict):
        return bool(value)
    return True


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


def _image_value(hotel: Mapping[str, Any]) -> Any:
    image = _pick(hotel, "imageUrl", "image_url", "mainImageUrl", "mainPhotoUrl", "photoUrl")
    if image:
        return image

    images = _pick(hotel, "images", "photos", "imageUrls")
    if isinstance(images, list) and images:
        first = images[0]
        if isinstance(first, Mapping):
            return _pick(first, "url", "imageUrl", "image_url")
        return first
    return None


def _print_hotel_row(hotel: Mapping[str, Any]) -> None:
    hotel_id = _pick(hotel, "hotelId", "hotel_id", "id", "providerHotelId") or "<missing>"
    name = _pick(hotel, "nameEn", "name", "hotelName", "title") or "<missing>"
    star_rating = _pick(hotel, "starRating", "star_rating", "stars") or "<missing>"
    booking_url = _pick(hotel, "bookingUrl", "booking_url", "url")
    image_url = _image_value(hotel)
    print(
        "hotelId={hotel_id} | name={name} | starRating={star_rating} | "
        "imageUrl exists={image_exists} | bookingUrl exists={booking_exists}".format(
            hotel_id=hotel_id,
            name=name,
            star_rating=star_rating,
            image_exists=_exists(image_url),
            booking_exists=_exists(booking_url),
        )
    )


async def main() -> int:
    url, api_key, accept_language = _load_runtime_settings()
    if not url:
        print("Skipping RollingGo MCP connection test: ROLLINGGO_MCP_URL is not set.")
        return 0
    if not api_key:
        print("Skipping RollingGo MCP connection test: ROLLINGGO_API_KEY is not set.")
        return 0

    client = RollingGoMCPClient(url=url, api_key=api_key, accept_language=accept_language)

    try:
        tools = await client.list_tools()
        print("RollingGo MCP tools:")
        for tool in tools:
            print(f"- {tool.get('name') or '<unnamed>'}")

        print("\nTrying getHotelSearchTags...")
        try:
            tags = await client.get_hotel_search_tags()
            if isinstance(tags, Mapping):
                print(f"getHotelSearchTags returned keys: {', '.join(sorted(map(str, tags.keys())))}")
            else:
                print(f"getHotelSearchTags returned {type(tags).__name__}")
        except RollingGoMCPError as exc:
            print(f"getHotelSearchTags failed: {exc}")

        print("\nTrying searchHotels for Hurghada...")
        result = await client.search_hotels(
            {
                "originQuery": "Hurghada hotels general best hotels",
                "place": "Hurghada",
                "placeType": "city",
                "checkInParam": {
                    "checkInDate": "2026-08-01",
                    "stayNights": 2,
                    "adultCount": 2,
                },
                "filterOptions": {"starRatings": [0.0, 5.0]},
                "hotelTags": {
                    "preferredTags": [],
                    "requiredTags": [],
                    "excludedTags": [],
                },
                "size": 3,
                "countryCode": "EG",
            }
        )
        hotels = _find_hotel_items(result)
        if not hotels:
            print("No hotel rows found in searchHotels response.")
            return 0

        for hotel in hotels[:3]:
            _print_hotel_row(hotel)

    except RollingGoMCPError as exc:
        print(f"RollingGo MCP connection test failed: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
