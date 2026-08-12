"""Optional local MCP adapter for RollingGo hotel tools.

This module does not scrape booking sites and does not replace GuideMe's
backend. It exposes RollingGo-compatible MCP tools on a separate local port and
forwards calls to a configured RollingGo HTTP API.

Run:
    python -m backend.rollinggo_mcp_server
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urljoin

import httpx
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP


load_dotenv()


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _env_int(name: str, default: int) -> int:
    value = _env(name)
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


@dataclass(frozen=True)
class RollingGoAdapterSettings:
    host: str = _env("ROLLINGGO_LOCAL_MCP_HOST", "127.0.0.1")
    port: int = _env_int("ROLLINGGO_LOCAL_MCP_PORT", 8010)
    api_base_url: str = _env("ROLLINGGO_API_BASE_URL")
    api_key: str = _env("ROLLINGGO_API_KEY")
    accept_language: str = _env("ROLLINGGO_ACCEPT_LANGUAGE", "en_US") or "en_US"
    search_hotels_path: str = _env("ROLLINGGO_SEARCH_HOTELS_PATH", "/searchHotels")
    hotel_detail_path: str = _env("ROLLINGGO_HOTEL_DETAIL_PATH", "/getHotelDetail")
    search_tags_path: str = _env("ROLLINGGO_SEARCH_TAGS_PATH", "/getHotelSearchTags")
    api_key_header: str = _env("ROLLINGGO_API_KEY_HEADER", "Authorization")
    api_key_prefix: str = _env("ROLLINGGO_API_KEY_PREFIX", "Bearer")
    timeout_seconds: float = float(_env("ROLLINGGO_API_TIMEOUT_SECONDS", "30") or "30")


SETTINGS = RollingGoAdapterSettings()

mcp = FastMCP(
    name="rollinggo-hotels",
    instructions=(
        "RollingGo-compatible hotel MCP adapter for GuideMe. "
        "Tools forward to the configured RollingGo HTTP API."
    ),
    host=SETTINGS.host,
    port=SETTINGS.port,
    streamable_http_path="/mcp",
    json_response=False,
    stateless_http=False,
)


def _mask_secret(secret: str) -> str:
    if not secret:
        return "<missing>"
    if len(secret) <= 8:
        return "<set>"
    return f"{secret[:4]}...{secret[-4:]}"


def _redact(value: str) -> str:
    redacted = value
    if SETTINGS.api_key:
        redacted = redacted.replace(SETTINGS.api_key, _mask_secret(SETTINGS.api_key))
    return re.sub(r"Bearer\s+[A-Za-z0-9._~+/=-]+", "Bearer <redacted>", redacted)


def _provider_url(path: str) -> str:
    base = SETTINGS.api_base_url.rstrip("/") + "/"
    return urljoin(base, path.lstrip("/"))


def _headers() -> dict[str, str]:
    headers = {
        "Accept": "application/json",
        "Accept-Language": SETTINGS.accept_language,
        "Content-Type": "application/json",
        "User-Agent": "GuideMe-RollingGo-MCP-Adapter/1.0",
    }
    if SETTINGS.api_key:
        prefix = SETTINGS.api_key_prefix.strip()
        value = f"{prefix} {SETTINGS.api_key}".strip() if prefix else SETTINGS.api_key
        headers[SETTINGS.api_key_header or "Authorization"] = value
    return headers


def _without_none(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if value is not None}


def _safe_body(text: str, limit: int = 700) -> str:
    text = _redact(text.strip())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


async def _call_provider(
    path: str,
    payload: dict[str, Any] | None = None,
    *,
    allow_get: bool = False,
) -> Any:
    payload = payload or {}
    if not SETTINGS.api_base_url:
        return {
            "success": False,
            "provider": "rollinggo",
            "setup_required": True,
            "error": (
                "ROLLINGGO_API_BASE_URL is not set. The local MCP adapter is running, "
                "but it needs RollingGo's HTTP API base URL to return real hotel data."
            ),
        }
    if not SETTINGS.api_key:
        return {
            "success": False,
            "provider": "rollinggo",
            "setup_required": True,
            "error": "ROLLINGGO_API_KEY is not set.",
        }

    url = _provider_url(path)
    timeout = httpx.Timeout(SETTINGS.timeout_seconds)

    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        try:
            if allow_get and not payload:
                response = await client.get(url, headers=_headers())
            else:
                response = await client.post(url, headers=_headers(), json=payload)
        except httpx.HTTPError as exc:
            return {
                "success": False,
                "provider": "rollinggo",
                "error": f"RollingGo HTTP request failed: {_redact(str(exc))}",
            }

    if response.status_code >= 400:
        return {
            "success": False,
            "provider": "rollinggo",
            "status_code": response.status_code,
            "error": (
                "RollingGo HTTP request failed with status "
                f"{response.status_code}: {_safe_body(response.text)}"
            ),
        }

    content_type = response.headers.get("content-type", "")
    if "application/json" in content_type.lower():
        return response.json()

    text = response.text.strip()
    if not text:
        return {}
    return {"text": _safe_body(text)}


@mcp.tool(
    name="getHotelSearchTags",
    description="Return hotel search tags and amenities from RollingGo.",
)
async def get_hotel_search_tags() -> Any:
    return await _call_provider(SETTINGS.search_tags_path, allow_get=True)


@mcp.tool(
    name="searchHotels",
    description="Search RollingGo hotels for static discovery cards.",
)
async def search_hotels(
    originQuery: str | None = None,
    place: str | None = None,
    placeType: str | None = None,
    countryCode: str | None = None,
    size: int | None = None,
    checkInParam: dict[str, Any] | None = None,
    filterOptions: dict[str, Any] | None = None,
    hotelTags: dict[str, Any] | None = None,
    query: str | None = None,
    checkInDate: str | None = None,
    stayNights: int | None = None,
    adultCount: int | None = None,
) -> Any:
    resolved_query = originQuery or query
    resolved_place = place or query or originQuery
    resolved_check_in = checkInParam or _without_none(
        {
            "checkInDate": checkInDate,
            "stayNights": stayNights,
            "adultCount": adultCount,
        }
    )

    payload = _without_none(
        {
            "originQuery": resolved_query,
            "place": resolved_place,
            "placeType": placeType or "\u57ce\u5e02",
            "countryCode": countryCode or "EG",
            "size": size or 20,
            "checkInParam": resolved_check_in or None,
            "filterOptions": filterOptions,
            "hotelTags": hotelTags,
        }
    )
    return await _call_provider(SETTINGS.search_hotels_path, payload)


@mcp.tool(
    name="getHotelDetail",
    description="Return live room availability and booking detail for one RollingGo hotel.",
)
async def get_hotel_detail(
    hotelId: int | str,
    dateParam: dict[str, Any] | None = None,
    occupancyParam: dict[str, Any] | None = None,
    localeParam: dict[str, Any] | None = None,
    checkInDate: str | None = None,
    checkOutDate: str | None = None,
    adultCount: int | None = None,
    childCount: int | None = None,
    childAgeDetails: list[int] | None = None,
    roomCount: int | None = None,
    countryCode: str | None = None,
    currency: str | None = None,
) -> Any:
    resolved_date = dateParam or _without_none(
        {
            "checkInDate": checkInDate,
            "checkOutDate": checkOutDate,
        }
    )
    resolved_occupancy = occupancyParam or _without_none(
        {
            "adultCount": adultCount,
            "childCount": childCount,
            "childAgeDetails": childAgeDetails,
            "roomCount": roomCount,
        }
    )
    resolved_locale = localeParam or _without_none(
        {
            "countryCode": countryCode or "EG",
            "currency": currency or "EGP",
        }
    )
    payload = _without_none(
        {
            "hotelId": hotelId,
            "dateParam": resolved_date or None,
            "occupancyParam": resolved_occupancy or None,
            "localeParam": resolved_locale or None,
        }
    )
    return await _call_provider(SETTINGS.hotel_detail_path, payload)


def main() -> None:
    print(
        "Starting RollingGo local MCP adapter at "
        f"http://{SETTINGS.host}:{SETTINGS.port}/mcp"
    )
    print(f"RollingGo API base URL: {SETTINGS.api_base_url or '<missing>'}")
    print(f"RollingGo API key: {_mask_secret(SETTINGS.api_key)}")
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
