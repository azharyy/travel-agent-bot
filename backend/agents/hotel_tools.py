"""Agent tools for local hotel RAG search and live RollingGo availability."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from datetime import date, datetime
from math import ceil
from typing import Any
from urllib.parse import urlparse

import numpy as np

from backend.agents.query_rewriter import generate_query_rewrites
from backend.core.config import settings
from backend.core.database import get_chroma_client, get_hotel_collection_name
from backend.hotel_collector.ingest_chroma import CLIP_MODEL_NAME
from backend.hotel_collector.rollinggo_mcp_client import RollingGoMCPClient
from backend.hotel_collector.utils import canonicalize_city, safe_float, safe_int


_CLIP_MODEL = None
INTERNAL_RETRIEVAL_LIMIT = 20


def _load_clip_model():
    global _CLIP_MODEL
    if _CLIP_MODEL is None:
        from sentence_transformers import SentenceTransformer

        _CLIP_MODEL = SentenceTransformer(CLIP_MODEL_NAME)
    return _CLIP_MODEL


def _normalize_vector(vector: Any) -> list[float]:
    array = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(array))
    if norm != 0.0:
        array = array / norm
    return array.tolist()


def _query_embedding(query: str) -> list[float]:
    model = _load_clip_model()
    return _normalize_vector(model.encode(query, convert_to_numpy=True))


def _parse_jsonish(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped:
        return value
    if not ((stripped.startswith("[") and stripped.endswith("]")) or (
        stripped.startswith("{") and stripped.endswith("}")
    )):
        return value
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return value


def _as_list(value: Any) -> list[str]:
    parsed = _parse_jsonish(value)
    if parsed is None:
        return []
    if isinstance(parsed, str):
        return [parsed] if parsed else []
    if isinstance(parsed, Mapping):
        return [str(key) for key, enabled in parsed.items() if enabled]
    if isinstance(parsed, list | tuple | set):
        return [str(item) for item in parsed if str(item)]
    return [str(parsed)]


def _join_text(value: Any) -> str:
    return ", ".join(_as_list(value))


def _clean_url(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    text = value.strip()
    if not text or text in {"{}", "[]", "null", "None", "none"}:
        return ""
    parsed = urlparse(text)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return ""
    return text


def _build_hotel_option(metadata: Mapping[str, Any], option_number: int) -> dict[str, Any]:
    return {
        "option_number": option_number,
        "name": metadata.get("name", ""),
        "city": metadata.get("city", ""),
        "hotel_type": metadata.get("hotel_type", ""),
        "summary": metadata.get("summary", ""),
        "tags_text": _join_text(metadata.get("tags")),
        "amenities_text": _join_text(metadata.get("amenities")),
        "image_url": _clean_url(metadata.get("image_url", "")),
        "hotel_id": metadata.get("hotel_id", ""),
        "booking_url": _clean_url(metadata.get("booking_url", "")),
    }


def _hotel_key(metadata: Mapping[str, Any]) -> str:
    return str(metadata.get("hotel_id") or metadata.get("id") or metadata.get("name") or "")


def _is_exact_city(metadata: Mapping[str, Any], city: str) -> bool:
    return str(metadata.get("city", "")).casefold() == city.casefold()


def _empty_collection_response(collection_name: str) -> dict[str, Any]:
    return {
        "status": "empty",
        "message": (
            f"No local hotel recommendations are available in Chroma collection "
            f"'{collection_name}'. Build and ingest the local hotel database first."
        ),
        "commands": [
            "python -m backend.hotel_collector.build_hotels_db",
            "python -m backend.hotel_collector.ingest_chroma --reset-collection --skip-images",
        ],
        "recommendations": [],
        "show_more": [],
    }


def _query_collection(collection, queries: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for query_index, query in enumerate(queries):
        result = collection.query(
            query_embeddings=[_query_embedding(query)],
            n_results=INTERNAL_RETRIEVAL_LIMIT,
            include=["documents", "metadatas", "distances"],
        )
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        for rank, metadata in enumerate(metadatas):
            if not metadata:
                continue
            distance = distances[rank] if rank < len(distances) else 1.0
            document = documents[rank] if rank < len(documents) else ""
            rows.append(
                {
                    "metadata": dict(metadata),
                    "document": document,
                    "distance": float(distance),
                    "query_index": query_index,
                    "rank": rank,
                }
            )
    return rows


def _get_exact_city_candidates(collection, city: str) -> list[dict[str, Any]]:
    try:
        result = collection.get(
            where={"city": {"$eq": city}},
            include=["documents", "metadatas"],
        )
    except Exception:
        return []

    metadatas = result.get("metadatas") or []
    documents = result.get("documents") or []
    rows = []
    for index, metadata in enumerate(metadatas):
        if not metadata:
            continue
        document = documents[index] if index < len(documents) else ""
        rows.append(
            {
                "metadata": dict(metadata),
                "document": document,
                "score": 0.0,
                "best_distance": 1.0,
                "match_count": 0,
            }
        )
    return rows


def _aggregate_rows(rows: list[dict[str, Any]], city: str) -> list[dict[str, Any]]:
    by_hotel: dict[str, dict[str, Any]] = {}
    for row in rows:
        metadata = row["metadata"]
        hotel_id = _hotel_key(metadata)
        if not hotel_id:
            continue

        similarity = 1.0 - float(row["distance"])
        rank_score = 1.0 / (row["rank"] + 1)
        exact_city_boost = 0.35 if _is_exact_city(metadata, city) else 0.0
        query_weight = 1.15 if row["query_index"] == 0 else 1.0
        score = (similarity * query_weight) + rank_score + exact_city_boost

        existing = by_hotel.get(hotel_id)
        if existing is None:
            by_hotel[hotel_id] = {
                "metadata": metadata,
                "score": score,
                "best_distance": row["distance"],
                "match_count": 1,
            }
            continue

        existing["score"] += score * 0.65
        existing["match_count"] += 1
        if row["distance"] < existing["best_distance"]:
            existing["best_distance"] = row["distance"]
            existing["metadata"] = metadata

    ranked = sorted(
        by_hotel.values(),
        key=lambda item: (item["score"], item["match_count"], -item["best_distance"]),
        reverse=True,
    )
    return ranked[:INTERNAL_RETRIEVAL_LIMIT]


def _prioritize_exact_city(
    collection,
    ranked: list[dict[str, Any]],
    city: str,
    display_limit: int,
) -> list[dict[str, Any]]:
    exact = [item for item in ranked if _is_exact_city(item["metadata"], city)]
    non_exact = [item for item in ranked if not _is_exact_city(item["metadata"], city)]

    if len(exact) < display_limit:
        seen = {_hotel_key(item["metadata"]) for item in ranked}
        for candidate in _get_exact_city_candidates(collection, city):
            key = _hotel_key(candidate["metadata"])
            if not key or key in seen:
                continue
            exact.append(candidate)
            seen.add(key)
            if len(exact) >= display_limit:
                break

    return (exact + non_exact)[:INTERNAL_RETRIEVAL_LIMIT]


def search_hotels(location: str, hotel_type: str, limit: int | None = None) -> dict[str, Any]:
    """Search local egypt_hotels Chroma recommendations only."""
    if not location or not str(location).strip():
        return {"status": "error", "message": "location is required.", "recommendations": [], "show_more": []}
    if not hotel_type or not str(hotel_type).strip():
        return {"status": "error", "message": "hotel_type is required.", "recommendations": [], "show_more": []}

    city = canonicalize_city(str(location).strip())
    hotel_type_text = " ".join(str(hotel_type).split())
    collection_name = get_hotel_collection_name()

    try:
        collection = get_chroma_client().get_collection(collection_name)
    except Exception:
        return _empty_collection_response(collection_name)

    if collection.count() == 0:
        return _empty_collection_response(collection_name)

    queries = [f"{hotel_type_text} in {city}"] + generate_query_rewrites(city, hotel_type_text)
    rows = _query_collection(collection, queries)
    ranked = _aggregate_rows(rows, city)
    display_limit = limit if limit is not None else settings.hotel_display_top_n
    display_limit = max(1, min(int(display_limit), INTERNAL_RETRIEVAL_LIMIT))
    ranked = _prioritize_exact_city(collection, ranked, city, display_limit)

    show_more = [
        _build_hotel_option(item["metadata"], option_number=index + 1)
        for index, item in enumerate(ranked)
    ]
    exact_city_recommendations = [
        option for option in show_more if str(option.get("city", "")).casefold() == city.casefold()
    ]
    recommendations = (
        exact_city_recommendations[:display_limit]
        if exact_city_recommendations
        else show_more[:display_limit]
    )

    if not recommendations:
        return {
            "status": "empty",
            "message": f"No local hotel matches found for {hotel_type_text} in {city}.",
            "queries": queries,
            "recommendations": [],
            "show_more": [],
        }

    return {
        "status": "ok",
        "message": (
            f"Found {len(recommendations)} local hotel recommendations for "
            f"{hotel_type_text} in {city}. Select a hotel and provide dates/guests "
            "to check live availability."
        ),
        "city": city,
        "hotel_type": hotel_type_text,
        "queries": queries,
        "recommendations": recommendations,
        "show_more": show_more,
        "total_found": len(show_more),
    }


def search_properties(location: str, description: str) -> dict[str, Any]:
    """Compatibility wrapper for older callers."""
    return search_hotels(location=location, hotel_type=description)


def _parse_iso_date(value: str, field_name: str) -> date:
    if not value or not isinstance(value, str):
        raise ValueError(f"{field_name} is required in YYYY-MM-DD format.")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a valid ISO date: YYYY-MM-DD.") from exc


def _validate_live_pricing_inputs(
    hotel_id: Any,
    check_in: str,
    check_out: str,
    total_adults: Any,
    children_ages: Any,
    room_count: Any,
) -> tuple[list[str], date | None, date | None, int, list[int], int]:
    errors: list[str] = []

    if hotel_id in (None, ""):
        errors.append("hotel_id is required.")

    check_in_date = None
    check_out_date = None
    try:
        check_in_date = _parse_iso_date(check_in, "check_in")
    except ValueError as exc:
        errors.append(str(exc))
    try:
        check_out_date = _parse_iso_date(check_out, "check_out")
    except ValueError as exc:
        errors.append(str(exc))

    if check_in_date and check_out_date and check_out_date <= check_in_date:
        errors.append("check_out must be after check_in.")
    if check_in_date and check_in_date < datetime.now().date():
        errors.append("check_in must not be in the past.")

    adults = safe_int(total_adults)
    if adults is None or adults < 1:
        errors.append("total_adults must be at least 1.")
        adults = 1

    rooms = safe_int(room_count)
    if rooms is None or rooms < 1:
        errors.append("room_count must be at least 1.")
        rooms = 1

    if children_ages is None:
        ages: list[int] = []
    elif isinstance(children_ages, list | tuple):
        ages = []
        for age in children_ages:
            parsed_age = safe_int(age)
            if parsed_age is None or parsed_age < 0 or parsed_age > 17:
                errors.append("children_ages must contain only ages from 0 to 17.")
            else:
                ages.append(parsed_age)
    else:
        errors.append("children_ages must be a list of ages from 0 to 17.")
        ages = []

    return errors, check_in_date, check_out_date, adults, ages, rooms


def _numeric_hotel_id_if_possible(hotel_id: Any) -> int | str:
    text = str(hotel_id).strip()
    return int(text) if text.isdigit() else text


def _find_first_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if isinstance(value, list | tuple):
        for item in value:
            if isinstance(item, Mapping):
                return item
    return {}


def _find_detail_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        if any(
            key in value
            for key in (
                "bookingUrl",
                "booking_url",
                "roomRatePlans",
                "room_rate_plans",
                "ratePlans",
                "rooms",
                "currency",
                "currencyCode",
            )
        ):
            return value
        for key in ("data", "result", "payload", "hotel", "detail"):
            nested = value.get(key)
            found = _find_detail_mapping(nested)
            if found:
                return found
    elif isinstance(value, list | tuple):
        for item in value:
            found = _find_detail_mapping(item)
            if found:
                return found
    return _find_first_mapping(value)


def _find_room_rate_plans(value: Any) -> list[Mapping[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, Mapping)]
    if not isinstance(value, Mapping):
        return []

    for key in (
        "roomRatePlans",
        "room_rate_plans",
        "ratePlans",
        "rate_plans",
        "rooms",
        "roomList",
        "roomRatePlanList",
        "data",
        "result",
        "payload",
        "hotel",
        "detail",
    ):
        nested = value.get(key)
        if isinstance(nested, list):
            return [item for item in nested if isinstance(item, Mapping)]
        if isinstance(nested, Mapping):
            found = _find_room_rate_plans(nested)
            if found:
                return found
    return []


def _pick(data: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        value = data.get(key)
        if value not in (None, ""):
            return value
    return None


def _extract_currency(plan: Mapping[str, Any], detail: Mapping[str, Any]) -> str:
    price_value = _pick(plan, "totalSalesRate", "totalPrice")
    price_currency = None
    if isinstance(price_value, Mapping):
        price_currency = _pick(price_value, "currency", "currencyCode")
    return str(
        _pick(plan, "currency", "currencyCode")
        or price_currency
        or _pick(detail, "currency", "currencyCode")
        or settings.hotel_currency
    )


def _extract_price(plan: Mapping[str, Any]) -> float | None:
    price_value = _pick(plan, "totalSalesRate", "totalPrice")
    if isinstance(price_value, Mapping):
        price_value = _pick(price_value, "amount", "value", "price")
    return safe_float(price_value)


def _normalize_room(plan: Mapping[str, Any], detail: Mapping[str, Any], fallback_booking_url: str | None) -> dict[str, Any]:
    booking_url = _pick(plan, "bookingUrl", "booking_url", "url") or _pick(
        detail,
        "bookingUrl",
        "booking_url",
        "url",
    ) or fallback_booking_url
    cancellation = _pick(
        plan,
        "cancellationSummary",
        "cancellation_summary",
        "cancellationPolicy",
        "cancelPolicy",
    )
    if isinstance(cancellation, Mapping):
        cancellation = _pick(cancellation, "summary", "description", "name")

    return {
        "room_name": _pick(plan, "roomName", "room_name", "name"),
        "rate_plan_name": _pick(plan, "ratePlanName", "rate_plan_name", "rateName", "boardName"),
        "price": _extract_price(plan),
        "currency": _extract_currency(plan, detail),
        "inventory_count": safe_int(_pick(plan, "inventoryCount", "inventory_count", "inventory")),
        "is_on_request": bool(_pick(plan, "isOnRequest", "is_on_request", "onRequest") or False),
        "cancellation_summary": cancellation,
        "booking_url": booking_url,
    }


def _provider_success(value: Any) -> bool | None:
    if isinstance(value, Mapping):
        success = _pick(value, "success", "isSuccess", "ok")
        if isinstance(success, bool):
            return success
        for key in ("data", "result", "payload", "hotel", "detail"):
            nested = value.get(key)
            nested_success = _provider_success(nested)
            if nested_success is not None:
                return nested_success
    if isinstance(value, list | tuple):
        for item in value:
            nested_success = _provider_success(item)
            if nested_success is not None:
                return nested_success
    return None


async def fetch_live_pricing(
    hotel_id: Any,
    check_in: str,
    check_out: str,
    total_adults: int,
    children_ages: list[int] | None,
    room_count: int = 1,
    fallback_booking_url: str | None = None,
) -> dict[str, Any]:
    """Fetch live room pricing for a selected hotel/date/guest request."""
    errors, check_in_date, check_out_date, adults, ages, rooms = _validate_live_pricing_inputs(
        hotel_id,
        check_in,
        check_out,
        total_adults,
        children_ages,
        room_count,
    )
    if errors:
        return {
            "status": "invalid",
            "available": None,
            "hotel_id": hotel_id,
            "rooms": [],
            "errors": errors,
            "message": "Live pricing was not checked because the request is invalid.",
        }

    payload = {
        "hotelId": _numeric_hotel_id_if_possible(hotel_id),
        "dateParam": {
            "checkInDate": check_in_date.isoformat(),
            "checkOutDate": check_out_date.isoformat(),
        },
        "occupancyParam": {
            "adultCount": max(1, ceil(adults / rooms)),
            "childCount": len(ages),
            "childAgeDetails": ages,
            "roomCount": rooms,
        },
        "localeParam": {
            "countryCode": settings.hotel_country_code,
            "currency": settings.hotel_currency,
        },
    }

    try:
        detail_result = await RollingGoMCPClient().get_hotel_detail(payload)
    except Exception as exc:
        error_message = str(exc)
        return {
            "status": "provider_error",
            "available": None,
            "hotel_id": hotel_id,
            "rooms": [],
            "errors": [error_message],
            "message": f"Live availability could not be checked: {error_message}",
        }

    detail = _find_detail_mapping(detail_result)
    room_plans = _find_room_rate_plans(detail_result)
    detail_booking_url = _pick(detail, "bookingUrl", "booking_url", "url") or fallback_booking_url
    rooms_payload = [
        _normalize_room(plan, detail, fallback_booking_url=detail_booking_url)
        for plan in room_plans
    ]

    success = _provider_success(detail_result)
    if rooms_payload:
        available = True
        message = "Live room options are available for this hotel."
    elif success is True:
        available = False
        message = "No live rooms were returned. Would you like to try another hotel?"
    else:
        available = None
        message = "Live availability could not be confirmed. Would you like to try another hotel?"

    return {
        "status": "ok" if available is not None else "provider_error",
        "available": available,
        "hotel_id": hotel_id,
        "check_in": check_in_date.isoformat(),
        "check_out": check_out_date.isoformat(),
        "adult_count_per_room": max(1, ceil(adults / rooms)),
        "child_count": len(ages),
        "room_count": rooms,
        "booking_url": detail_booking_url,
        "rooms": rooms_payload,
        "message": message,
    }


def fetch_live_pricing_sync(*args, **kwargs) -> dict[str, Any]:
    """Optional sync wrapper for scripts/tests that cannot await."""
    return asyncio.run(fetch_live_pricing(*args, **kwargs))
