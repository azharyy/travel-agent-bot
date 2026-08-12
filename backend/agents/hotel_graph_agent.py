"""LangGraph-based hotel assistant for local RAG and live availability."""

from __future__ import annotations

import asyncio
import os
import re
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import date
from typing import Any, TypedDict

from pydantic import BaseModel, Field

from backend.agents.hotel_tools import fetch_live_pricing, search_hotels
from backend.agents.query_rewriter import generate_query_rewrites
from backend.hotel_collector.cities import CITIES
from backend.hotel_collector.utils import canonicalize_city, city_query_to_display_name, safe_int

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - python-dotenv is declared in pyproject.
    load_dotenv = None


SYSTEM_PROMPT = (
    "You are an Egypt hotel travel assistant. Recommend hotels from static RAG first. "
    "Do not fetch live prices until the user selects a specific hotel. Do not show "
    "booking links in the first recommendation. Before live lookup collect check-in, "
    "check-out, total adults, children ages, and room count. If user says Sahel, "
    "treat it as North Coast. Do not claim prices are live unless live lookup was called."
)

HOTEL_TYPE_KEYWORDS = {
    "luxury": "luxury hotel",
    "premium": "luxury hotel",
    "5 star": "luxury hotel",
    "five star": "luxury hotel",
    "upscale": "luxury hotel",
    "beach": "beach resort",
    "beachfront": "beach resort",
    "sea view": "beach resort",
    "seaside": "beach resort",
    "family": "family resort",
    "kids": "family resort",
    "aqua": "family resort",
    "budget": "budget value hotel",
    "cheap": "budget value hotel",
    "affordable": "budget value hotel",
    "value": "budget value hotel",
    "romantic": "romantic spa resort",
    "couples": "romantic spa resort",
    "honeymoon": "romantic spa resort",
    "quiet": "romantic spa resort",
    "business": "business hotel",
    "work trip": "business hotel",
    "central": "business hotel",
    "nature": "nature hotel",
    "eco": "nature hotel",
    "desert": "nature hotel",
    "mountain": "nature hotel",
    "lake": "nature hotel",
    "oasis": "nature hotel",
    "spa": "spa resort",
    "wellness": "spa resort",
    "relaxation": "spa resort",
    "boutique": "boutique cultural stay",
    "cultural": "boutique cultural stay",
    "best": "general best hotels",
}

ORDINALS = {
    "first": 1,
    "1st": 1,
    "second": 2,
    "2nd": 2,
    "third": 3,
    "3rd": 3,
    "fourth": 4,
    "4th": 4,
    "fifth": 5,
    "5th": 5,
    "sixth": 6,
    "6th": 6,
    "seventh": 7,
    "7th": 7,
    "eighth": 8,
    "8th": 8,
    "ninth": 9,
    "9th": 9,
    "tenth": 10,
    "10th": 10,
}

AMBIGUOUS_DATE_WORDS = {
    "today",
    "tomorrow",
    "tonight",
    "next monday",
    "next tuesday",
    "next wednesday",
    "next thursday",
    "next friday",
    "next saturday",
    "next sunday",
    "this weekend",
    "next weekend",
}


class HotelAgentState(TypedDict, total=False):
    messages: list[dict[str, str]]
    city: str | None
    hotel_type: str | None
    query_rewrites: list[str]
    rag_results: list[dict[str, Any]]
    shown_offset: int
    selected_hotel: dict[str, Any] | None
    check_in: str | None
    check_out: str | None
    total_adults: int | None
    children_count: int | None
    children_ages: list[int]
    room_count: int | None
    live_availability: dict[str, Any] | None
    pending_question: str | None
    assistant_message: str
    properties: list[dict[str, Any]]
    pipeline_stage_reached: str


class IntentModel(BaseModel):
    city: str | None = None
    hotel_type: str | None = None
    selected_option: int | None = None
    selected_hotel_name: str | None = None
    show_more: bool = False
    check_in: str | None = None
    check_out: str | None = None
    total_adults: int | None = None
    children_count: int | None = None
    children_ages: list[int] = Field(default_factory=list)
    room_count: int | None = None
    ambiguous_date: bool = False


@dataclass
class ParsedIntent:
    city: str | None = None
    hotel_type: str | None = None
    selected_option: int | None = None
    selected_hotel_name: str | None = None
    show_more: bool = False
    check_in: str | None = None
    check_out: str | None = None
    total_adults: int | None = None
    children_count: int | None = None
    children_ages: list[int] = field(default_factory=list)
    room_count: int | None = None
    ambiguous_date: bool = False


def default_state() -> HotelAgentState:
    return {
        "messages": [],
        "city": None,
        "hotel_type": None,
        "query_rewrites": [],
        "rag_results": [],
        "shown_offset": 0,
        "selected_hotel": None,
        "check_in": None,
        "check_out": None,
        "total_adults": None,
        "children_count": None,
        "children_ages": [],
        "room_count": None,
        "live_availability": None,
        "pending_question": None,
        "assistant_message": "",
        "properties": [],
        "pipeline_stage_reached": "start",
    }


def _city_aliases() -> dict[str, str]:
    aliases = {
        "sahel": "North Coast",
        "north coast": "North Coast",
        "north coast egypt": "North Coast",
        "sharm": "Sharm El-Sheikh",
        "sharm el sheikh": "Sharm El-Sheikh",
        "sharm el-sheikh": "Sharm El-Sheikh",
    }
    for city_query in CITIES:
        display = city_query_to_display_name(city_query)
        aliases[display.lower()] = display
        aliases[city_query.lower()] = display
        aliases[city_query.lower().replace(" hotels", "")] = display
    return aliases


CITY_ALIASES = _city_aliases()


def _latest_user_message(state: HotelAgentState) -> str:
    for message in reversed(state.get("messages", [])):
        if message.get("role") == "user":
            return message.get("content", "")
    return ""


def _detect_city(text: str) -> str | None:
    lower = text.lower()
    for alias in sorted(CITY_ALIASES, key=len, reverse=True):
        if re.search(rf"\b{re.escape(alias)}\b", lower):
            return canonicalize_city(CITY_ALIASES[alias])
    return None


def _detect_hotel_type(text: str) -> str | None:
    lower = text.lower()
    matches: list[str] = []
    for keyword, hotel_type in HOTEL_TYPE_KEYWORDS.items():
        if keyword in lower and hotel_type not in matches:
            matches.append(hotel_type)

    if not matches:
        return None
    if "luxury hotel" in matches and "beach resort" in matches:
        return "luxury beach resort"
    if "family resort" in matches and "beach resort" in matches:
        return "family beach resort"
    if "romantic spa resort" in matches:
        return "romantic spa resort"
    if "spa resort" in matches and "luxury hotel" in matches:
        return "luxury spa resort"
    return matches[0]


def _detect_show_more(text: str) -> bool:
    lower = text.lower().strip()
    return lower in {"more", "show more", "next", "next 5", "more options"} or "show more" in lower


def _detect_selected_option(text: str, *, allow_bare_number: bool = True) -> int | None:
    lower = text.lower().strip()
    if allow_bare_number and re.fullmatch(r"\d{1,2}", lower):
        return int(lower)

    option_match = re.search(r"(?:option|#|number|pick|choose|select)\s*(\d{1,2})", lower)
    if option_match:
        return int(option_match.group(1))

    for word, number in ORDINALS.items():
        if re.search(rf"\b{re.escape(word)}\b", lower):
            return number
    return None


def _normalize_date_token(token: str) -> str | None:
    match = re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})", token.strip())
    if not match:
        return None
    year, month, day = (int(part) for part in match.groups())
    try:
        return date(year, month, day).isoformat()
    except ValueError:
        return None


def _detect_dates(text: str, state: HotelAgentState) -> tuple[str | None, str | None, bool]:
    date_tokens = re.findall(r"\b\d{4}-\d{1,2}-\d{1,2}\b", text)
    iso_dates = [normalized for token in date_tokens if (normalized := _normalize_date_token(token))]
    lower = text.lower()
    ambiguous = any(word in lower for word in AMBIGUOUS_DATE_WORDS) and not iso_dates

    if len(iso_dates) >= 2:
        return iso_dates[0], iso_dates[1], ambiguous
    if not iso_dates:
        return None, None, ambiguous

    single_date = iso_dates[0]
    mentions_checkout = bool(re.search(r"\b(check[-\s]?out|checkout|depart|leav(?:e|ing))\b", lower))
    mentions_checkin = bool(re.search(r"\b(check[-\s]?in|checkin|arriv(?:e|al|ing))\b", lower))

    if mentions_checkout and not mentions_checkin:
        return None, single_date, ambiguous
    if state.get("selected_hotel") and state.get("check_in") and not state.get("check_out"):
        return None, single_date, ambiguous

    check_in = single_date
    check_out = None
    return check_in, check_out, ambiguous


def _bare_int(text: str) -> int | None:
    match = re.fullmatch(r"\s*(\d{1,2})\s*", text)
    return safe_int(match.group(1)) if match else None


def _collecting_live_details(state: HotelAgentState) -> bool:
    return bool(state.get("selected_hotel")) and state.get("live_availability") is None


def _detect_adults(text: str, state: HotelAgentState) -> int | None:
    lower = text.lower()
    match = re.search(r"(\d+)\s*(?:adults?|adult guests?)\b", lower)
    if match:
        return safe_int(match.group(1))
    guest_match = re.search(r"(\d+)\s*(?:guests?|people)\b", lower)
    if guest_match:
        return safe_int(guest_match.group(1))
    bare = _bare_int(text)
    if (
        bare is not None
        and _collecting_live_details(state)
        and state.get("check_in")
        and state.get("check_out")
        and state.get("total_adults") is None
    ):
        return bare
    return None


def _detect_children_count(text: str, state: HotelAgentState) -> int | None:
    lower = text.lower()
    if re.search(r"\b(no|zero)\s+(?:children|kids|child)\b", lower):
        return 0
    match = re.search(r"(\d+)\s*(?:children|kids|child)\b", lower)
    if match:
        return safe_int(match.group(1))
    bare = _bare_int(text)
    if (
        bare is not None
        and _collecting_live_details(state)
        and state.get("check_in")
        and state.get("check_out")
        and state.get("total_adults") is not None
        and state.get("children_count") is None
    ):
        return bare
    return None


def _detect_children_ages(text: str, state: HotelAgentState) -> list[int]:
    lower = text.lower()
    match = re.search(r"(?:ages?|aged)\s+([0-9,\sand]+)", lower)
    if match:
        source = match.group(1)
    elif (
        _collecting_live_details(state)
        and (state.get("children_count") or 0) > 0
        and len(state.get("children_ages", [])) < (state.get("children_count") or 0)
    ):
        source = lower
    else:
        return []

    ages = []
    for number in re.findall(r"\d+", source):
        parsed = safe_int(number)
        if parsed is not None and 0 <= parsed <= 17:
            ages.append(parsed)
    return ages


def _detect_room_count(text: str, state: HotelAgentState) -> int | None:
    match = re.search(r"(\d+)\s*(?:rooms?|room count)\b", text.lower())
    if match:
        return safe_int(match.group(1))
    bare = _bare_int(text)
    children_count = state.get("children_count")
    children_ages = state.get("children_ages", [])
    child_details_done = children_count == 0 or (
        children_count is not None and len(children_ages) >= children_count
    )
    if (
        bare is not None
        and _collecting_live_details(state)
        and state.get("total_adults") is not None
        and child_details_done
        and state.get("room_count") is None
    ):
        return bare
    return None


def _parse_intent_deterministic(text: str, state: HotelAgentState) -> ParsedIntent:
    check_in, check_out, ambiguous_date = _detect_dates(text, state)
    allow_bare_option = not _collecting_live_details(state) or state.get("pending_question") == "hotel_selection"
    return ParsedIntent(
        city=_detect_city(text),
        hotel_type=_detect_hotel_type(text),
        selected_option=_detect_selected_option(text, allow_bare_number=allow_bare_option),
        selected_hotel_name=_detect_hotel_name(text, state.get("rag_results", [])),
        show_more=_detect_show_more(text),
        check_in=check_in,
        check_out=check_out,
        total_adults=_detect_adults(text, state),
        children_count=_detect_children_count(text, state),
        children_ages=_detect_children_ages(text, state),
        room_count=_detect_room_count(text, state),
        ambiguous_date=ambiguous_date,
    )


def _detect_hotel_name(text: str, rag_results: list[dict[str, Any]]) -> str | None:
    lower = text.lower().strip()
    if not lower or len(lower) < 4:
        return None
    for hotel in rag_results:
        name = str(hotel.get("name") or "").strip()
        if not name:
            continue
        name_lower = name.lower()
        if name_lower in lower or (len(lower) >= 6 and lower in name_lower):
            return name
    return None


def _intent_needs_gemini(intent: ParsedIntent, text: str, state: HotelAgentState) -> bool:
    if not text.strip() or intent.show_more or intent.selected_option or intent.selected_hotel_name:
        return False
    if intent.ambiguous_date:
        return False
    if not intent.city or not intent.hotel_type:
        return True
    if state.get("selected_hotel") and (
        not intent.check_in
        or not intent.check_out
        or intent.total_adults is None
        or intent.children_count is None
        or intent.room_count is None
    ):
        return True
    return False


def _merge_intents(base: ParsedIntent, extra: IntentModel | None) -> ParsedIntent:
    if extra is None:
        return base
    data = extra.model_dump()
    for field_name, value in data.items():
        if field_name == "children_ages":
            if value and not base.children_ages:
                base.children_ages = [age for age in value if isinstance(age, int)]
            continue
        if field_name == "show_more":
            base.show_more = base.show_more or bool(value)
            continue
        if field_name == "ambiguous_date":
            base.ambiguous_date = base.ambiguous_date or bool(value)
            continue
        if getattr(base, field_name) in (None, "") and value not in (None, ""):
            setattr(base, field_name, value)

    if base.city:
        base.city = canonicalize_city(base.city)
    return base


def _sanitize_hotels(hotels: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sanitized = []
    for hotel in hotels:
        item = dict(hotel)
        item.pop("booking_url", None)
        sanitized.append(item)
    return sanitized


def _append_assistant(state: HotelAgentState, message: str) -> None:
    messages = list(state.get("messages", []))
    messages.append({"role": "assistant", "content": message})
    state["messages"] = messages
    state["assistant_message"] = message


def _reset_selection(state: HotelAgentState) -> None:
    state["selected_hotel"] = None
    state["check_in"] = None
    state["check_out"] = None
    state["total_adults"] = None
    state["children_count"] = None
    state["children_ages"] = []
    state["room_count"] = None
    state["live_availability"] = None
    state["shown_offset"] = 0


def _reset_search(state: HotelAgentState) -> None:
    _reset_selection(state)
    state["query_rewrites"] = []
    state["rag_results"] = []
    state["properties"] = []


def _format_recommendations(hotels: list[dict[str, Any]], city: str, hotel_type: str) -> str:
    lines = [f"Here are the top {len(hotels)} local matches for {hotel_type} in {city}:"]
    for hotel in hotels:
        lines.append(
            "\n{number}. {name} ({city}, {hotel_type})\n"
            "{summary}\n"
            "Tags: {tags}\n"
            "Amenities: {amenities}".format(
                number=hotel.get("option_number"),
                name=hotel.get("name"),
                city=hotel.get("city"),
                hotel_type=hotel.get("hotel_type"),
                summary=hotel.get("summary") or "No summary available.",
                tags=hotel.get("tags_text") or "Not listed",
                amenities=hotel.get("amenities_text") or "Not listed",
            )
        )
    lines.append("\nWhich hotel should I check live availability for?")
    return "\n".join(lines)


def _format_more(hotels: list[dict[str, Any]]) -> str:
    if not hotels:
        return "There are no more local matches to show. Pick one of the hotels already listed."
    lines = ["Here are more local matches:"]
    for hotel in hotels:
        lines.append(
            "\n{number}. {name} ({city}, {hotel_type})\n{summary}".format(
                number=hotel.get("option_number"),
                name=hotel.get("name"),
                city=hotel.get("city"),
                hotel_type=hotel.get("hotel_type"),
                summary=hotel.get("summary") or "No summary available.",
            )
        )
    lines.append("\nChoose an option number when you want live availability.")
    return "\n".join(lines)


def _select_hotel(state: HotelAgentState, intent: ParsedIntent) -> dict[str, Any] | None:
    rag_results = state.get("rag_results", [])
    if intent.selected_option is not None:
        for hotel in rag_results:
            if safe_int(hotel.get("option_number")) == intent.selected_option:
                return hotel
    if intent.selected_hotel_name:
        target = intent.selected_hotel_name.lower()
        for hotel in rag_results:
            if str(hotel.get("name", "")).lower() == target:
                return hotel
    return None


def _apply_trip_details(state: HotelAgentState, intent: ParsedIntent) -> None:
    if intent.check_in:
        state["check_in"] = intent.check_in
        state["live_availability"] = None
    if intent.check_out:
        state["check_out"] = intent.check_out
        state["live_availability"] = None
    if intent.total_adults is not None:
        state["total_adults"] = intent.total_adults
        state["live_availability"] = None
    if intent.children_count is not None:
        state["children_count"] = intent.children_count
        state["live_availability"] = None
    if intent.children_ages:
        state["children_ages"] = intent.children_ages
        state["live_availability"] = None
    if intent.room_count is not None:
        state["room_count"] = intent.room_count
        state["live_availability"] = None


def _next_missing_detail_question(state: HotelAgentState, ambiguous_date: bool) -> str | None:
    if ambiguous_date:
        return "Please provide exact check-in and check-out dates in YYYY-MM-DD format."
    if not state.get("check_in"):
        return "What is your check-in date? Please use YYYY-MM-DD."
    if not state.get("check_out"):
        return "What is your check-out date? Please use YYYY-MM-DD."
    if state.get("total_adults") is None:
        return "How many adults are traveling?"
    if state.get("children_count") is None:
        return "How many children are traveling? Say 0 if none."
    children_count = state.get("children_count") or 0
    children_ages = state.get("children_ages", [])
    if children_count > 0 and len(children_ages) < children_count:
        return "What are the children's ages? Please list each age from 0 to 17."
    if state.get("room_count") is None:
        return "How many rooms do you need?"
    return None


def _format_selected_hotel(hotel: dict[str, Any]) -> str:
    return f"Great, I will check live availability for {hotel.get('name')}."


def _format_live_availability(selected_hotel: dict[str, Any], live: dict[str, Any]) -> str:
    rooms = live.get("rooms") or []
    if live.get("available") is True and rooms:
        lines = [
            "Live availability for {name} ({check_in} to {check_out}):".format(
                name=selected_hotel.get("name"),
                check_in=live.get("check_in"),
                check_out=live.get("check_out"),
            )
        ]
        for index, room in enumerate(rooms, start=1):
            price = room.get("price")
            currency = room.get("currency") or ""
            price_text = f"{price} {currency}" if price is not None else "Price not returned"
            inventory = room.get("inventory_count")
            cancellation = room.get("cancellation_summary") or "Cancellation details not returned"
            lines.append(
                "\n{index}. {room_name} - {rate_name}\n"
                "Price: {price}\n"
                "Inventory: {inventory}\n"
                "Cancellation: {cancellation}\n"
                "bookingUrl: {booking_url}".format(
                    index=index,
                    room_name=room.get("room_name") or "Room",
                    rate_name=room.get("rate_plan_name") or "Rate plan",
                    price=price_text,
                    inventory=inventory if inventory is not None else "Not returned",
                    cancellation=cancellation,
                    booking_url=room.get("booking_url") or live.get("booking_url") or "Not returned",
                )
            )
        return "\n".join(lines)

    if live.get("available") is False:
        return "I did not find live rooms for that hotel. Would you like to choose another hotel?"
    return live.get("message") or "Live availability could not be confirmed. Would you like another hotel?"


class HotelGraphAgent:
    """Stateful LangGraph hotel assistant."""

    def __init__(self) -> None:
        self.state = default_state()
        self.setup_error: str | None = None
        self.llm = None
        self.graph = None
        self._setup_llm_and_graph()

    def _setup_llm_and_graph(self) -> None:
        if load_dotenv is not None:
            load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            self.setup_error = "GEMINI_API_KEY is missing. Add it to .env before using the hotel agent."
            return

        try:
            from langchain_google_genai import ChatGoogleGenerativeAI

            self.llm = ChatGoogleGenerativeAI(
                model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
                temperature=0.2,
                google_api_key=api_key,
            )
        except Exception as exc:
            self.setup_error = f"Gemini setup failed: {exc}"
            return

        try:
            from langgraph.graph import END, START, StateGraph

            graph = StateGraph(HotelAgentState)
            graph.add_node("hotel_turn", self._hotel_turn)
            graph.add_edge(START, "hotel_turn")
            graph.add_edge("hotel_turn", END)
            self.graph = graph.compile()
        except Exception as exc:
            self.setup_error = f"LangGraph setup failed: {exc}"

    async def _parse_intent(self, text: str, state: HotelAgentState) -> ParsedIntent:
        intent = _parse_intent_deterministic(text, state)
        if not self.llm or not _intent_needs_gemini(intent, text, state):
            return intent

        try:
            from langchain_core.messages import HumanMessage, SystemMessage

            parser = self.llm.with_structured_output(IntentModel)
            gemini_intent = await parser.ainvoke(
                [
                    SystemMessage(
                        content=(
                            SYSTEM_PROMPT
                            + " Extract only hotel intent fields. Use null for unknown values. "
                            "Do not rewrite search queries."
                        )
                    ),
                    HumanMessage(content=text),
                ]
            )
            return _merge_intents(intent, gemini_intent)
        except Exception:
            return intent

    async def _hotel_turn(self, state: HotelAgentState) -> HotelAgentState:
        state = {**default_state(), **state}
        text = _latest_user_message(state)
        intent = await self._parse_intent(text, state)

        city_changed = bool(intent.city and intent.city != state.get("city"))
        type_changed = bool(intent.hotel_type and intent.hotel_type != state.get("hotel_type"))
        if city_changed or type_changed:
            _reset_search(state)
        if intent.city:
            state["city"] = canonicalize_city(intent.city)
        if intent.hotel_type:
            state["hotel_type"] = intent.hotel_type

        state["properties"] = []
        _apply_trip_details(state, intent)

        if not state.get("city"):
            message = "Which Egypt destination should I search in? For example: Cairo, Hurghada, Sharm El-Sheikh, Luxor, or North Coast."
            state["pending_question"] = "city"
            state["pipeline_stage_reached"] = "need_city"
            _append_assistant(state, message)
            return state

        if not state.get("hotel_type"):
            message = "What kind of hotel would you prefer: luxury, beach, family, budget, romantic, business, nature, or spa?"
            state["pending_question"] = "hotel_type"
            state["pipeline_stage_reached"] = "need_preference"
            _append_assistant(state, message)
            return state

        if not state.get("rag_results"):
            search_result = search_hotels(state["city"], state["hotel_type"], limit=5)
            if search_result.get("status") != "ok":
                state["pipeline_stage_reached"] = "rag_search_empty"
                state["pending_question"] = "hotel_database"
                _append_assistant(state, search_result.get("message", "No local hotel matches are available."))
                return state

            state["query_rewrites"] = search_result.get("queries", [])[1:]
            state["rag_results"] = search_result.get("show_more", [])
            shown = search_result.get("recommendations", [])[:5]
            state["shown_offset"] = len(shown)
            state["properties"] = _sanitize_hotels(shown)
            state["pipeline_stage_reached"] = "rag_recommendations"
            state["pending_question"] = "hotel_selection"
            _append_assistant(
                state,
                _format_recommendations(shown, state["city"], state["hotel_type"]),
            )
            return state

        if intent.show_more:
            offset = state.get("shown_offset", 0)
            next_hotels = state.get("rag_results", [])[offset : offset + 5]
            state["shown_offset"] = offset + len(next_hotels)
            state["properties"] = _sanitize_hotels(next_hotels)
            state["pipeline_stage_reached"] = "show_more"
            state["pending_question"] = "hotel_selection"
            _append_assistant(state, _format_more(next_hotels))
            return state

        selected = _select_hotel(state, intent)
        if selected:
            state["selected_hotel"] = selected
            state["live_availability"] = None
            state["pipeline_stage_reached"] = "hotel_selected"

        if not state.get("selected_hotel"):
            state["pipeline_stage_reached"] = "awaiting_hotel_selection"
            state["pending_question"] = "hotel_selection"
            _append_assistant(state, "Which hotel should I check live availability for? You can reply with an option number or hotel name.")
            return state

        missing_question = _next_missing_detail_question(state, intent.ambiguous_date)
        if missing_question:
            state["pending_question"] = "live_lookup_details"
            state["pipeline_stage_reached"] = "collecting_live_details"
            prefix = ""
            if selected:
                prefix = _format_selected_hotel(selected) + " "
            _append_assistant(state, prefix + missing_question)
            return state

        if state.get("live_availability") is None:
            selected_hotel = state["selected_hotel"] or {}
            live = await fetch_live_pricing(
                hotel_id=selected_hotel.get("hotel_id"),
                check_in=state["check_in"],
                check_out=state["check_out"],
                total_adults=state["total_adults"],
                children_ages=state.get("children_ages", []),
                room_count=state["room_count"],
                fallback_booking_url=selected_hotel.get("booking_url"),
            )
            state["live_availability"] = live
            state["pipeline_stage_reached"] = "live_availability"
            state["pending_question"] = None if live.get("available") else "hotel_selection"
            _append_assistant(state, _format_live_availability(selected_hotel, live))
            return state

        state["pipeline_stage_reached"] = "live_availability"
        _append_assistant(state, _format_live_availability(state["selected_hotel"] or {}, state["live_availability"] or {}))
        return state

    async def arun(self, user_message: str, state: HotelAgentState | None = None) -> dict[str, Any]:
        working_state = deepcopy(state if state is not None else self.state)
        messages = list(working_state.get("messages", []))
        messages.append({"role": "user", "content": user_message})
        working_state["messages"] = messages

        if self.setup_error:
            assistant_message = self.setup_error
            messages.append({"role": "assistant", "content": assistant_message})
            working_state["messages"] = messages
            working_state["assistant_message"] = assistant_message
            working_state["pipeline_stage_reached"] = "setup_error"
            result_state = working_state
        else:
            result_state = await self.graph.ainvoke(working_state)

        if state is None:
            self.state = result_state
        return self._response_dict(result_state)

    def run(self, user_message: str, state: HotelAgentState | None = None) -> dict[str, Any]:
        return asyncio.run(self.arun(user_message, state=state))

    @staticmethod
    def _response_dict(state: HotelAgentState) -> dict[str, Any]:
        return {
            "assistant_message": state.get("assistant_message", ""),
            "properties": state.get("properties", []),
            "live_availability": state.get("live_availability"),
            "pipeline_stage_reached": state.get("pipeline_stage_reached", "unknown"),
        }


async def _run_cli() -> None:
    agent = HotelGraphAgent()
    print("GuideMe Egypt hotel agent. Type 'quit' to exit.")
    while True:
        try:
            user_message = input("You: ").strip()
        except EOFError:
            break
        if user_message.lower() in {"quit", "exit"}:
            break
        if not user_message:
            continue
        response = await agent.arun(user_message)
        print(f"Assistant: {response['assistant_message']}")


def main() -> int:
    asyncio.run(_run_cli())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
