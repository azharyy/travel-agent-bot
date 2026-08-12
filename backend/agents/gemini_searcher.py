# backend/gemini_search.py
from google import genai
import httpx
import json
from backend.core.config import settings
from backend.core.schemas import (
    VectorSearchResult, PropertyRecord, PropertyType,
    GeocodingResult, DistanceResult, HydrationRequest
)
from backend.core import database


class GeminiSearcher:
    """
    Agent 1 — The Decision Maker.

    RESPONSIBILITIES:
    1. Parse the user's natural language query for intent, location, and constraints
    2. Query ChromaDB using semantic vector search
    3. Geocode locations using Nominatim (free OSM API)
    4. Calculate distances using OSRM (free routing API)
    5. Decide whether live hydration (Scrapling) is needed
    """

    def __init__(self):
        # Stateful SDK initialization pattern passing the API key directly
        self.client = genai.Client(api_key=settings.gemini_api_key)
        
        # FIX: Dynamically strip out legacy "models/" prefix layout syntax if present
        configured_model = settings.gemini_model
        if configured_model.startswith("models/"):
            self.model_name = configured_model.replace("models/", "", 1)
        else:
            self.model_name = configured_model

    async def parse_query_intent(self, user_message: str, conversation_history: list) -> dict:
        """
        Uses Gemini to extract structured intent from natural language.
        Leverages the modern SDK's native schema enforcement.
        """
        history_text = ""
        if conversation_history:
            recent = conversation_history[-4:]  # Last 4 turns for context
            history_text = "\n".join([
                f"{m['role'].upper()}: {m['content']}"
                for m in recent
            ])

        system_instruction = (
            "You are a travel query parser. Your job is to extract structured information "
            "from the traveler's message and return a clean data object matching the requested schema."
        )

        user_prompt = f"""
Conversation history (for context):
{history_text}

User message: "{user_message}"

Rules for fields:
- wants_live_prices is true if user says: current price, live rates, available, book, tonight, this weekend, or asks for prices right now.
- distance_query origin and destination should remain null unless the user explicitly asks about distances or routes.
- search_query should be a descriptive natural language phrasing optimized for semantic search vector databases.
"""
        
        # Define an explicit structure inline using Pydantic or basic types
        # to guarantee the shape returned by the Gemini engine.
        try:
            from pydantic import BaseModel, Field
            from typing import Optional, List

            class DistanceQuerySchema(BaseModel):
                origin: Optional[str] = None
                destination: Optional[str] = None

            class IntentSchema(BaseModel):
                city: Optional[str] = None
                country: Optional[str] = None
                property_type: Optional[str] = None
                max_budget_usd: Optional[float] = None
                amenities_requested: List[str] = Field(default_factory=list)
                wants_live_prices: bool = False
                distance_query: DistanceQuerySchema
                search_query: str

            # OPTIMIZATION: Request structured output natively from the API model
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=user_prompt,
                config={
                    "system_instruction": system_instruction,
                    "response_mime_type": "application/json",
                    "response_schema": IntentSchema,
                    "temperature": 0.1
                }
            )
            
            # The .text value is guaranteed to conform exactly to IntentSchema layout
            return json.loads(response.text)
            
        except Exception as e:
            print(f"[GeminiSearcher] Native intent parsing optimization failed: {e}")
            # Fallback: treat entire message as search query safely
            return {
                "city": None,
                "country": None,
                "property_type": None,
                "max_budget_usd": None,
                "amenities_requested": [],
                "wants_live_prices": False,
                "distance_query": {"origin": None, "destination": None},
                "search_query": user_message,
            }

    async def geocode_location(self, location: str) -> GeocodingResult | None:
        """
        Converts a place name to coordinates using Nominatim (free OSM geocoding).
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{settings.nominatim_base_url}/search",
                    params={
                        "q": location,
                        "format": "json",
                        "limit": 1,
                        "addressdetails": 1,
                    },
                    headers={"User-Agent": "TravelAgentBot/1.0 (educational project)"},
                    timeout=10.0,
                )
                data = response.json()

            if not data:
                return None

            result = data[0]
            address = result.get("address", {})

            return GeocodingResult(
                display_name=result.get("display_name", location),
                latitude=float(result["lat"]),
                longitude=float(result["lon"]),
                city=address.get("city") or address.get("town") or address.get("village"),
                country=address.get("country"),
            )
        except Exception as e:
            print(f"[GeminiSearcher] Geocoding failed for '{location}': {e}")
            return None

    async def get_distance(
        self,
        origin_lat: float, origin_lon: float,
        dest_lat: float, dest_lon: float,
        origin_name: str, dest_name: str,
    ) -> DistanceResult | None:
        """
        Calculates driving distance and duration using OSRM (free routing API).
        """
        try:
            url = (
                f"{settings.osrm_base_url}/route/v1/driving/"
                f"{origin_lon},{origin_lat};{dest_lon},{dest_lat}"
            )
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    url,
                    params={"overview": "false", "alternatives": "false"},
                    timeout=10.0,
                )
                data = response.json()

            if data.get("code") != "Ok":
                return None

            route = data["routes"][0]
            distance_km = round(route["distance"] / 1000, 1)
            duration_min = round(route["duration"] / 60, 0)

            return DistanceResult(
                origin=origin_name,
                destination=dest_name,
                distance_km=distance_km,
                duration_minutes=duration_min,
            )
        except Exception as e:
            print(f"[GeminiSearcher] Distance calculation failed: {e}")
            return None

    def _metadata_to_property_record(self, metadata: dict) -> PropertyRecord:
        """Converts ChromaDB flat metadata dict back into a typed PropertyRecord."""
        import json as json_lib
        amenities = metadata.get("amenities", "[]")
        if isinstance(amenities, str):
            amenities = json_lib.loads(amenities)

        return PropertyRecord(
            property_id=metadata["property_id"],
            name=metadata["name"],
            property_type=PropertyType(metadata["property_type"]),
            city=metadata["city"],
            country=metadata["country"],
            latitude=float(metadata["latitude"]),
            longitude=float(metadata["longitude"]),
            source_url=metadata["source_url"],
            amenities=amenities,
            avg_price_usd=float(metadata.get("avg_price_usd", 0)) or None,
            star_rating=float(metadata.get("star_rating", 0)) or None,
            description=metadata.get("description", ""),
        )

    async def search(
        self,
        user_message: str,
        conversation_history: list,
    ) -> VectorSearchResult:
        """
        Full Agent 1 execution loop.
        """
        print(f"[GeminiSearcher] Processing: '{user_message}'")

        # Step 1 — Parse intent safely
        intent = await self.parse_query_intent(user_message, conversation_history)
        print(f"[GeminiSearcher] Intent: {intent}")

        # Step 2 — Search ChromaDB
        raw_results = database.search_properties(
            query=intent["search_query"],
            city_filter=intent.get("city"),
            property_type_filter=intent.get("property_type"),
            max_price_filter=intent.get("max_budget_usd"),
            n_results=5,
        )

        properties = [self._metadata_to_property_record(r) for r in raw_results]

        # Step 3 — Geocode location contexts
        location_context = None
        if intent.get("city"):
            query_location = intent["city"]
            if intent.get("country"):
                query_location += f", {intent['country']}"
            location_context = await self.geocode_location(query_location)

        # Step 4 — Formulate execution requirements
        requires_hydration = intent.get("wants_live_prices", False) or len(properties) == 0

        return VectorSearchResult(
            properties=properties,
            query_used=intent["search_query"],
            requires_live_hydration=requires_hydration,
            location_context=location_context,
        )