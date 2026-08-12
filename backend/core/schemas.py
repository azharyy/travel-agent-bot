from pydantic import BaseModel, Field, field_validator
from typing import Any, Optional
from enum import Enum


class PropertyType(str, Enum):
    HOTEL = "hotel"
    APARTMENT = "apartment"
    CHALET = "chalet"
    VILLA = "villa"


class PipelineStage(str, Enum):
    VECTOR_SEARCH = "vector_search"
    LIVE_HYDRATION = "live_hydration"
    CLEANING = "cleaning"
    REVIEW = "review"
    COMPLETE = "complete"


# ── Frontend → FastAPI ──

class ChatRequest(BaseModel):
    session_id: str
    user_message: str = Field(..., min_length=1, max_length=2000)
    conversation_history: list[dict] = Field(default_factory=list)

    @field_validator("user_message")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        return v.strip()


# ── ChromaDB property record ──

class PropertyRecord(BaseModel):
    property_id: str
    name: str
    property_type: PropertyType
    city: str
    country: str
    latitude: float
    longitude: float
    source_url: str
    amenities: list[str]
    avg_price_usd: Optional[float] = None
    star_rating: Optional[float] = None
    description: str  # This string gets embedded into ChromaDB


# ── Geocoding result from Nominatim ──

class GeocodingResult(BaseModel):
    display_name: str
    latitude: float
    longitude: float
    city: Optional[str] = None
    country: Optional[str] = None


# ── Distance result from OSRM ──

class DistanceResult(BaseModel):
    origin: str
    destination: str
    distance_km: float
    duration_minutes: float


# ── Agent 1 output ──

class VectorSearchResult(BaseModel):
    properties: list[PropertyRecord]
    query_used: str
    requires_live_hydration: bool
    location_context: Optional[GeocodingResult] = None


# ── Scraper input ──

class HydrationRequest(BaseModel):
    property_id: str
    source_url: str
    check_in: Optional[str] = None
    check_out: Optional[str] = None
    guests: int = 2


# ── Scraper raw output → Agent 2 input ──

class RawHydrationData(BaseModel):
    property_id: str
    source_url: str
    compressed_html: str
    scrape_timestamp: str


# ── Agent 2 output ──

class LivePriceData(BaseModel):
    property_id: str
    nightly_price_usd: Optional[float] = None
    total_price_usd: Optional[float] = None
    currency_original: Optional[str] = None
    cleaning_fee_usd: Optional[float] = None
    is_available: Optional[bool] = None
    availability_notes: Optional[str] = None
    scrape_timestamp: str


# ── Agent 3 final verified output ──

class VerifiedProperty(BaseModel):
    property_id: str
    name: str
    property_type: PropertyType
    city: str
    country: str
    latitude: float
    longitude: float
    amenities: list[str]
    star_rating: Optional[float] = None
    nightly_price_usd: Optional[float] = None
    total_price_usd: Optional[float] = None
    is_available: Optional[bool] = None
    source_url: str
    reviewer_notes: str
    passed_review: bool
    distance_info: Optional[DistanceResult] = None


# ── FastAPI → Streamlit ──

class ChatResponse(BaseModel):
    session_id: str
    assistant_message: str
    properties: list[dict[str, Any]] = Field(default_factory=list)
    live_availability: Optional[dict[str, Any]] = None
    pipeline_stage_reached: str
    error: Optional[str] = None
    
    
    
# backend/core/schemas.py
from pydantic import BaseModel, Field
from typing import Optional, List

class HydrationRequest(BaseModel):
    property_id: str
    source_url: str

class RawHydrationData(BaseModel):
    property_id: str
    source_url: str
    compressed_html: str
    scrape_timestamp: str

class LivePriceData(BaseModel):
    property_id: str
    nightly_price_usd: Optional[float] = None
    total_price_usd: Optional[float] = None
    currency_original: Optional[str] = None
    cleaning_fee_usd: Optional[float] = None
    is_available: Optional[bool] = True
    availability_notes: Optional[str] = None
    scrape_timestamp: str

class AgentReviewRequest(BaseModel):
    property_id: str
    static_db_data: dict
    live_scraped_data: LivePriceData
    user_preferences: dict

class ValidationResult(BaseModel):
    property_id: str
    is_valid_match: bool
    confidence_score: float
    reasoning_summary: str
    actionable_alerts: List[str] = Field(default_factory=list)
