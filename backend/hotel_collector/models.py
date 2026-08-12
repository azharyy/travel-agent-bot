"""Pydantic models for static hotel cards and live availability snapshots."""

from pydantic import BaseModel, Field


class HotelCard(BaseModel):
    """Static hotel profile safe to store in the local catalog and Chroma."""

    id: str
    source: str = "rollinggo"
    hotel_id: str
    name: str
    city: str
    hotel_type: str = "hotel"
    summary: str | None = None
    amenities: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    image_url: str | None = None
    booking_url: str | None = None


class LiveRoomOption(BaseModel):
    """A live room/rate option returned by the external hotel MCP provider."""

    room_id: str | None = None
    room_name: str | None = None
    board_type: str | None = None
    occupancy: str | None = None
    nightly_price: float | None = None
    total_price: float | None = None
    currency: str = "EGP"
    refundable: bool | None = None
    cancellation_policy: str | None = None
    taxes_and_fees: float | None = None
    booking_url: str | None = None


class LiveHotelAvailability(BaseModel):
    """Live availability and room options for a hotel/date search."""

    hotel_id: str
    provider: str = "rollinggo"
    provider_hotel_id: str | None = None
    checkin: str
    nights: int
    adults: int
    room_count: int = 1
    currency: str = "EGP"
    is_available: bool | None = None
    rooms: list[LiveRoomOption] = Field(default_factory=list)
    fetched_at: str | None = None
    availability_notes: str | None = None
