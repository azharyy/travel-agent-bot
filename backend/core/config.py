from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Application
    app_name: str = "Travel Agent Bot"
    app_env: str = "development"
    debug: bool = True

    @field_validator("debug", mode="before")
    @classmethod
    def parse_debug(cls, value):
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"release", "production", "prod", "false", "0", "off", "no"}:
                return False
            if normalized in {"debug", "development", "dev", "true", "1", "on", "yes"}:
                return True
        return value

    # FastAPI
    api_host: str = "127.0.0.1"
    api_port: int = 8000

    # Gemini — free tier: 15 rpm, 1M tokens/day
    gemini_api_key: str
    gemini_model: str = "gemini-2.5-flash"

    # RollingGo MCP server. Default is the optional local adapter on its own port.
    rollinggo_mcp_url: str = "http://127.0.0.1:8010/mcp"
    rollinggo_api_key: str = ""
    rollinggo_accept_language: str = "en_US"
    rollinggo_local_mcp_host: str = "127.0.0.1"
    rollinggo_local_mcp_port: int = 8010
    rollinggo_api_base_url: str = ""
    rollinggo_search_hotels_path: str = "/searchHotels"
    rollinggo_hotel_detail_path: str = "/getHotelDetail"
    rollinggo_search_tags_path: str = "/getHotelSearchTags"

    # Hotel catalog build defaults
    hotel_country_code: str = "EG"
    hotel_currency: str = "EGP"
    hotel_db_build_checkin: str = "2026-08-01"
    hotel_db_build_nights: int = 2
    hotel_db_build_adults: int = 2
    hotel_db_build_room_count: int = 1
    hotel_db_top_n: int = 20
    hotel_display_top_n: int = 5

    # Hotel retrieval and image embedding defaults
    enable_clip_image_embeddings: bool = True
    clip_image_weight: float = 0.20
    chroma_db_path: str = "backend/chroma_db"
    hotel_collection_name: str = "egypt_hotels"

    # ChromaDB — fully local
    chroma_persist_directory: str = "./data/chromadb"
    chroma_collection_name: str = "travel_properties"

    # Sentence Transformers — fully local embedding model
    embedding_model: str = "all-MiniLM-L6-v2"

    # Ollama — fully local
    ollama_base_url: str = "http://localhost:11434"
    ollama_cleaner_model: str = "phi3:mini"
    ollama_reviewer_model: str = "phi3:mini"

    # Free geocoding & routing — no API key needed
    nominatim_base_url: str = "https://nominatim.openstreetmap.org"
    osrm_base_url: str = "https://router.project-osrm.org"

    # Streamlit -> FastAPI
    api_base_url: str = "http://127.0.0.1:8000"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
