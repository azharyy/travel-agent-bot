# backend/core/database.py
import json
from typing import Optional
import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer

from backend.core.config import settings
from backend.core.schemas import PropertyRecord

# ── Singleton clients ──
# Both are expensive to initialize. We create them once and reuse.
_chroma_client: Optional[chromadb.PersistentClient] = None
_property_chroma_client: Optional[chromadb.PersistentClient] = None
_embedding_model: Optional[SentenceTransformer] = None


def get_chroma_client() -> chromadb.PersistentClient:
    """
    Returns a persistent ChromaDB client.
    Data survives restarts because it writes to disk at CHROMA_DB_PATH.
    """
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(
            path=settings.chroma_db_path,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
    return _chroma_client


def get_property_chroma_client() -> chromadb.PersistentClient:
    """
    Returns the legacy ChromaDB client used by travel_properties.
    This keeps old all-MiniLM embeddings separate from egypt_hotels CLIP vectors.
    """
    global _property_chroma_client
    if _property_chroma_client is None:
        _property_chroma_client = chromadb.PersistentClient(
            path=settings.chroma_persist_directory,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
    return _property_chroma_client


def get_hotel_collection_name() -> str:
    """Return the Chroma collection name used for hotel cards."""
    return settings.hotel_collection_name or "egypt_hotels"


def get_embedding_model() -> SentenceTransformer:
    """
    Loads all-MiniLM-L6-v2 locally.
    First call downloads ~80MB model to ~/.cache/huggingface.
    Every subsequent call loads from cache instantly.
    """
    global _embedding_model
    if _embedding_model is None:
        print(f"[Database] Loading embedding model: {settings.embedding_model}")
        _embedding_model = SentenceTransformer(settings.embedding_model)
        print("[Database] Embedding model ready.")
    return _embedding_model


def get_collection() -> chromadb.Collection:
    """
    Gets or creates the main property collection.
    ChromaDB stores documents + embeddings + metadata together.
    We use cosine similarity — best for semantic text search.
    """
    client = get_property_chroma_client()
    collection = client.get_or_create_collection(
        name=settings.chroma_collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def add_property(record: PropertyRecord) -> None:
    """
    Embeds a property description and stores it in ChromaDB.

    WHY embed only the description:
    The description is natural language — rich semantic content.
    Structured fields (price, city, type) go into metadata for
    exact filtering BEFORE the vector search runs. This is the
    correct hybrid search pattern.
    """
    model = get_embedding_model()
    collection = get_collection()

    embedding = model.encode(record.description).tolist()

    # Safe fallback conversions for metadata properties
    avg_price = float(record.avg_price_usd) if record.avg_price_usd is not None else 0.0
    rating = float(record.star_rating) if record.star_rating is not None else 0.0

    collection.upsert(
        ids=[record.property_id],
        embeddings=[embedding],
        documents=[record.description],
        metadatas=[{
            "property_id": str(record.property_id),
            "name": str(record.name),
            "property_type": str(record.property_type.value),
            "city": str(record.city).lower().strip(),
            "country": str(record.country).lower().strip(),
            "latitude": float(record.latitude),
            "longitude": float(record.longitude),
            "source_url": str(record.source_url),
            "amenities": json.dumps(record.amenities),
            "avg_price_usd": avg_price,
            "star_rating": rating,
        }],
    )


def search_properties(
    query: str,
    city_filter: Optional[str] = None,
    property_type_filter: Optional[str] = None,
    max_price_filter: Optional[float] = None,
    n_results: int = 5,
) -> list[dict]:
    """
    Semantic vector search with optional pre-filters.

    HOW IT WORKS:
    1. Embed the user query into a vector using the same model used at index time.
    2. ChromaDB finds the n_results most semantically similar property descriptions.
    3. Optional where filters narrow results BEFORE vector search (faster + cheaper).
    """
    model = get_embedding_model()
    collection = get_collection()

    # Defensively handle empty or missing search queries
    if not query or not query.strip():
        query = "comfortable stay holiday location"

    query_embedding = model.encode(query).tolist()

    # Build metadata filter mapping dynamically
    where_filter = {}
    and_conditions = []

    if city_filter and city_filter.strip():
        and_conditions.append({"city": {"$eq": city_filter.lower().strip()}})
    
    if property_type_filter and property_type_filter.strip():
        and_conditions.append({"property_type": {"$eq": property_type_filter.lower().strip()}})

    # If multiple conditions exist, wrap them in a valid Chroma logical dictionary structure
    if len(and_conditions) > 1:
        where_filter = {"$and": and_conditions}
    elif len(and_conditions) == 1:
        where_filter = and_conditions[0]

    search_kwargs = {
        "query_embeddings": [query_embedding],
        "n_results": int(n_results),
        "include": ["documents", "metadatas", "distances"],
    }

    if where_filter:
        search_kwargs["where"] = where_filter

    try:
        results = collection.query(**search_kwargs)
    except Exception as e:
        print(f"[Database] ChromaDB query tracking exception caught: {e}")
        return []

    # Flatten ChromaDB's nested result format into clean dicts matching schema needs
    output = []
    if results and results.get("metadatas") and len(results["metadatas"]) > 0:
        for i, metadata in enumerate(results["metadatas"][0]):
            if metadata is None:
                continue
            
            # Safely rebuild nested keys from documents collection matrix
            metadata["description"] = results["documents"][0][i] if (results.get("documents") and len(results["documents"]) > 0) else ""
            
            # Reconstruct dynamic similarity metrics securely
            if results.get("distances") and len(results["distances"]) > 0:
                metadata["similarity_score"] = round(1 - results["distances"][0][i], 4)
            else:
                metadata["similarity_score"] = 1.0
                
            output.append(metadata)

    return output


def get_collection_count() -> int:
    """Returns how many properties are stored. Useful for health checks."""
    try:
        return get_collection().count()
    except Exception:
        return 0


def seed_sample_data() -> None:
    """
    Inserts sample properties so the system works immediately without scraping.
    Run this once to populate the database for testing.
    """
    from backend.core.schemas import PropertyType

    samples = [
        PropertyRecord(
            property_id="prop_001",
            name="Nile View Luxury Hotel",
            property_type=PropertyType.HOTEL,
            city="Cairo",
            country="Egypt",
            latitude=30.0444,
            longitude=31.2357,
            source_url="https://www.booking.com/hotel/eg/nile-view-luxury.html",
            amenities=["pool", "wifi", "breakfast", "gym", "river view", "spa"],
            avg_price_usd=120.0,
            star_rating=5.0,
            description=(
                "A luxurious 5-star hotel on the banks of the Nile River in Cairo. "
                "Features stunning river views, rooftop pool, full spa, and complimentary "
                "breakfast. Walking distance to the Egyptian Museum and Khan el-Khalili bazaar. "
                "Perfect for couples and business travelers seeking premium comfort in central Cairo."
            ),
        ),
        PropertyRecord(
            property_id="prop_002",
            name="Old Town Cairo Apartment",
            property_type=PropertyType.APARTMENT,
            city="Cairo",
            country="Egypt",
            latitude=30.0459,
            longitude=31.2624,
            source_url="https://www.airbnb.com/rooms/12345678",
            amenities=["wifi", "kitchen", "washing machine", "air conditioning"],
            avg_price_usd=45.0,
            star_rating=4.2,
            description=(
                "A charming 2-bedroom apartment in the heart of Islamic Cairo. "
                "Fully equipped kitchen, fast wifi, and authentic local feel. "
                "Steps from Al-Azhar mosque and the historic bazaars. "
                "Ideal for budget-conscious travelers who want an authentic Cairo experience."
            ),
        ),
        PropertyRecord(
            property_id="prop_003",
            name="Sahara Desert Chalet",
            property_type=PropertyType.CHALET,
            city="Siwa",
            country="Egypt",
            latitude=29.2032,
            longitude=25.5194,
            source_url="https://www.booking.com/hotel/eg/sahara-desert-chalet.html",
            amenities=["wifi", "desert view", "private pool", "all-inclusive", "camel tours"],
            avg_price_usd=200.0,
            star_rating=4.8,
            description=(
                "An exclusive eco-chalet on the edge of the Sahara Desert in Siwa Oasis. "
                "Private plunge pool, stunning sand dune views, and all-inclusive meals featuring "
                "local Siwan cuisine. Offers guided camel tours, sandboarding, and stargazing. "
                "Perfect for adventurous travelers seeking a unique luxury desert escape."
            ),
        ),
        PropertyRecord(
            property_id="prop_004",
            name="Red Sea Beach Villa",
            property_type=PropertyType.VILLA,
            city="Hurghada",
            country="Egypt",
            latitude=27.2579,
            longitude=33.8116,
            source_url="https://www.airbnb.com/rooms/87654321",
            amenities=["private beach", "pool", "wifi", "snorkeling gear", "bbq", "sea view"],
            avg_price_usd=350.0,
            star_rating=4.9,
            description=(
                "A stunning beachfront villa on the Red Sea coast in Hurghada. "
                "Private beach access, infinity pool, and world-class snorkeling right off the shore. "
                "Sleeps up to 8 guests. Fully staffed with a private chef and housekeeper. "
                "Ideal for families or groups wanting a premium Red Sea holiday experience."
            ),
        ),
        PropertyRecord(
            property_id="prop_005",
            name="Luxor Heritage Guesthouse",
            property_type=PropertyType.HOTEL,
            city="Luxor",
            country="Egypt",
            latitude=25.6872,
            longitude=32.6396,
            source_url="https://www.booking.com/hotel/eg/luxor-heritage.html",
            amenities=["wifi", "rooftop terrace", "breakfast", "nile view", "temple tours"],
            avg_price_usd=65.0,
            star_rating=4.5,
            description=(
                "A beautifully restored heritage guesthouse overlooking the Nile in Luxor. "
                "Rooftop terrace with direct views of the Luxor Temple. Complimentary breakfast "
                "and guided temple tours available. Walking distance to Karnak Temple and the "
                "Valley of the Kings ferry. Perfect for history lovers and culture travelers."
            ),
        ),
        PropertyRecord(
            property_id="prop_006",
            name="Sharm El Sheikh Resort Hotel",
            property_type=PropertyType.HOTEL,
            city="Sharm El Sheikh",
            country="Egypt",
            latitude=27.9158,
            longitude=34.3299,
            source_url="https://www.booking.com/hotel/eg/sharm-resort.html",
            amenities=["pool", "wifi", "all-inclusive", "diving center", "kids club", "beach"],
            avg_price_usd=180.0,
            star_rating=5.0,
            description=(
                "A world-class all-inclusive 5-star resort in Sharm El Sheikh on the Red Sea. "
                "Multiple pools, private beach, PADI certified diving center, and a full kids club. "
                "All meals and drinks included. Renowned for incredible coral reef snorkeling. "
                "Ideal for families, divers, and couples seeking a complete beach resort experience."
            ),
        ),
    ]

    print(f"[Database] Seeding {len(samples)} sample properties...")
    for record in samples:
        add_property(record)
    print(f"[Database] Done. Collection now has {get_collection_count()} properties.")
