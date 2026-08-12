"""Ingest static hotel cards into the egypt_hotels Chroma collection."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import requests
from PIL import Image

from backend.hotel_collector.models import HotelCard
from backend.hotel_collector.utils import flatten_chroma_metadata, slugify, summarize_hotel_for_rag


CLIP_MODEL_NAME = "sentence-transformers/clip-ViT-B-32"
HOTELS_PATH = Path("backend/data/hotels.json")
IMAGE_CACHE_DIR = Path("backend/data/hotel_images")
MAX_IMAGE_BYTES = 15 * 1024 * 1024


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest static hotel cards into Chroma.")
    parser.add_argument("--reset-collection", action="store_true", help="Delete and recreate collection.")
    parser.add_argument("--skip-images", action="store_true", help="Use text embeddings only.")
    return parser.parse_args()


def _load_hotels() -> list[HotelCard]:
    if not HOTELS_PATH.exists():
        print(f"No hotel database found at {HOTELS_PATH}.")
        print('Run: python -m backend.hotel_collector.build_hotels_db')
        return []

    data = json.loads(HOTELS_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{HOTELS_PATH} must contain a JSON list.")

    hotels: list[HotelCard] = []
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            print(f"Skipping hotel row {index}: expected object.")
            continue
        try:
            hotels.append(HotelCard.model_validate(item))
        except Exception as exc:
            print(f"Skipping hotel row {index}: {exc}")
    return hotels


def _normalize_vector(vector: Any) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(array))
    if norm == 0.0:
        return array
    return array / norm


def _encode_text(model: Any, text: str) -> np.ndarray:
    return _normalize_vector(model.encode(text, convert_to_numpy=True))


def _encode_image(model: Any, image: Image.Image) -> np.ndarray:
    return _normalize_vector(model.encode(image, convert_to_numpy=True))


def _fuse_embeddings(text_vector: np.ndarray, image_vector: np.ndarray, image_weight: float) -> np.ndarray:
    safe_weight = min(max(float(image_weight), 0.0), 1.0)
    fused = ((1.0 - safe_weight) * text_vector) + (safe_weight * image_vector)
    return _normalize_vector(fused)


def _image_cache_path(hotel: HotelCard) -> Path:
    parsed = urlparse(hotel.image_url or "")
    suffix = Path(parsed.path).suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
        suffix = ".jpg"
    url_hash = hashlib.sha1((hotel.image_url or hotel.id).encode("utf-8")).hexdigest()[:12]
    return IMAGE_CACHE_DIR / f"{slugify(hotel.id)}-{url_hash}{suffix}"


def _download_image(url: str, target_path: Path) -> None:
    response = requests.get(url, timeout=20, headers={"User-Agent": "GuideMeHotelIngest/1.0"})
    response.raise_for_status()

    content_length = response.headers.get("Content-Length")
    if content_length and int(content_length) > MAX_IMAGE_BYTES:
        raise ValueError("image is larger than the cache limit")
    if len(response.content) > MAX_IMAGE_BYTES:
        raise ValueError("image is larger than the cache limit")

    target_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, dir=target_path.parent, suffix=target_path.suffix) as tmp:
        tmp.write(response.content)
        tmp_path = Path(tmp.name)
    tmp_path.replace(target_path)


def _load_cached_or_downloaded_image(hotel: HotelCard) -> Image.Image | None:
    if not hotel.image_url:
        return None

    cache_path = _image_cache_path(hotel)
    try:
        if not cache_path.exists():
            _download_image(hotel.image_url, cache_path)
        with Image.open(cache_path) as image:
            return image.convert("RGB")
    except Exception as exc:
        print(f"Warning: image embedding failed for {hotel.name}: {exc}")
        return None


def _embedding_for_hotel(
    model: Any,
    hotel: HotelCard,
    document: str,
    use_images: bool,
    image_weight: float,
) -> list[float]:
    text_vector = _encode_text(model, document)
    if not use_images or not hotel.image_url:
        return text_vector.tolist()

    image = _load_cached_or_downloaded_image(hotel)
    if image is None:
        return text_vector.tolist()

    try:
        image_vector = _encode_image(model, image)
        return _fuse_embeddings(text_vector, image_vector, image_weight).tolist()
    except Exception as exc:
        print(f"Warning: image embedding failed for {hotel.name}: {exc}")
        return text_vector.tolist()


def _get_collection(reset_collection: bool):
    from backend.core.database import get_chroma_client, get_hotel_collection_name

    client = get_chroma_client()
    collection_name = get_hotel_collection_name()
    if reset_collection:
        try:
            client.delete_collection(collection_name)
            print(f"Deleted Chroma collection: {collection_name}")
        except Exception:
            pass

    return client.get_or_create_collection(
        name=collection_name,
        metadata={
            "hnsw:space": "cosine",
            "embedding_model": CLIP_MODEL_NAME,
            "source": "rollinggo_static_hotels",
        },
    )


def _hotel_metadata(hotel: HotelCard) -> dict[str, str | int | float | bool]:
    return flatten_chroma_metadata(hotel.model_dump())


def _load_settings():
    from backend.core.config import settings

    return settings


def ingest(reset_collection: bool = False, skip_images: bool = False) -> int:
    hotels = _load_hotels()
    if not hotels:
        print("No static hotel cards to ingest.")
        return 0

    print(f"Loading CLIP embedding model: {CLIP_MODEL_NAME}")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(CLIP_MODEL_NAME)
    collection = _get_collection(reset_collection)
    app_settings = _load_settings()
    use_images = bool(app_settings.enable_clip_image_embeddings and not skip_images)
    image_weight = app_settings.clip_image_weight

    ids: list[str] = []
    documents: list[str] = []
    embeddings: list[list[float]] = []
    metadatas: list[dict[str, str | int | float | bool]] = []

    for hotel in hotels:
        document = summarize_hotel_for_rag(hotel)
        ids.append(hotel.id)
        documents.append(document)
        metadatas.append(_hotel_metadata(hotel))
        embeddings.append(
            _embedding_for_hotel(
                model,
                hotel,
                document,
                use_images=use_images,
                image_weight=image_weight,
            )
        )

    try:
        collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
    except Exception as exc:
        message = str(exc)
        if "dimension" in message.lower():
            print(
                "Chroma rejected the embeddings, likely because this collection already "
                "contains vectors from a different model."
            )
            print("Run: python -m backend.hotel_collector.ingest_chroma --reset-collection")
        raise

    print(f"Upserted {len(ids)} hotel cards into collection '{collection.name}'.")
    print(f"Collection count: {collection.count()}")
    return 0


def main() -> int:
    args = _parse_args()
    return ingest(reset_collection=args.reset_collection, skip_images=args.skip_images)


if __name__ == "__main__":
    raise SystemExit(main())
