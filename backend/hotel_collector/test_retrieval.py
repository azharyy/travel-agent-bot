"""Manual retrieval smoke test for the egypt_hotels Chroma collection."""

from __future__ import annotations

from typing import Any

import numpy as np

from backend.hotel_collector.ingest_chroma import CLIP_MODEL_NAME


QUERY = "luxury beach resort in Hurghada"


def _normalize_vector(vector: Any) -> list[float]:
    array = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(array))
    if norm != 0.0:
        array = array / norm
    return array.tolist()


def _print_empty_instructions() -> None:
    print("No hotel vectors found in Chroma.")
    print("Run these commands first:")
    print("python -m backend.hotel_collector.build_hotels_db")
    print("python -m backend.hotel_collector.ingest_chroma --reset-collection --skip-images")


def main() -> int:
    from backend.core.database import get_chroma_client, get_hotel_collection_name

    client = get_chroma_client()
    collection_name = get_hotel_collection_name()
    try:
        collection = client.get_collection(collection_name)
    except Exception:
        _print_empty_instructions()
        return 0

    if collection.count() == 0:
        _print_empty_instructions()
        return 0

    print(f"Loading CLIP embedding model: {CLIP_MODEL_NAME}")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(CLIP_MODEL_NAME)
    query_embedding = _normalize_vector(model.encode(QUERY, convert_to_numpy=True))

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5,
        include=["documents", "metadatas", "distances"],
    )

    metadatas = results.get("metadatas") or []
    if not metadatas or not metadatas[0]:
        _print_empty_instructions()
        return 0

    print(f"Top results for: {QUERY}")
    for index, metadata in enumerate(metadatas[0], start=1):
        metadata = metadata or {}
        print(f"\n{index}. {metadata.get('name', '<missing>')}")
        print(f"   city: {metadata.get('city', '<missing>')}")
        print(f"   hotel_type: {metadata.get('hotel_type', '<missing>')}")
        print(f"   image_url: {metadata.get('image_url', '')}")
        print(f"   summary: {metadata.get('summary', '')}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
