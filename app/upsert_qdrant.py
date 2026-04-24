#!/usr/bin/env python3
"""Upsert prepared point payloads into Qdrant from data/points_*.json files."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, TypeVar

import httpx
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

from client_embeddings import EMBEDDINGS_BASE_URL, embed_texts

_APP_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _APP_DIR.parent
load_dotenv(_ROOT_DIR / ".env")
load_dotenv()


def _required_env(name: str) -> str:
    v = (os.getenv(name) or "").strip()
    if not v:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return v


def _resolve_collection_name(collection_name: str, env_name: str) -> str:
    c = (collection_name or "").strip()
    if not c:
        raise RuntimeError("Collection name is empty. Set COLLECTION_NAME or pass --collection.")
    if env_name.strip().lower() == "dev" and not c.endswith("_dev"):
        return f"{c}_dev"
    return c


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read data/points_*.json, embed missing vectors from payload.embed_text, "
            "and upsert into Qdrant."
        )
    )
    parser.add_argument("--data-dir", default="data", help="Directory containing points_*.json")
    parser.add_argument("--pattern", default="points_*.json", help="Glob pattern (default: points_*.json)")
    parser.add_argument(
        "--collection",
        default=(os.getenv("COLLECTION_NAME") or "").strip(),
        help="Base collection name (default: COLLECTION_NAME env; ENV=dev adds _dev suffix).",
    )
    parser.add_argument(
        "--qdrant-url",
        default=(os.getenv("QDRANT_URL") or "").strip(),
        help="Qdrant URL (default: QDRANT_URL env).",
    )
    parser.add_argument(
        "--qdrant-api-key",
        default=(os.getenv("QDRANT_API_KEY") or "").strip(),
        help="Optional Qdrant API key (default: QDRANT_API_KEY env).",
    )
    parser.add_argument(
        "--env",
        default=(os.getenv("ENV") or "").strip(),
        help="Environment name (default: ENV env). If 'dev', collection becomes <name>_dev.",
    )
    parser.add_argument(
        "--embedding-base-url",
        default=(os.getenv("EMBEDDINGS_BASE_URL") or "").strip() or EMBEDDINGS_BASE_URL,
        help="Embeddings service root without /v1 (default: EMBEDDINGS_BASE_URL env/module).",
    )
    parser.add_argument(
        "--embedding-model",
        default=(os.getenv("EMBEDDING_MODEL") or "").strip() or "BAAI/bge-m3",
        help="Embedding model id (default: EMBEDDING_MODEL env or BAAI/bge-m3).",
    )
    parser.add_argument(
        "--embedding-api-key",
        default=(os.getenv("EMBEDDING_API_KEY") or "").strip(),
        help="Optional Bearer token for embeddings API.",
    )
    parser.add_argument(
        "--embedding-internal-key",
        default=(os.getenv("EMBEDDING_INTERNAL_KEY") or "").strip(),
        help="Optional X-Internal-Key for embeddings API.",
    )
    parser.add_argument(
        "--vector-size",
        type=int,
        default=int((os.getenv("VECTOR_SIZE") or "1024").strip()),
        help="Embedding vector size (default: VECTOR_SIZE env or 1024).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int((os.getenv("BATCH_SIZE") or "20").strip()),
        help="Upsert batch size (default: BATCH_SIZE env or 20).",
    )
    parser.add_argument(
        "--distance",
        default="COSINE",
        choices=["COSINE", "DOT", "EUCLID", "MANHATTAN"],
        help="Distance metric when creating a new collection.",
    )
    parser.add_argument(
        "--skip-create-collection",
        action="store_true",
        help="Do not create collection if missing.",
    )
    parser.add_argument(
        "--skip-indexes",
        action="store_true",
        help="Do not create payload indexes for metadata/filter keys.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse/embed but do not write to Qdrant.",
    )
    parser.add_argument(
        "--skip-embedding",
        action="store_true",
        help="Do not call embeddings API; requires vectors already present for real upsert.",
    )
    return parser.parse_args()


def _load_points(path: Path) -> list[dict[str, Any]]:
    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON array")
    out: list[dict[str, Any]] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"{path} item[{i}] is not an object")
        out.append(item)
    return out


T = TypeVar("T")


def _iter_batches(items: list[T], size: int) -> list[list[T]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _distance_from_name(name: str) -> models.Distance:
    return getattr(models.Distance, name)


def ensure_collection(
    client: QdrantClient,
    collection: str,
    vector_size: int,
    distance_name: str,
    skip_create: bool,
) -> None:
    exists = client.collection_exists(collection_name=collection)
    if exists:
        return
    if skip_create:
        raise RuntimeError(f"Collection {collection!r} does not exist and --skip-create-collection is set")
    client.create_collection(
        collection_name=collection,
        vectors_config=models.VectorParams(size=vector_size, distance=_distance_from_name(distance_name)),
    )


def ensure_indexes(client: QdrantClient, collection: str) -> None:
    # Matches metadata/filter guidance in docs/schema.md and docs/plan.md.
    index_specs: list[tuple[str, models.PayloadSchemaType]] = [
        ("source", models.PayloadSchemaType.KEYWORD),
        ("doc_type", models.PayloadSchemaType.KEYWORD),
        ("section", models.PayloadSchemaType.KEYWORD),
        ("content_hash", models.PayloadSchemaType.KEYWORD),
        ("chunk_id_parent", models.PayloadSchemaType.KEYWORD),
        ("was_split", models.PayloadSchemaType.BOOL),
        ("token_count", models.PayloadSchemaType.INTEGER),
        ("embed_token_count", models.PayloadSchemaType.INTEGER),
        ("split_index", models.PayloadSchemaType.INTEGER),
        ("ingest_ts", models.PayloadSchemaType.DATETIME),
    ]
    for field_name, field_schema in index_specs:
        try:
            client.create_payload_index(
                collection_name=collection,
                field_name=field_name,
                field_schema=field_schema,
            )
        except Exception:
            # Index may already exist or field may be sparse. Safe to continue.
            continue


def _build_headers(internal_key: str) -> dict[str, str]:
    headers: dict[str, str] = {}
    if internal_key:
        headers["X-Internal-Key"] = internal_key
    return headers


def _ensure_vectors(
    points: list[dict[str, Any]],
    *,
    model: str,
    base_url: str,
    api_key: str,
    internal_key: str,
) -> None:
    missing_idx: list[int] = []
    texts: list[str] = []
    for i, point in enumerate(points):
        vector = point.get("vector")
        if isinstance(vector, list) and len(vector) > 0:
            continue
        payload = point.get("payload")
        if not isinstance(payload, dict):
            raise ValueError(f"Point at index {i} missing payload")
        embed_text = str(payload.get("embed_text") or "").strip()
        if not embed_text:
            raise ValueError(f"Point at index {i} missing payload.embed_text")
        missing_idx.append(i)
        texts.append(embed_text)

    if not missing_idx:
        return

    try:
        with httpx.Client(timeout=120.0) as http_client:
            vectors = embed_texts(
                base_url=base_url,
                model=model,
                texts=texts,
                api_key=api_key or None,
                client=http_client,
                extra_headers=_build_headers(internal_key),
                timeout=120.0,
            )
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(
            "Embedding request failed. Check EMBEDDINGS_BASE_URL and auth settings "
            "(--embedding-api-key / --embedding-internal-key or .env values). "
            f"HTTP {exc.response.status_code}: {exc.response.text}"
        ) from exc

    for i, vec in zip(missing_idx, vectors):
        points[i]["vector"] = vec


def _to_point_struct(point: dict[str, Any]) -> models.PointStruct:
    pid = point.get("id")
    vec = point.get("vector")
    payload = point.get("payload")
    if not isinstance(pid, (str, int)):
        raise ValueError(f"Invalid point id: {pid!r}")
    if not isinstance(vec, list) or not vec:
        raise ValueError(f"Point {pid!r} has empty vector")
    if not isinstance(payload, dict):
        raise ValueError(f"Point {pid!r} has invalid payload")
    return models.PointStruct(id=pid, vector=vec, payload=payload)


def _count_missing_vectors(points: list[dict[str, Any]]) -> int:
    missing = 0
    for point in points:
        vector = point.get("vector")
        if not isinstance(vector, list) or len(vector) == 0:
            missing += 1
    return missing


def main() -> None:
    started = time.perf_counter()
    args = parse_args()
    data_dir = Path(args.data_dir)
    files = sorted(data_dir.glob(args.pattern))
    if not files:
        raise SystemExit(f"No files matched {args.pattern!r} under {data_dir}")

    collection_base = args.collection or _required_env("COLLECTION_NAME")
    collection = _resolve_collection_name(collection_base, args.env)
    qdrant_url = args.qdrant_url or _required_env("QDRANT_URL")

    all_points: list[dict[str, Any]] = []
    for fp in files:
        all_points.extend(_load_points(fp))

    if not args.skip_embedding:
        _ensure_vectors(
            all_points,
            model=args.embedding_model,
            base_url=args.embedding_base_url,
            api_key=args.embedding_api_key,
            internal_key=args.embedding_internal_key,
        )

    for point in all_points:
        vector = point.get("vector")
        if args.skip_embedding and args.dry_run and (not isinstance(vector, list) or len(vector) == 0):
            continue
        if not isinstance(vector, list) or len(vector) != args.vector_size:
            pid = point.get("id")
            raise ValueError(f"Point {pid!r} vector size mismatch: expected {args.vector_size}, got {len(vector) if isinstance(vector, list) else 'invalid'}")

    if args.dry_run:
        missing_vectors = _count_missing_vectors(all_points)
        print(
            f"Dry run OK: prepared {len(all_points)} points from {len(files)} files "
            f"(missing_vectors={missing_vectors}, skip_embedding={args.skip_embedding})"
        )
        return

    if args.skip_embedding:
        missing_vectors = _count_missing_vectors(all_points)
        if missing_vectors:
            raise RuntimeError(
                f"--skip-embedding set but {missing_vectors} points have empty vectors; "
                "cannot upsert empty vectors."
            )

    client = QdrantClient(url=qdrant_url, api_key=args.qdrant_api_key or None)
    ensure_collection(
        client=client,
        collection=collection,
        vector_size=args.vector_size,
        distance_name=args.distance,
        skip_create=args.skip_create_collection,
    )
    if not args.skip_indexes:
        ensure_indexes(client, collection)

    batches = _iter_batches(all_points, args.batch_size)
    upserted = 0
    for batch in batches:
        point_structs = [_to_point_struct(p) for p in batch]
        client.upsert(collection_name=collection, points=point_structs, wait=True)
        upserted += len(point_structs)
        print(f"Upserted {upserted}/{len(all_points)}")

    print(f"Done: upserted {upserted} point(s) into collection {collection!r}")
    elapsed_ms = (time.perf_counter() - started) * 1000
    print(f"upsert_qdrant total_latency_ms={elapsed_ms:.1f}")


if __name__ == "__main__":
    main()

