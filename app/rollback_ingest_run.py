#!/usr/bin/env python3
"""Rollback lifecycle state to a target ingest run manifest."""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

logger = logging.getLogger(__name__)
_EST = timezone(timedelta(hours=-5), name="EST")
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
    env = env_name.strip().lower()
    if env in {"dev", "qa", "prod"}:
        suffix = f"_{env}"
        if not c.endswith(suffix):
            return f"{c}{suffix}"
    return c


def _default_rollback_run_id() -> str:
    return datetime.now(_EST).strftime("rollback_%Y%m%d_%H%M%S_EST")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rollback active lifecycle state to a target ingest run.")
    p.add_argument("--target-run-id", required=True, help="Target ingest run id to restore.")
    p.add_argument("--manifest-path", default="", help="Optional explicit manifest path.")
    p.add_argument("--manifest-dir", default="data/processed", help="Manifest directory (default: data/processed).")
    p.add_argument("--collection", default=(os.getenv("COLLECTION_NAME") or "").strip())
    p.add_argument("--env", default=(os.getenv("ENV") or "").strip())
    p.add_argument("--qdrant-url", default=(os.getenv("QDRANT_URL") or "").strip())
    p.add_argument("--qdrant-api-key", default=(os.getenv("QDRANT_API_KEY") or "").strip())
    p.add_argument("--scope-key", choices=["source", "doc_type", "collection"], default="collection")
    p.add_argument("--scope-value", default="")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--apply", action="store_true", help="Apply rollback actions (default preview only).")
    p.add_argument("--run-id", default="", help="Optional fixed rollback run id.")
    p.add_argument("--output-dir", default="", help="Output directory for rollback action report.")
    return p.parse_args()


def _load_manifest(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Manifest must be JSON object.")
    return data


def _manifest_path(args: argparse.Namespace) -> Path:
    if args.manifest_path.strip():
        return Path(args.manifest_path)
    return Path(args.manifest_dir) / f"ingest_manifest_{args.target_run_id}.json"


def _scope_filter(*, scope_key: str, scope_value: str) -> models.Filter | None:
    if scope_key == "collection":
        return None
    return models.Filter(must=[models.FieldCondition(key=scope_key, match=models.MatchValue(value=scope_value))])


def _fetch_points(client: QdrantClient, *, collection: str, flt: models.Filter | None) -> list[Any]:
    out: list[Any] = []
    offset: Any = None
    while True:
        resp = client.scroll(
            collection_name=collection,
            scroll_filter=flt,
            limit=256,
            with_payload=True,
            with_vectors=False,
            offset=offset,
        )
        if isinstance(resp, tuple) and len(resp) == 2:
            points, next_page = resp
        else:
            points = getattr(resp, "points", []) or []
            next_page = getattr(resp, "next_page_offset", None)
        out.extend(points)
        if next_page is None:
            break
        offset = next_page
    return out


def _set_payload(client: QdrantClient, *, collection: str, ids: list[Any], payload: dict[str, Any], batch_size: int = 256) -> None:
    for i in range(0, len(ids), batch_size):
        client.set_payload(collection_name=collection, payload=payload, points=ids[i : i + batch_size])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
    args = parse_args()
    if args.scope_key in {"source", "doc_type"} and not args.scope_value.strip():
        raise SystemExit("--scope-value is required when --scope-key is source/doc_type")
    do_apply = bool(args.apply and not args.dry_run)
    run_id = args.run_id.strip() or _default_rollback_run_id()
    manifest_path = _manifest_path(args)
    manifest = _load_manifest(manifest_path)
    manifest_items = manifest.get("items", [])
    if not isinstance(manifest_items, list):
        raise SystemExit("Invalid manifest: items list missing")

    target_ids = {str(item.get("id")) for item in manifest_items if isinstance(item, dict) and item.get("id")}
    if args.scope_key in {"source", "doc_type"}:
        target_ids = {
            str(item.get("id"))
            for item in manifest_items
            if isinstance(item, dict) and item.get("id") and str(item.get(args.scope_key) or "") == args.scope_value
        }
    collection = _resolve_collection_name(args.collection or _required_env("COLLECTION_NAME"), args.env)
    qdrant_url = args.qdrant_url or _required_env("QDRANT_URL")
    client = QdrantClient(url=qdrant_url, api_key=args.qdrant_api_key or None)
    points = _fetch_points(
        client,
        collection=collection,
        flt=_scope_filter(scope_key=args.scope_key, scope_value=args.scope_value.strip()),
    )
    existing_ids = {str(p.id) for p in points}
    reactivate_ids = sorted(existing_ids.intersection(target_ids))
    tombstone_ids = sorted(existing_ids.difference(target_ids))

    report = {
        "rollback_run_id": run_id,
        "target_run_id": args.target_run_id,
        "manifest_path": str(manifest_path),
        "collection": collection,
        "scope_key": args.scope_key,
        "scope_value": args.scope_value.strip(),
        "dry_run": bool(args.dry_run),
        "apply": do_apply,
        "counts": {
            "target_ids": len(target_ids),
            "existing_ids_in_scope": len(existing_ids),
            "reactivate_ids": len(reactivate_ids),
            "tombstone_ids": len(tombstone_ids),
        },
        "reactivate_ids": reactivate_ids,
        "tombstone_ids": tombstone_ids,
    }

    if do_apply:
        if reactivate_ids:
            _set_payload(
                client,
                collection=collection,
                ids=reactivate_ids,
                payload={"lifecycle_status": "active", "deleted_at": None, "deleted_by_run_id": None},
            )
        if tombstone_ids:
            _set_payload(
                client,
                collection=collection,
                ids=tombstone_ids,
                payload={
                    "lifecycle_status": "deleted",
                    "deleted_at": datetime.now(_EST).isoformat(),
                    "deleted_by_run_id": run_id,
                },
            )

    out_dir = Path(args.output_dir) if args.output_dir.strip() else manifest_path.parent / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"rollback_actions_{run_id}.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    logger.info("Wrote rollback action report: %s", out_path)
    logger.info(
        "Rollback summary run_id=%s target=%s reactivate=%d tombstone=%d apply=%s dry_run=%s",
        run_id,
        args.target_run_id,
        len(reactivate_ids),
        len(tombstone_ids),
        do_apply,
        bool(args.dry_run),
    )


if __name__ == "__main__":
    main()
