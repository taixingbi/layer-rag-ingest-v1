#!/usr/bin/env python3
"""Reconcile Qdrant points against an ingest manifest (dry-run by default)."""

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


def _now_est_iso() -> str:
    return datetime.now(_EST).isoformat()


def _default_run_id() -> str:
    return datetime.now(_EST).strftime("reconcile_%Y%m%d_%H%M%S_EST")


def _parse_deleted_at(value: Any) -> datetime | None:
    if not value:
        return None
    s = str(value).strip()
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reconcile stale points against a run manifest.")
    p.add_argument("--manifest-path", required=True, help="Path to ingest_manifest_<run_id>.json")
    p.add_argument("--collection", default=(os.getenv("COLLECTION_NAME") or "").strip())
    p.add_argument("--env", default=(os.getenv("ENV") or "").strip())
    p.add_argument("--qdrant-url", default=(os.getenv("QDRANT_URL") or "").strip())
    p.add_argument("--qdrant-api-key", default=(os.getenv("QDRANT_API_KEY") or "").strip())
    p.add_argument(
        "--scope-key",
        choices=["source", "doc_type", "collection"],
        default="collection",
        help="Scope key for reconciliation (default: collection).",
    )
    p.add_argument("--scope-value", default="", help="Scope value for source/doc_type.")
    p.add_argument("--delete-mode", choices=["off", "soft", "hard"], default="soft")
    p.add_argument(
        "--apply-soft-delete",
        action="store_true",
        help="Apply soft delete to stale points (otherwise preview only).",
    )
    p.add_argument(
        "--apply-hard-delete",
        action="store_true",
        help="Apply hard delete for retention-eligible tombstones.",
    )
    p.add_argument("--retention-days", type=int, default=30)
    p.add_argument("--dry-run", action="store_true", help="Preview mode; no mutations.")
    p.add_argument("--output-dir", default="", help="Directory for stale/delete action reports.")
    p.add_argument("--run-id", default="", help="Optional fixed reconcile run id.")
    return p.parse_args()


def _load_manifest(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Manifest must be a JSON object.")
    return data


def _scope_matches_item(item: dict[str, Any], *, scope_key: str, scope_value: str) -> bool:
    if scope_key == "collection":
        return True
    return str(item.get(scope_key) or "") == scope_value


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


def _scope_filter(*, scope_key: str, scope_value: str, lifecycle_status: str | None = None) -> models.Filter | None:
    must: list[models.FieldCondition] = []
    if scope_key in {"source", "doc_type"}:
        must.append(models.FieldCondition(key=scope_key, match=models.MatchValue(value=scope_value)))
    if lifecycle_status:
        must.append(models.FieldCondition(key="lifecycle_status", match=models.MatchValue(value=lifecycle_status)))
    if not must:
        return None
    return models.Filter(must=must)


def _set_payload_in_batches(
    client: QdrantClient, *, collection: str, ids: list[Any], payload: dict[str, Any], batch_size: int = 256
) -> None:
    for i in range(0, len(ids), batch_size):
        batch = ids[i : i + batch_size]
        client.set_payload(
            collection_name=collection,
            payload=payload,
            points=batch,
        )


def _delete_in_batches(client: QdrantClient, *, collection: str, ids: list[Any], batch_size: int = 256) -> None:
    for i in range(0, len(ids), batch_size):
        batch = ids[i : i + batch_size]
        client.delete(
            collection_name=collection,
            points_selector=models.PointIdsList(points=batch),
            wait=True,
        )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
    args = parse_args()
    run_id = args.run_id.strip() or _default_run_id()
    manifest_path = Path(args.manifest_path)
    manifest = _load_manifest(manifest_path)
    collection = _resolve_collection_name(args.collection or _required_env("COLLECTION_NAME"), args.env)
    qdrant_url = args.qdrant_url or _required_env("QDRANT_URL")
    out_dir = Path(args.output_dir) if args.output_dir else manifest_path.parent / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.scope_key in {"source", "doc_type"} and not args.scope_value.strip():
        raise SystemExit("--scope-value is required when --scope-key is source/doc_type")
    scope_value = args.scope_value.strip()

    manifest_items = manifest.get("items", [])
    if not isinstance(manifest_items, list):
        raise SystemExit("Manifest missing items list")
    manifest_ids = {
        str(item.get("id"))
        for item in manifest_items
        if isinstance(item, dict) and item.get("id") and _scope_matches_item(item, scope_key=args.scope_key, scope_value=scope_value)
    }

    client = QdrantClient(url=qdrant_url, api_key=args.qdrant_api_key or None)
    active_points = _fetch_points(
        client,
        collection=collection,
        flt=_scope_filter(scope_key=args.scope_key, scope_value=scope_value, lifecycle_status="active"),
    )
    active_ids = {str(p.id) for p in active_points}
    stale_ids = sorted(active_ids - manifest_ids)

    stale_candidates = {
        "reconcile_run_id": run_id,
        "collection": collection,
        "scope_key": args.scope_key,
        "scope_value": scope_value,
        "manifest_path": str(manifest_path),
        "counts": {
            "manifest_ids": len(manifest_ids),
            "active_ids_in_scope": len(active_ids),
            "stale_ids": len(stale_ids),
        },
        "stale_ids": stale_ids,
    }
    stale_path = out_dir / f"stale_candidates_{run_id}.json"
    stale_path.write_text(json.dumps(stale_candidates, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    actions: dict[str, Any] = {
        "reconcile_run_id": run_id,
        "collection": collection,
        "scope_key": args.scope_key,
        "scope_value": scope_value,
        "delete_mode": args.delete_mode,
        "dry_run": bool(args.dry_run),
        "applied_soft_delete": 0,
        "applied_hard_delete": 0,
        "soft_delete_ids": [],
        "hard_delete_ids": [],
    }

    do_soft = args.delete_mode == "soft" and args.apply_soft_delete and not args.dry_run
    if stale_ids and do_soft:
        _set_payload_in_batches(
            client,
            collection=collection,
            ids=stale_ids,
            payload={
                "lifecycle_status": "deleted",
                "deleted_at": _now_est_iso(),
                "deleted_by_run_id": run_id,
            },
        )
        actions["applied_soft_delete"] = len(stale_ids)
        actions["soft_delete_ids"] = stale_ids

    if args.delete_mode == "hard":
        deleted_points = _fetch_points(
            client,
            collection=collection,
            flt=_scope_filter(scope_key=args.scope_key, scope_value=scope_value, lifecycle_status="deleted"),
        )
        cutoff = datetime.now(_EST) - timedelta(days=max(0, int(args.retention_days)))
        hard_candidates: list[Any] = []
        for point in deleted_points:
            payload = point.payload if isinstance(point.payload, dict) else {}
            deleted_at = _parse_deleted_at(payload.get("deleted_at"))
            if deleted_at is None:
                continue
            if deleted_at.tzinfo is None:
                deleted_at = deleted_at.replace(tzinfo=_EST)
            if deleted_at <= cutoff:
                hard_candidates.append(point.id)
        actions["hard_delete_ids"] = [str(x) for x in hard_candidates]
        if hard_candidates and args.apply_hard_delete and not args.dry_run:
            _delete_in_batches(client, collection=collection, ids=hard_candidates)
            actions["applied_hard_delete"] = len(hard_candidates)

    actions_path = out_dir / f"delete_actions_{run_id}.json"
    actions_path.write_text(json.dumps(actions, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    logger.info("Wrote stale candidates: %s", stale_path)
    logger.info("Wrote delete actions: %s", actions_path)
    logger.info(
        "Reconcile summary run_id=%s stale=%d soft_deleted=%d hard_deleted=%d dry_run=%s",
        run_id,
        len(stale_ids),
        int(actions["applied_soft_delete"]),
        int(actions["applied_hard_delete"]),
        bool(args.dry_run),
    )


if __name__ == "__main__":
    main()
