#!/usr/bin/env python3
"""Build metadata-rich, filter-ready ingest payloads from chunk JSON (`chunks_*.json`)."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)
_EST = timezone(timedelta(hours=-5), name="EST")
_TOKENIZER = None
_TOKENIZER_MODEL = ""


UUID_NAMESPACE = uuid.NAMESPACE_URL
DEFAULT_ID_KEY_VERSION = "v3"
_APP_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _APP_DIR.parent
load_dotenv(_ROOT_DIR / ".env")
load_dotenv()


def _resolve_collection_name(collection_name: str, env_name: str) -> str:
    """ resolve collection name."""
    c = (collection_name or "").strip()
    if not c:
        raise RuntimeError("Collection name is empty. Set COLLECTION_NAME or pass --collection.")
    env = env_name.strip().lower()
    if env in {"dev", "qa", "prod"}:
        suffix = f"_{env}"
        if not c.endswith(suffix):
            return f"{c}{suffix}"
    return c


def parse_args() -> argparse.Namespace:
    """Parse args."""
    parser = argparse.ArgumentParser(
        description=(
            "Read chunk JSON files from data/, attach metadata/filter fields, "
            "and write Stage-5-style PointStruct payload files."
        )
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Input directory containing chunks_*.json (default: data).",
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Output directory for points_*.json and run summary (default: data).",
    )
    parser.add_argument(
        "--pattern",
        default="chunks_*.json",
        help="Glob pattern under --data-dir (default: chunks_*.json).",
    )
    parser.add_argument(
        "--source-prefix",
        default="",
        help="Optional prefix for payload.source (example: taixing).",
    )
    parser.add_argument(
        "--default-language",
        default="en",
        help="Default payload.language when unknown (default: en).",
    )
    parser.add_argument(
        "--collection",
        default=(os.getenv("COLLECTION_NAME") or "").strip(),
        help=(
            "Base collection name for run summary "
            "(default: COLLECTION_NAME env; ENV in {dev,qa,prod} adds matching suffix)."
        ),
    )
    parser.add_argument(
        "--env",
        default=(os.getenv("ENV") or "").strip(),
        help=(
            "Environment name (default: ENV env). "
            "If set to dev/qa/prod, collection becomes <name>_<env>."
        ),
    )
    parser.add_argument(
        "--ingest-run-id",
        default="",
        help="Optional fixed ingest_run_id. Default: run_YYYYMMDD_HHMMSS.",
    )
    parser.add_argument(
        "--ingest-ts",
        default="",
        help="Optional fixed ingest timestamp (ISO-8601). Default: now EST.",
    )
    parser.add_argument(
        "--id-key-version",
        default=DEFAULT_ID_KEY_VERSION,
        help=(
            "Identity schema version used in canonical UUID key "
            f"(default: {DEFAULT_ID_KEY_VERSION}; use v2 for backward-compatible IDs)."
        ),
    )
    parser.add_argument(
        "--document-version",
        default="v1",
        help="Document/source revision version tag (default: v1).",
    )
    parser.add_argument(
        "--chunk-version",
        default="v1",
        help="Chunking strategy version tag (default: v1).",
    )
    parser.add_argument(
        "--embedding-version",
        default=(os.getenv("EMBEDDING_VERSION") or os.getenv("EMBEDDING_MODEL") or "BAAI/bge-m3").strip(),
        help=(
            "Embedding space version tag (default: EMBEDDING_VERSION env, "
            "else EMBEDDING_MODEL env, else BAAI/bge-m3)."
        ),
    )
    parser.add_argument(
        "--profile-role-map",
        default="",
        help=(
            "Optional JSON object mapping role by source/document "
            '(example: \'{"personal_profile":"AI Infrastructure Engineer",'
            '"personal_profile:profile":"AI Infrastructure Engineer"}\').'
        ),
    )
    parser.add_argument(
        "--profile-role-map-file",
        default="",
        help=(
            "Optional path to JSON object mapping role by source/document. "
            "Merge order: file first, then --profile-role-map overrides."
        ),
    )
    parser.add_argument(
        "--access-control-file",
        default="",
        help=(
            "Optional JSON file mapping source/document -> access policy object "
            '(keys: roles, groups, teams). If omitted, auto-loads '
            "<dataset-root>/raw/access_control.json when present."
        ),
    )
    return parser.parse_args()


def _now_iso_est() -> str:
    """ now iso est."""
    return datetime.now(_EST).replace(microsecond=0).isoformat()


def _default_run_id() -> str:
    """ default run id."""
    return datetime.now(_EST).strftime("run_%Y%m%d_%H%M%S_EST")


def _token_count(text: str) -> int:
    """ token count."""
    global _TOKENIZER, _TOKENIZER_MODEL
    if _TOKENIZER is None:
        model_name = (os.getenv("EMBEDDING_MODEL") or "BAAI/bge-m3").strip() or "BAAI/bge-m3"
        _TOKENIZER_MODEL = model_name
        _TOKENIZER = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        logger.info("Loaded tokenizer model for token counting: %s", _TOKENIZER_MODEL)
    return int(len(_TOKENIZER.encode(text, add_special_tokens=False)))


def _build_embed_text(section: str, text: str, synthetic_questions: list[str]) -> tuple[str, int, int]:
    """ build embed text."""
    header = f"[SECTION: {section}]"
    question_lines = [f"Q: {q.strip()}" for q in synthetic_questions if q and q.strip()]
    embed_text = "\n".join([header, text, *question_lines]).strip()
    return embed_text, len(question_lines), 0


def _doc_type_from_name(path: Path) -> str:
    """ doc type from name."""
    stem = path.stem
    if stem.startswith("chunks_"):
        return stem[len("chunks_") :]
    return stem


def _source_name(doc_type: str, source_prefix: str) -> str:
    """ source name."""
    if source_prefix:
        return f"{source_prefix}_{doc_type}"
    return f"{doc_type}_source"


def _tags_for(doc_type: str, section: str) -> list[str]:
    """ tags for."""
    return [doc_type, section.lower().replace(" ", "_")]


def _content_hash(text: str) -> str:
    """ content hash."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _point_id(
    *,
    id_key_version: str,
    source: str,
    document_id: str,
    chunk_id: str,
    document_version: str,
    chunk_version: str,
    embedding_version: str,
) -> str:
    """Deterministic point id by canonical identity key."""
    key_version = (id_key_version or "").strip() or DEFAULT_ID_KEY_VERSION
    if key_version == "v2":
        canonical = f"v2|source={source}|document_id={document_id}|chunk_id={chunk_id}"
    else:
        canonical = (
            f"{key_version}|source={source}|document_id={document_id}|"
            f"document_version={document_version}|chunk_version={chunk_version}|"
            f"embedding_version={embedding_version}|chunk_id={chunk_id}"
        )
    return str(uuid.uuid5(UUID_NAMESPACE, canonical))


def _load_chunks(path: Path) -> list[dict[str, Any]]:
    """ load chunks."""
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


def _parse_role_map_json(raw: str, label: str) -> dict[str, str]:
    """Parse and validate role mapping JSON object."""
    text = (raw or "").strip()
    if not text:
        return {}
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid {label} JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{label} must be a JSON object of key -> role")
    out: dict[str, str] = {}
    for key, value in data.items():
        k = str(key).strip()
        v = str(value).strip()
        if not k or not v:
            continue
        out[k] = v
    return out


def _load_profile_role_map(args: argparse.Namespace) -> dict[str, str]:
    """Load optional profile role mapping from file and inline JSON."""
    merged: dict[str, str] = {}
    map_file = (args.profile_role_map_file or "").strip()
    if map_file:
        file_path = Path(map_file)
        raw = file_path.read_text(encoding="utf-8")
        merged.update(_parse_role_map_json(raw, "--profile-role-map-file"))
    merged.update(_parse_role_map_json(args.profile_role_map, "--profile-role-map"))
    return merged


def _resolve_profile_role(*, source: str, document_id: str, profile_role_map: dict[str, str]) -> str:
    """Resolve profile role by source first, then document fallback."""
    if not profile_role_map:
        return ""
    src = source.strip()
    doc = document_id.strip()
    lookup_keys = []
    if src and doc:
        lookup_keys.append(f"{src}:{doc}")
    if src:
        lookup_keys.append(src)
    if doc:
        lookup_keys.append(doc)
    for key in lookup_keys:
        role = str(profile_role_map.get(key) or "").strip()
        if role:
            return role
    return "role"


def _as_clean_list(value: Any) -> list[str]:
    """Normalize ACL list-like values to clean string list."""
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    v = str(value).strip()
    return [v] if v else []


def _parse_access_control_json(raw: str, label: str) -> dict[str, dict[str, Any]]:
    """Parse and validate access control mapping JSON."""
    text = (raw or "").strip()
    if not text:
        return {}
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid {label} JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{label} must be a JSON object of key -> access policy")
    out: dict[str, dict[str, Any]] = {}
    for key, policy in data.items():
        k = str(key).strip()
        if not k or not isinstance(policy, dict):
            continue
        roles = _as_clean_list(policy.get("roles"))
        groups = _as_clean_list(policy.get("groups"))
        teams = _as_clean_list(policy.get("teams"))
        normalized: dict[str, Any] = {}
        if roles:
            normalized["roles"] = roles
        if groups:
            normalized["groups"] = groups
        if teams:
            normalized["teams"] = teams
        if normalized:
            out[k] = normalized
    return out


def _default_access_control_path(data_dir: Path) -> Path:
    """Resolve default access_control.json beside dataset raw folder."""
    dataset_root = data_dir.parent
    return dataset_root / "raw" / "access_control.json"


def _load_access_control_map(args: argparse.Namespace, data_dir: Path) -> dict[str, dict[str, Any]]:
    """Load optional access control mapping from explicit/default file."""
    map_path = (args.access_control_file or "").strip()
    candidate = Path(map_path) if map_path else _default_access_control_path(data_dir)
    if not candidate.exists():
        return {}
    raw = candidate.read_text(encoding="utf-8")
    acl_map = _parse_access_control_json(raw, "--access-control-file")
    if acl_map:
        logger.info("Loaded access control map entries=%d from %s", len(acl_map), candidate)
    return acl_map


def _resolve_access_policy(
    *, source: str, document_id: str, access_control_map: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Resolve ACL policy by source/document lookup precedence."""
    if not access_control_map:
        return {}
    src = source.strip()
    doc = document_id.strip()
    lookup_keys = []
    if src and doc:
        lookup_keys.append(f"{src}:{doc}")
    if src:
        lookup_keys.append(src)
    if doc:
        lookup_keys.append(doc)
    for key in lookup_keys:
        policy = access_control_map.get(key)
        if isinstance(policy, dict) and policy:
            return policy
    return {}


def _to_point(
    *,
    chunk: dict[str, Any],
    doc_type: str,
    source: str,
    language: str,
    ingest_run_id: str,
    ingest_ts: str,
    id_key_version: str,
    document_version: str,
    chunk_version: str,
    embedding_version: str,
    profile_role_map: dict[str, str],
    access_control_map: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """ to point."""
    section = str(chunk.get("section") or "ROOT")
    text = str(chunk.get("text") or "").strip()
    if not text:
        raise ValueError(f"chunk {chunk.get('chunk_id', '<unknown>')} has empty text")
    chunk_id = str(chunk.get("chunk_id") or "")
    if not chunk_id:
        raise ValueError("chunk_id is required")
    document_id = str(chunk.get("document_id") or "").strip()
    if not document_id:
        # Backward compatibility for existing chunks_*.json generated before document_id was introduced.
        document_id = doc_type

    synthetic_questions_raw = chunk.get("synthetic_questions", [])
    if not isinstance(synthetic_questions_raw, list):
        synthetic_questions_raw = []
    synthetic_questions = [str(q).strip() for q in synthetic_questions_raw if str(q).strip()]

    token_count = _token_count(text)
    embed_text, used_q, trimmed_q = _build_embed_text(section, text, synthetic_questions)
    embed_token_count = _token_count(embed_text)
    content_hash = _content_hash(text)
    point_id = _point_id(
        id_key_version=id_key_version,
        source=source,
        document_id=document_id,
        chunk_id=chunk_id,
        document_version=document_version,
        chunk_version=chunk_version,
        embedding_version=embedding_version,
    )

    payload = {
        "chunk_id": chunk_id,
        "document_id": document_id,
        "chunk_id_parent": chunk.get("chunk_id_parent"),
        "was_split": bool(chunk.get("was_split", False)),
        "split_index": chunk.get("split_index"),
        "section": section,
        "doc_type": doc_type,
        "language": language,
        "tags": _tags_for(doc_type, section),
        "text": text,
        "embed_text": embed_text,
        "token_count": token_count,
        "embed_token_count": embed_token_count,
        "synthetic_questions_used": used_q,
        "synthetic_questions_trimmed": trimmed_q,
        "synthetic_questions": synthetic_questions,
        "content_hash": content_hash,
        "source": source,
        "id_key_version": id_key_version,
        "document_version": document_version,
        "chunk_version": chunk_version,
        "embedding_version": embedding_version,
        "ingest_run_id": ingest_run_id,
        "ingest_ts": ingest_ts,
        "lifecycle_status": "active",
        "deleted_at": None,
        "deleted_by_run_id": None,
    }
    profile_role = _resolve_profile_role(
        source=source, document_id=document_id, profile_role_map=profile_role_map
    )
    if profile_role:
        payload["profile"] = {"role": profile_role}
    access_policy = _resolve_access_policy(
        source=source, document_id=document_id, access_control_map=access_control_map
    )
    if access_policy:
        payload["access"] = access_policy
    return {"id": point_id, "vector": [], "payload": payload}


def run_prepare(args: argparse.Namespace) -> dict[str, Any]:
    """
    Build points_*.json from chunk JSON under data_dir; write ingest_prepare_summary.json.
    Returns the summary dict (same structure as written to disk).
    """
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ingest_run_id = args.ingest_run_id.strip() or _default_run_id()
    ingest_ts = args.ingest_ts.strip() or _now_iso_est()
    id_key_version = (args.id_key_version or "").strip() or DEFAULT_ID_KEY_VERSION
    document_version = (args.document_version or "").strip() or "v1"
    chunk_version = (args.chunk_version or "").strip() or "v1"
    embedding_version = (args.embedding_version or "").strip() or "BAAI/bge-m3"

    files = sorted(data_dir.glob(args.pattern))
    if not files:
        raise SystemExit(f"No files matched {args.pattern!r} under {data_dir}")

    collection = _resolve_collection_name(
        args.collection or (os.getenv("COLLECTION_NAME") or "").strip(),
        args.env or (os.getenv("ENV") or "").strip(),
    )

    summary: dict[str, Any] = {
        "ingest_run_id": ingest_run_id,
        "collection": collection,
        "ingest_ts": ingest_ts,
        "id_key_version": id_key_version,
        "document_version": document_version,
        "chunk_version": chunk_version,
        "embedding_version": embedding_version,
        "files": [],
        "stats": {
            "files_total": 0,
            "chunks_input": 0,
            "chunks_total_prepared": 0,
            "chunks_failed": 0,
        },
    }
    manifest: dict[str, Any] = {
        "ingest_run_id": ingest_run_id,
        "collection": collection,
        "ingest_ts": ingest_ts,
        "id_key_version": id_key_version,
        "document_version": document_version,
        "chunk_version": chunk_version,
        "embedding_version": embedding_version,
        "items": [],
        "stats": {"points_total": 0, "by_source": {}},
    }
    profile_role_map = _load_profile_role_map(args)
    access_control_map = _load_access_control_map(args, data_dir)

    for file_path in files:
        doc_type = _doc_type_from_name(file_path)
        source = _source_name(doc_type, args.source_prefix)
        chunks = _load_chunks(file_path)

        points: list[dict[str, Any]] = []
        file_failed = 0
        for chunk in chunks:
            try:
                points.append(
                    _to_point(
                        chunk=chunk,
                        doc_type=doc_type,
                        source=source,
                        language=args.default_language,
                        ingest_run_id=ingest_run_id,
                        ingest_ts=ingest_ts,
                        id_key_version=id_key_version,
                        document_version=document_version,
                        chunk_version=chunk_version,
                        embedding_version=embedding_version,
                        profile_role_map=profile_role_map,
                        access_control_map=access_control_map,
                    )
                )
            except Exception:
                file_failed += 1

        out_path = out_dir / f"points_{doc_type}.json"
        out_path.write_text(json.dumps(points, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        summary["files"].append(
            {
                "input_file": str(file_path),
                "output_file": str(out_path),
                "doc_type": doc_type,
                "source": source,
                "chunks_input": len(chunks),
                "chunks_prepared": len(points),
                "chunks_failed": file_failed,
            }
        )
        summary["stats"]["files_total"] += 1
        summary["stats"]["chunks_input"] += len(chunks)
        summary["stats"]["chunks_total_prepared"] += len(points)
        summary["stats"]["chunks_failed"] += file_failed
        for point in points:
            payload = point.get("payload") if isinstance(point, dict) else None
            if not isinstance(payload, dict):
                continue
            source_name = str(payload.get("source") or "")
            item = {
                "id": point.get("id"),
                "source": source_name,
                "doc_type": payload.get("doc_type"),
                "document_id": payload.get("document_id"),
                "chunk_id": payload.get("chunk_id"),
                "id_key_version": payload.get("id_key_version"),
                "document_version": payload.get("document_version"),
                "chunk_version": payload.get("chunk_version"),
                "embedding_version": payload.get("embedding_version"),
                "section": payload.get("section"),
                "content_hash": payload.get("content_hash"),
            }
            manifest["items"].append(item)
            manifest["stats"]["points_total"] += 1
            by_source = manifest["stats"]["by_source"]
            by_source[source_name] = int(by_source.get(source_name, 0)) + 1

    summary_path = out_dir / "ingest_prepare_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest_path = out_dir / f"ingest_manifest_{ingest_run_id}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    latest_manifest_path = out_dir / "ingest_manifest_latest.json"
    latest_manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    """Main."""
    started = time.perf_counter()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
    args = parse_args()
    summary = run_prepare(args)
    summary_path = Path(args.output_dir) / "ingest_prepare_summary.json"
    logger.info(
        "Prepared %d chunks across %d files",
        summary["stats"]["chunks_total_prepared"],
        summary["stats"]["files_total"],
    )
    logger.info("Wrote summary to %s", summary_path)
    elapsed_ms = (time.perf_counter() - started) * 1000
    logger.info("prepare_payloads total_latency_ms=%.1f", elapsed_ms)


if __name__ == "__main__":
    main()

