#!/usr/bin/env python3
"""Build metadata-rich, filter-ready ingest payloads from chunk JSON (`chunks_*.json`)."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

logger = logging.getLogger(__name__)
_EST = timezone(timedelta(hours=-5), name="EST")


UUID_NAMESPACE = uuid.NAMESPACE_URL
_APP_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _APP_DIR.parent
load_dotenv(_ROOT_DIR / ".env")
load_dotenv()


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


def parse_args() -> argparse.Namespace:
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
    return parser.parse_args()


def _now_iso_est() -> str:
    return datetime.now(_EST).replace(microsecond=0).isoformat()


def _default_run_id() -> str:
    return datetime.now(_EST).strftime("run_%Y%m%d_%H%M%S_EST")


def _token_count(text: str) -> int:
    # Lightweight approximation without extra dependencies.
    return len(re.findall(r"\S+", text))


def _build_embed_text(section: str, text: str, synthetic_questions: list[str]) -> tuple[str, int, int]:
    header = f"[SECTION: {section}]"
    question_lines = [f"Q: {q.strip()}" for q in synthetic_questions if q and q.strip()]
    embed_text = "\n".join([header, text, *question_lines]).strip()
    return embed_text, len(question_lines), 0


def _doc_type_from_name(path: Path) -> str:
    stem = path.stem
    if stem.startswith("chunks_"):
        return stem[len("chunks_") :]
    return stem


def _source_name(doc_type: str, source_prefix: str) -> str:
    if source_prefix:
        return f"{source_prefix}_{doc_type}"
    return f"{doc_type}_source"


def _tags_for(doc_type: str, section: str) -> list[str]:
    return [doc_type, section.lower().replace(" ", "_")]


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _point_id(content_hash: str) -> str:
    return str(uuid.uuid5(UUID_NAMESPACE, content_hash))


def _load_chunks(path: Path) -> list[dict[str, Any]]:
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


def _to_point(
    *,
    chunk: dict[str, Any],
    doc_type: str,
    source: str,
    language: str,
    ingest_run_id: str,
    ingest_ts: str,
) -> dict[str, Any]:
    section = str(chunk.get("section") or "ROOT")
    text = str(chunk.get("text") or "").strip()
    if not text:
        raise ValueError(f"chunk {chunk.get('chunk_id', '<unknown>')} has empty text")
    chunk_id = str(chunk.get("chunk_id") or "")
    if not chunk_id:
        raise ValueError("chunk_id is required")

    synthetic_questions_raw = chunk.get("synthetic_questions", [])
    if not isinstance(synthetic_questions_raw, list):
        synthetic_questions_raw = []
    synthetic_questions = [str(q).strip() for q in synthetic_questions_raw if str(q).strip()]

    token_count = _token_count(text)
    embed_text, used_q, trimmed_q = _build_embed_text(section, text, synthetic_questions)
    embed_token_count = _token_count(embed_text)
    content_hash = _content_hash(text)
    point_id = _point_id(content_hash)

    payload = {
        "chunk_id": chunk_id,
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
        "ingest_run_id": ingest_run_id,
        "ingest_ts": ingest_ts,
    }
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
        "files": [],
        "stats": {
            "files_total": 0,
            "chunks_input": 0,
            "chunks_total_prepared": 0,
            "chunks_failed": 0,
        },
    }

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

    summary_path = out_dir / "ingest_prepare_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def main() -> None:
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

