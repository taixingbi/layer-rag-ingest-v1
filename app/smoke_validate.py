#!/usr/bin/env python3
"""Post-upsert retrieval smoke validation for points_*.json payloads."""

from __future__ import annotations

import asyncio
import argparse
import json
import logging
import os
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

from client_embeddings import EMBEDDINGS_BASE_URL, embed_texts
from client_inference import (
    DEFAULT_CHAT_MODEL,
    INFERENCE_BASE_URL,
    async_chat_completions,
    normalize_chat_base_url,
)

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


def _load_points(path: Path) -> list[dict[str, Any]]:
    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON array")
    return [x for x in data if isinstance(x, dict)]


def _default_report_path(data_dir: Path) -> Path:
    ts = datetime.now(_EST).strftime("%Y%m%dT%H%M%S_EST")
    return data_dir / "reports" / f"smoke_validate_{ts}.json"


def _percentile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * max(0.0, min(1.0, p))
    lo = int(rank)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = rank - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def _probe_text(payload: dict[str, Any]) -> str:
    qs_raw = payload.get("synthetic_questions")
    if isinstance(qs_raw, list):
        for q in qs_raw:
            qv = str(q).strip()
            if qv:
                return qv
    text = str(payload.get("text") or "").strip()
    if not text:
        return ""
    return text[:280]


def _judge_prompt(*, probe_text: str, payload: dict[str, Any]) -> tuple[str, str]:
    context_text = str(payload.get("text") or "").strip()
    if not context_text:
        context_text = str(payload.get("embed_text") or "").strip()
    system = (
        "You are evaluating retrieval quality for a RAG smoke test. "
        "Decide if the candidate context can answer the user question using ONLY the context."
    )
    user = (
        f"Question:\n{probe_text}\n\n"
        f"Candidate context:\n---\n{context_text}\n---\n\n"
        "Return ONLY valid JSON with keys: "
        '{"verdict":"supported|not_supported|uncertain","reason":"<short reason>"}'
    )
    return system, user


def _parse_judge_response(content: str) -> tuple[str, str]:
    parsed = json.loads(content)
    verdict = str(parsed.get("verdict") or "uncertain").strip().lower()
    reason = str(parsed.get("reason") or "").strip()
    if verdict not in {"supported", "not_supported", "uncertain"}:
        verdict = "uncertain"
    return verdict, reason


async def _judge_with_llm_async(
    *,
    sem: asyncio.Semaphore,
    client: httpx.AsyncClient,
    probe_text: str,
    payload: dict[str, Any],
    chat_base_url: str,
    chat_model: str,
    chat_api_key: str | None,
) -> tuple[str, str]:
    system, user = _judge_prompt(probe_text=probe_text, payload=payload)
    async with sem:
        data = await async_chat_completions(
            client=client,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            base_url=normalize_chat_base_url(chat_base_url),
            model=chat_model,
            max_tokens=120,
            temperature=0.0,
            api_key=chat_api_key if chat_api_key else None,
            response_format={"type": "json_object"},
            timeout=120.0,
        )
    content = str(data["choices"][0]["message"]["content"])
    return _parse_judge_response(content)


async def _run_llm_judges(
    *,
    jobs: list[tuple[int, str, dict[str, Any]]],
    chat_base_url: str,
    chat_model: str,
    chat_api_key: str | None,
    max_concurrency: int = 12,
) -> dict[int, tuple[str, str]]:
    sem = asyncio.Semaphore(max(1, max_concurrency))
    timeout = httpx.Timeout(120.0, connect=15.0)
    limits = httpx.Limits(max_connections=max(20, max_concurrency * 2))
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        tasks = [
            asyncio.create_task(
                _judge_with_llm_async(
                    sem=sem,
                    client=client,
                    probe_text=probe_text,
                    payload=payload,
                    chat_base_url=chat_base_url,
                    chat_model=chat_model,
                    chat_api_key=chat_api_key,
                )
            )
            for _, probe_text, payload in jobs
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    out: dict[int, tuple[str, str]] = {}
    for (row_idx, _, _), result in zip(jobs, results):
        if isinstance(result, Exception):
            out[row_idx] = ("uncertain", f"judge_error: {type(result).__name__}: {result}")
        else:
            out[row_idx] = result
    return out


def _build_filter(source: str, section: str, doc_type: str) -> models.Filter:
    return models.Filter(
        must=[
            models.FieldCondition(key="source", match=models.MatchValue(value=source)),
            models.FieldCondition(key="section", match=models.MatchValue(value=section)),
            models.FieldCondition(key="doc_type", match=models.MatchValue(value=doc_type)),
        ]
    )


def _matches_scope(hit_payload: dict[str, Any], *, source: str, section: str, doc_type: str) -> bool:
    return (
        str(hit_payload.get("source") or "") == source
        and str(hit_payload.get("section") or "") == section
        and str(hit_payload.get("doc_type") or "") == doc_type
    )


def _search_hits(
    *,
    client: QdrantClient,
    collection: str,
    vector: list[float],
    query_filter: models.Filter,
    limit: int,
) -> list[Any]:
    # Qdrant client API differs across versions: older clients expose `search`,
    # newer clients expose `query_points`.
    if hasattr(client, "query_points"):
        resp = client.query_points(
            collection_name=collection,
            query=vector,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        )
        points = getattr(resp, "points", None)
        if isinstance(points, list):
            return points
        if isinstance(resp, list):
            return resp
        return []
    return client.search(
        collection_name=collection,
        query_vector=vector,
        query_filter=query_filter,
        limit=limit,
        with_payload=True,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run post-upsert retrieval smoke validation against points_*.json "
            "using filtered semantic search in Qdrant."
        )
    )
    p.add_argument("--data-dir", default="data", help="Directory containing points_*.json.")
    p.add_argument("--pattern", default="points_*.json", help="Glob pattern under --data-dir.")
    p.add_argument(
        "--collection",
        default=(os.getenv("COLLECTION_NAME") or "").strip(),
        help="Base collection name (default: COLLECTION_NAME env).",
    )
    p.add_argument(
        "--env",
        default=(os.getenv("ENV") or "").strip(),
        help="Environment name; dev/qa/prod append matching collection suffix.",
    )
    p.add_argument("--qdrant-url", default=(os.getenv("QDRANT_URL") or "").strip(), help="Qdrant URL.")
    p.add_argument(
        "--qdrant-api-key",
        default=(os.getenv("QDRANT_API_KEY") or "").strip(),
        help="Optional Qdrant API key.",
    )
    p.add_argument(
        "--embedding-base-url",
        default=(os.getenv("EMBEDDINGS_BASE_URL") or "").strip() or EMBEDDINGS_BASE_URL,
        help="Embeddings service root without /v1.",
    )
    p.add_argument(
        "--embedding-model",
        default=(os.getenv("EMBEDDING_MODEL") or "").strip() or "BAAI/bge-m3",
        help="Embedding model id.",
    )
    p.add_argument(
        "--embedding-api-key",
        default=(os.getenv("EMBEDDING_API_KEY") or "").strip(),
        help="Optional Bearer token for embeddings API.",
    )
    p.add_argument("--judge-enabled", action="store_true", help="Enable LLM judge as secondary signal.")
    p.add_argument(
        "--judge-rescue-floor",
        type=float,
        default=0.58,
        help="Only failed probes with score >= this value are sent to judge (default: 0.58).",
    )
    p.add_argument(
        "--chat-base-url",
        default=(os.getenv("CHAT_BASE_URL") or "").strip() or INFERENCE_BASE_URL,
        help="Chat API root (default: CHAT_BASE_URL or INFERENCE_BASE_URL).",
    )
    p.add_argument(
        "--chat-model",
        default=(os.getenv("CHAT_MODEL") or "").strip() or DEFAULT_CHAT_MODEL,
        help="Chat model id for judge (default: CHAT_MODEL or built-in default).",
    )
    p.add_argument(
        "--chat-api-key",
        default=(os.getenv("CHAT_API_KEY") or "").strip(),
        help="Optional Bearer token for judge (default: CHAT_API_KEY).",
    )
    p.add_argument("--threshold", type=float, default=0.75, help="Minimum top score to pass (default: 0.75).")
    p.add_argument("--max-probes", type=int, default=0, help="Maximum probes to validate; 0 means all.")
    p.add_argument("--report-path", default="", help="Optional report path (default: <data-dir>/reports/...).")
    p.add_argument("--strict", action="store_true", help="Exit non-zero if any probe fails.")
    return p.parse_args()


def _run_smoke(
    *,
    data_dir: Path,
    pattern: str,
    collection: str,
    qdrant_url: str,
    qdrant_api_key: str,
    embedding_base_url: str,
    embedding_model: str,
    embedding_api_key: str,
    judge_enabled: bool,
    judge_rescue_floor: float,
    chat_base_url: str,
    chat_model: str,
    chat_api_key: str,
    threshold: float,
    max_probes: int,
) -> dict[str, Any]:
    files = sorted(data_dir.glob(pattern))
    if not files:
        raise SystemExit(f"No files matched {pattern!r} under {data_dir}")

    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    for fp in files:
        for row in _load_points(fp):
            payload = row.get("payload")
            if not isinstance(payload, dict):
                continue
            source = str(payload.get("source") or "").strip()
            section = str(payload.get("section") or "").strip()
            doc_type = str(payload.get("doc_type") or "").strip()
            if not source or not section or not doc_type:
                continue
            key = (source, section, doc_type)
            if key in grouped:
                continue
            probe = _probe_text(payload)
            if not probe:
                continue
            grouped[key] = {
                "source": source,
                "section": section,
                "doc_type": doc_type,
                "probe_text": probe,
            }

    probes = list(grouped.values())
    if max_probes > 0:
        probes = probes[:max_probes]
    if not probes:
        raise SystemExit("No valid probes could be generated from points payloads.")

    texts = [str(p["probe_text"]) for p in probes]
    with httpx.Client(timeout=120.0) as http_client:
        vectors = embed_texts(
            base_url=embedding_base_url,
            model=embedding_model,
            texts=texts,
            api_key=embedding_api_key or None,
            client=http_client,
            timeout=120.0,
        )

    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key or None)
    rows: list[dict[str, Any]] = []
    pass_count = 0
    fail_count = 0
    vector_pass_count = 0
    llm_rescued_count = 0
    reason_counts: Counter[str] = Counter()
    fail_group_counts: Counter[tuple[str, str]] = Counter()
    scored_values: list[float] = []
    judge_jobs: list[tuple[int, str, dict[str, Any]]] = []
    for probe, vec in zip(probes, vectors):
        source = str(probe["source"])
        section = str(probe["section"])
        doc_type = str(probe["doc_type"])
        flt = _build_filter(source, section, doc_type)
        hits = _search_hits(
            client=client,
            collection=collection,
            vector=vec,
            query_filter=flt,
            limit=1,
        )
        if not hits:
            fail_count += 1
            reason_counts["no_hits"] += 1
            fail_group_counts[(source, section)] += 1
            rows.append(
                {
                    "source": source,
                    "section": section,
                    "doc_type": doc_type,
                    "probe_text": probe["probe_text"],
                    "score": None,
                    "scope_match": False,
                    "passed": False,
                    "reason": "no_hits",
                }
            )
            continue

        top = hits[0]
        top_payload = top.payload if isinstance(top.payload, dict) else {}
        score = float(top.score or 0.0)
        scored_values.append(score)
        scope_match = _matches_scope(top_payload, source=source, section=section, doc_type=doc_type)
        vector_passed = score >= threshold and scope_match
        judge_invoked = False
        judge_label: str | None = None
        judge_reason: str | None = None
        final_reason = "vector_pass"

        if vector_passed:
            vector_pass_count += 1
            passed = True
        else:
            fail_count += 1
            fail_group_counts[(source, section)] += 1
            if not scope_match and score < threshold:
                reason = "below_threshold_and_scope_mismatch"
                reason_counts["below_threshold"] += 1
                reason_counts["scope_mismatch"] += 1
            elif not scope_match:
                reason = "scope_mismatch"
                reason_counts["scope_mismatch"] += 1
            else:
                reason = "below_threshold"
                reason_counts["below_threshold"] += 1
            final_reason = reason
            passed = False

            # Secondary signal: optionally rescue borderline, in-scope failures.
            if judge_enabled and scope_match and score >= judge_rescue_floor:
                judge_invoked = True
                judge_jobs.append((len(rows), str(probe["probe_text"]), top_payload))

        if passed and final_reason == "vector_pass":
            pass_count += 1
        rows.append(
            {
                "source": source,
                "section": section,
                "doc_type": doc_type,
                "probe_text": probe["probe_text"],
                "score": score,
                "scope_match": scope_match,
                "passed": passed,
                "threshold": threshold,
                "reason": None if passed else reason,
                "vector_passed": vector_passed,
                "judge_invoked": judge_invoked,
                "judge_label": judge_label,
                "judge_reason": judge_reason,
                "final_reason": final_reason,
                "top_hit_id": str(top.id),
            }
        )

    if judge_jobs:
        judge_results = asyncio.run(
            _run_llm_judges(
                jobs=judge_jobs,
                chat_base_url=chat_base_url,
                chat_model=chat_model,
                chat_api_key=chat_api_key or None,
            )
        )
        for row_idx, (judge_label, judge_reason) in judge_results.items():
            row = rows[row_idx]
            row["judge_label"] = judge_label
            row["judge_reason"] = judge_reason
            if judge_label == "supported" and not bool(row.get("passed")):
                row["passed"] = True
                row["reason"] = None
                row["final_reason"] = "llm_rescue"
                llm_rescued_count += 1
                pass_count += 1
                fail_count -= 1

    score_stats: dict[str, float] = {}
    if scored_values:
        sorted_scores = sorted(scored_values)
        score_stats = {
            "min": sorted_scores[0],
            "p50": _percentile(sorted_scores, 0.50),
            "p90": _percentile(sorted_scores, 0.90),
            "max": sorted_scores[-1],
        }
    worst_groups = [
        {"source": src, "section": sec, "failed": failed}
        for (src, sec), failed in fail_group_counts.most_common(5)
    ]

    return {
        "generated_at": datetime.now(_EST).isoformat(),
        "summary": {
            "files": len(files),
            "probes_total": len(probes),
            "probes_passed": pass_count,
            "probes_failed": fail_count,
            "vector_passed": vector_pass_count,
            "llm_rescued": llm_rescued_count,
            "threshold": threshold,
            "collection": collection,
            "failure_reasons": dict(reason_counts),
            "score_stats": score_stats,
            "worst_groups": worst_groups,
        },
        "run_args": {
            "data_dir": str(data_dir),
            "pattern": pattern,
            "max_probes": max_probes,
        },
        "probes": rows,
    }


def run_smoke_validation(
    *,
    data_dir: str,
    pattern: str,
    collection: str,
    env: str,
    qdrant_url: str,
    qdrant_api_key: str,
    embedding_base_url: str,
    embedding_model: str,
    embedding_api_key: str,
    judge_enabled: bool,
    judge_rescue_floor: float,
    chat_base_url: str,
    chat_model: str,
    chat_api_key: str,
    threshold: float,
    max_probes: int,
    report_path: str | None,
    strict: bool,
) -> tuple[dict[str, Any], Path]:
    resolved_collection = _resolve_collection_name(collection, env)
    report = _run_smoke(
        data_dir=Path(data_dir),
        pattern=pattern,
        collection=resolved_collection,
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_api_key,
        embedding_base_url=embedding_base_url,
        embedding_model=embedding_model,
        embedding_api_key=embedding_api_key,
        judge_enabled=judge_enabled,
        judge_rescue_floor=judge_rescue_floor,
        chat_base_url=chat_base_url,
        chat_model=chat_model,
        chat_api_key=chat_api_key,
        threshold=threshold,
        max_probes=max_probes,
    )
    out_path = Path(report_path) if report_path else _default_report_path(Path(data_dir))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    failed = int(report["summary"]["probes_failed"])
    logger.info(
        "Smoke validation summary: probes=%d passed=%d failed=%d threshold=%.3f vector_passed=%d llm_rescued=%d",
        int(report["summary"]["probes_total"]),
        int(report["summary"]["probes_passed"]),
        failed,
        threshold,
        int(report["summary"].get("vector_passed", 0)),
        int(report["summary"].get("llm_rescued", 0)),
    )
    reason_summary = report["summary"].get("failure_reasons") or {}
    if reason_summary:
        logger.info("Smoke failure reasons: %s", reason_summary)
    score_stats = report["summary"].get("score_stats") or {}
    if score_stats:
        logger.info(
            "Smoke score stats: min=%.4f p50=%.4f p90=%.4f max=%.4f",
            float(score_stats.get("min", 0.0)),
            float(score_stats.get("p50", 0.0)),
            float(score_stats.get("p90", 0.0)),
            float(score_stats.get("max", 0.0)),
        )
    worst_groups = report["summary"].get("worst_groups") or []
    if worst_groups:
        logger.info("Smoke worst groups: %s", worst_groups)
    logger.info("Wrote smoke report: %s", out_path)
    if failed > 0:
        if strict:
            raise SystemExit(f"Smoke validation failed: {failed} probe(s) below threshold/scope.")
        logger.warning("Smoke validation had %d failure(s); continuing (strict mode disabled).", failed)
    return report, out_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
    args = parse_args()
    run_smoke_validation(
        data_dir=args.data_dir,
        pattern=args.pattern,
        collection=args.collection or _required_env("COLLECTION_NAME"),
        env=args.env,
        qdrant_url=args.qdrant_url or _required_env("QDRANT_URL"),
        qdrant_api_key=args.qdrant_api_key,
        embedding_base_url=args.embedding_base_url,
        embedding_model=args.embedding_model,
        embedding_api_key=args.embedding_api_key,
        judge_enabled=args.judge_enabled,
        judge_rescue_floor=args.judge_rescue_floor,
        chat_base_url=args.chat_base_url,
        chat_model=args.chat_model,
        chat_api_key=args.chat_api_key,
        threshold=args.threshold,
        max_probes=args.max_probes,
        report_path=args.report_path or None,
        strict=args.strict,
    )


if __name__ == "__main__":
    main()
