"""Generate synthetic RAG questions per chunk via chat completions."""

from __future__ import annotations

import asyncio
import argparse
from datetime import datetime, timedelta, timezone
import json
import logging
import os
import random
import re
import time
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

from client_inference import (
    DEFAULT_CHAT_MODEL,
    INFERENCE_BASE_URL,
    async_chat_completions,
    normalize_chat_base_url,
)

_APP_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _APP_DIR.parent

logger = logging.getLogger(__name__)
_EST = timezone(timedelta(hours=-5), name="EST")


def _json_from_response(content: str) -> dict[str, Any]:
    content = content.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", content)
    if fence:
        content = fence.group(1).strip()
    return json.loads(content)


async def generate_questions_for_chunk(
    *,
    client: httpx.AsyncClient | None,
    base_url: str,
    model: str,
    api_key: str | None,
    section: str,
    text: str,
    num_questions: int,
    use_json_object: bool,
) -> list[str]:
    started = time.perf_counter()
    logger.debug(
        "chat_completions: section=%r text_chars=%d questions_requested=%d json_object=%s",
        section,
        len(text),
        num_questions,
        use_json_object,
    )
    system = (
        "You help build a RAG index. Given a text chunk, propose concise questions "
        "that are answerable only from this chunk. Do not invent facts."
    )
    user = (
        f"Section: {section}\n\n"
        f"Text:\n---\n{text}\n---\n\n"
        f"Return ONLY valid JSON as: "
        f'{{"questions": ["...", "..."]}} with exactly {num_questions} distinct strings.'
    )
    data = await async_chat_completions(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        base_url=normalize_chat_base_url(base_url),
        model=model,
        max_tokens=256,
        temperature=0.3,
        api_key=api_key if api_key else None,
        client=client,
        timeout=120.0,
        response_format={"type": "json_object"} if use_json_object else None,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000
    logger.info(
        "chat_completions latency_ms=%.1f section=%r requested_questions=%d",
        elapsed_ms,
        section,
        num_questions,
    )
    content = data["choices"][0]["message"]["content"]
    parsed = _json_from_response(str(content))
    questions = parsed.get("questions")
    if not isinstance(questions, list):
        raise ValueError(f"Expected questions list, got: {parsed!r}")
    out = [str(q).strip() for q in questions if str(q).strip()]
    if len(out) < num_questions:
        logger.debug("Model returned %d questions (requested %d)", len(out), num_questions)
        return out
    logger.debug("Parsed %d synthetic question(s)", len(out[:num_questions]))
    return out[:num_questions]


def _load_points(path: Path) -> list[dict[str, Any]]:
    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON array of points")
    return data


def _is_transient_error(exc: Exception) -> bool:
    if isinstance(exc, (httpx.TimeoutException, httpx.ConnectError)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        return code == 429 or 500 <= code <= 599
    return False


def _retry_delay_seconds(*, attempt: int, base_delay: float) -> float:
    # Exponential backoff with low jitter to avoid synchronized retries.
    return max(0.0, base_delay) * (2 ** (attempt - 1)) + random.uniform(0.0, 0.1)


def _default_failed_report_path(*, out_dir: Path | None, data_dir: Path) -> Path:
    target_dir = out_dir if out_dir is not None else data_dir
    ts = datetime.now(_EST).strftime("%Y%m%dT%H%M%S_EST")
    reports_dir = target_dir / "reports"
    return reports_dir / f"synthetic_questions_failed_{ts}.json"


def enrich_point_payload(payload: dict[str, Any], *, questions: list[str]) -> None:
    from prepare_payloads import _build_embed_text, _token_count

    section = str(payload.get("section") or "ROOT")
    text = str(payload.get("text") or "").strip()
    qs = [str(q).strip() for q in questions if str(q).strip()]
    embed_text, used_q, trimmed_q = _build_embed_text(section, text, qs)
    payload["synthetic_questions"] = qs
    payload["embed_text"] = embed_text
    payload["embed_token_count"] = _token_count(embed_text)
    payload["synthetic_questions_used"] = used_q
    payload["synthetic_questions_trimmed"] = trimmed_q


async def _enrich_row(
    row: dict[str, Any],
    *,
    num_questions: int,
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    api_key: str | None,
    use_json_object: bool,
    skip_existing: bool,
    sem: asyncio.Semaphore,
    source_file: Path,
    retry_max_attempts: int,
    retry_base_delay: float,
) -> tuple[bool, bool, dict[str, Any] | None, tuple[str, float] | None]:
    payload = row.get("payload")
    if not isinstance(payload, dict):
        logger.warning("Skipping row without dict payload")
        return False, False, None, None
    section = str(payload.get("section") or "ROOT")
    text = str(payload.get("text") or "").strip()
    if not text:
        return False, False, None, None
    existing = payload.get("synthetic_questions", [])
    if (
        skip_existing
        and isinstance(existing, list)
        and len([q for q in existing if str(q).strip()]) >= num_questions
    ):
        return False, False, None, None

    chunk_id = str(payload.get("chunk_id") or "?")
    attempts = max(1, retry_max_attempts)
    last_error_message = ""

    for attempt in range(1, attempts + 1):
        try:
            call_started = time.perf_counter()
            async with sem:
                qs = await generate_questions_for_chunk(
                    client=client,
                    base_url=base_url,
                    model=model,
                    api_key=api_key,
                    section=section,
                    text=text,
                    num_questions=num_questions,
                    use_json_object=use_json_object,
                )
            section_elapsed_ms = (time.perf_counter() - call_started) * 1000
            enrich_point_payload(payload, questions=qs)
            return True, False, None, (section, section_elapsed_ms)
        except Exception as exc:
            last_error_message = f"{type(exc).__name__}: {exc}"
            transient = _is_transient_error(exc)
            should_retry = transient and attempt < attempts
            logger.warning(
                "Question generation failed for chunk_id=%s attempt=%d/%d retry=%s: %s",
                chunk_id,
                attempt,
                attempts,
                should_retry,
                exc,
                exc_info=logger.isEnabledFor(logging.DEBUG),
            )
            if should_retry:
                delay = _retry_delay_seconds(attempt=attempt, base_delay=retry_base_delay)
                await asyncio.sleep(delay)
                continue
            break

    enrich_point_payload(payload, questions=[])
    failed_item = {
        "source_file": str(source_file),
        "chunk_id": chunk_id,
        "section": section,
        "attempts": attempts,
        "error": last_error_message,
    }
    return True, True, failed_item, None


async def enrich_points_file(
    path: Path,
    *,
    num_questions: int,
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    api_key: str | None,
    use_json_object: bool,
    skip_existing: bool,
    max_concurrency: int,
    retry_max_attempts: int,
    retry_base_delay: float,
) -> tuple[list[dict[str, Any]], int, int, list[dict[str, Any]]]:
    """
    Load points JSON, fill synthetic_questions per payload, refresh embed_text.
    Returns (points, updated_count, question_gen_failure_count, failed_items).
    """
    file_started = time.perf_counter()
    points = _load_points(path)
    sem = asyncio.Semaphore(max(1, max_concurrency))
    tasks: list[asyncio.Task[tuple[bool, bool, dict[str, Any] | None, tuple[str, float] | None]]] = []
    for row in points:
        if not isinstance(row, dict):
            continue
        tasks.append(
            asyncio.create_task(
                _enrich_row(
                    row,
                    num_questions=num_questions,
                    client=client,
                    base_url=base_url,
                    model=model,
                    api_key=api_key,
                    use_json_object=use_json_object,
                    skip_existing=skip_existing,
                    sem=sem,
                    source_file=path,
                    retry_max_attempts=retry_max_attempts,
                    retry_base_delay=retry_base_delay,
                )
            )
        )

    results = await asyncio.gather(*tasks) if tasks else []
    updated = sum(1 for did_update, _, _, _ in results if did_update)
    failed = sum(1 for _, did_fail, _, _ in results if did_fail)
    failed_items = [item for _, _, item, _ in results if item is not None]
    section_latency_totals: dict[str, float] = {}
    for _, _, _, section_latency in results:
        if section_latency is None:
            continue
        section_name, latency_ms = section_latency
        section_latency_totals[section_name] = section_latency_totals.get(section_name, 0.0) + latency_ms
    file_elapsed_ms = (time.perf_counter() - file_started) * 1000
    logger.info(
        "enrich_points_file latency_ms=%.1f path=%s updated=%d failed=%d",
        file_elapsed_ms,
        path,
        updated,
        failed,
    )
    if section_latency_totals:
        for section_name, total_ms in sorted(
            section_latency_totals.items(),
            key=lambda item: item[1],
            reverse=True,
        ):
            logger.info(
                "section_total_latency_ms=%.1f path=%s section=%r",
                total_ms,
                path,
                section_name,
            )
    return points, updated, failed, failed_items


def _configure_logging(*, verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def _parse_args_points() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Read points_*.json (Qdrant-ready payloads), call inference to add "
            "synthetic_questions, and refresh embed_text / embed_token_count."
        )
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing points JSON (default: data).",
    )
    p.add_argument(
        "--pattern",
        default="points_*.json",
        help="Glob under --data-dir (default: points_*.json).",
    )
    p.add_argument(
        "--questions-per-chunk",
        type=int,
        default=3,
        metavar="N",
        help="Number of synthetic questions per point (default: 3).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="If set, write enriched files here; otherwise overwrite inputs.",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip points that already have at least N non-empty questions.",
    )
    p.add_argument(
        "--chat-base-url",
        default=(os.getenv("CHAT_BASE_URL") or "").strip() or INFERENCE_BASE_URL,
        help="Chat API root (default: CHAT_BASE_URL or INFERENCE_BASE_URL).",
    )
    p.add_argument(
        "--chat-model",
        default=(os.getenv("CHAT_MODEL") or "").strip() or DEFAULT_CHAT_MODEL,
        help="Chat model id (default: CHAT_MODEL or built-in default).",
    )
    p.add_argument(
        "--chat-api-key",
        default=(os.getenv("CHAT_API_KEY") or "").strip(),
        help="Optional Bearer token (default: CHAT_API_KEY).",
    )
    p.add_argument(
        "--no-json-object-mode",
        action="store_true",
        help="Do not send response_format json_object.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="No inference and no writes; log matched files and text payload counts.",
    )
    p.add_argument(
        "--max-concurrency",
        type=int,
        default=40,
        help="Maximum concurrent inference requests (default: 40).",
    )
    p.add_argument(
        "--retry-max-attempts",
        type=int,
        default=3,
        help="Maximum attempts for transient failures (default: 3).",
    )
    p.add_argument(
        "--retry-base-delay",
        type=float,
        default=0.5,
        help="Base delay in seconds for retry backoff (default: 0.5).",
    )
    p.add_argument(
        "--failed-report-path",
        type=Path,
        default=None,
        help="Optional path for failed-items report JSON. Defaults under output/data dir.",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Debug logging.",
    )
    return p.parse_args()


async def main_points_async() -> None:
    started = time.perf_counter()
    load_dotenv(_ROOT_DIR / ".env")
    load_dotenv()
    args = _parse_args_points()
    _configure_logging(verbose=args.verbose)

    data_dir: Path = args.data_dir
    if not data_dir.is_dir():
        raise SystemExit(f"--data-dir is not a directory: {data_dir}")

    paths = sorted(data_dir.glob(args.pattern))
    if not paths:
        raise SystemExit(f"No files matched {args.pattern!r} under {data_dir}")

    n = args.questions_per_chunk
    if n < 1:
        raise SystemExit("--questions-per-chunk must be >= 1")
    if args.max_concurrency < 1:
        raise SystemExit("--max-concurrency must be >= 1")
    if args.retry_max_attempts < 1:
        raise SystemExit("--retry-max-attempts must be >= 1")
    if args.retry_base_delay < 0:
        raise SystemExit("--retry-base-delay must be >= 0")

    base_url = normalize_chat_base_url(args.chat_base_url)
    api_key = args.chat_api_key.strip() or None
    use_json = not args.no_json_object_mode
    out_dir: Path | None = args.output_dir

    logger.info(
        "Enriching points: dir=%s pattern=%r files=%d questions_per_chunk=%d dry_run=%s",
        data_dir,
        args.pattern,
        len(paths),
        n,
        args.dry_run,
    )

    total_updated = 0
    total_failed = 0
    failed_items_all: list[dict[str, Any]] = []
    if args.dry_run:
        for path in paths:
            data = _load_points(path)
            dest = (out_dir / path.name) if out_dir is not None else path
            n_points = sum(
                1
                for row in data
                if isinstance(row, dict)
                and isinstance(row.get("payload"), dict)
                and str((row.get("payload") or {}).get("text") or "").strip()
            )
            logger.info("Dry-run: would enrich %s -> %s (%d text payloads)", path, dest, n_points)
        logger.info("Dry-run complete: files=%d (no API calls, no writes)", len(paths))
        return

    timeout = httpx.Timeout(120.0, connect=15.0)
    limits = httpx.Limits(max_connections=max(20, args.max_concurrency * 2))
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        for path in paths:
            data, updated, failed, failed_items = await enrich_points_file(
                path,
                num_questions=n,
                client=client,
                base_url=base_url,
                model=args.chat_model,
                api_key=api_key,
                use_json_object=use_json,
                skip_existing=args.skip_existing,
                max_concurrency=args.max_concurrency,
                retry_max_attempts=args.retry_max_attempts,
                retry_base_delay=args.retry_base_delay,
            )
            total_updated += updated
            total_failed += failed
            failed_items_all.extend(failed_items)
            dest = (out_dir / path.name) if out_dir is not None else path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(
                json.dumps(data, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            logger.info("Wrote %s (updated_payloads=%d gen_failures=%d)", dest, updated, failed)

    failed_report_path = args.failed_report_path or _default_failed_report_path(
        out_dir=out_dir,
        data_dir=data_dir,
    )
    failed_report_path.parent.mkdir(parents=True, exist_ok=True)
    report_payload = {
        "generated_at": datetime.now(_EST).isoformat(),
        "summary": {
            "files": len(paths),
            "updated_payloads": total_updated,
            "gen_failures": total_failed,
        },
        "run_args": {
            "data_dir": str(data_dir),
            "pattern": args.pattern,
            "questions_per_chunk": n,
            "max_concurrency": args.max_concurrency,
            "retry_max_attempts": args.retry_max_attempts,
            "retry_base_delay": args.retry_base_delay,
        },
        "failed_items": failed_items_all,
    }
    failed_report_path.write_text(
        json.dumps(report_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.info("Wrote failed-items report: %s (count=%d)", failed_report_path, len(failed_items_all))

    logger.info(
        "Done: files=%d total_updated_payloads=%d total_gen_failures=%d",
        len(paths),
        total_updated,
        total_failed,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000
    logger.info("synthetic_questions total_latency_ms=%.1f", elapsed_ms)


def main_points() -> None:
    asyncio.run(main_points_async())


if __name__ == "__main__":
    main_points()
