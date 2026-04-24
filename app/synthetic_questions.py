"""Generate synthetic RAG questions per chunk via chat completions."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

from inference_client import (
    DEFAULT_CHAT_MODEL,
    INFERENCE_BASE_URL,
    chat_completions,
    normalize_chat_base_url,
)

_APP_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _APP_DIR.parent

logger = logging.getLogger(__name__)


def _json_from_response(content: str) -> dict[str, Any]:
    content = content.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", content)
    if fence:
        content = fence.group(1).strip()
    return json.loads(content)


def generate_questions_for_chunk(
    *,
    client: httpx.Client | None,
    base_url: str,
    model: str,
    api_key: str | None,
    section: str,
    text: str,
    num_questions: int,
    use_json_object: bool,
) -> list[str]:
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
    data = chat_completions(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        base_url=normalize_chat_base_url(base_url),
        model=model,
        max_tokens=512,
        temperature=0.3,
        api_key=api_key if api_key else None,
        client=client,
        timeout=120.0,
        response_format={"type": "json_object"} if use_json_object else None,
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


def enrich_point_payload(payload: dict[str, Any], *, questions: list[str]) -> None:
    from prepare_points import _build_embed_text, _token_count

    section = str(payload.get("section") or "ROOT")
    text = str(payload.get("text") or "").strip()
    qs = [str(q).strip() for q in questions if str(q).strip()]
    embed_text, used_q, trimmed_q = _build_embed_text(section, text, qs)
    payload["synthetic_questions"] = qs
    payload["embed_text"] = embed_text
    payload["embed_token_count"] = _token_count(embed_text)
    payload["synthetic_questions_used"] = used_q
    payload["synthetic_questions_trimmed"] = trimmed_q


def enrich_points_file(
    path: Path,
    *,
    num_questions: int,
    client: httpx.Client,
    base_url: str,
    model: str,
    api_key: str | None,
    use_json_object: bool,
    skip_existing: bool,
) -> tuple[list[dict[str, Any]], int, int]:
    """
    Load points JSON, fill synthetic_questions per payload, refresh embed_text.
    Returns (points, updated_count, question_gen_failure_count).
    """
    points = _load_points(path)
    failed = 0
    updated = 0
    for row in points:
        if not isinstance(row, dict):
            continue
        payload = row.get("payload")
        if not isinstance(payload, dict):
            logger.warning("Skipping row without dict payload")
            continue
        section = str(payload.get("section") or "ROOT")
        text = str(payload.get("text") or "").strip()
        if not text:
            continue
        existing = payload.get("synthetic_questions", [])
        if (
            skip_existing
            and isinstance(existing, list)
            and len([q for q in existing if str(q).strip()]) >= num_questions
        ):
            continue
        try:
            qs = generate_questions_for_chunk(
                client=client,
                base_url=base_url,
                model=model,
                api_key=api_key,
                section=section,
                text=text,
                num_questions=num_questions,
                use_json_object=use_json_object,
            )
        except Exception as exc:
            failed += 1
            logger.warning(
                "Question generation failed for chunk_id=%s: %s",
                payload.get("chunk_id", "?"),
                exc,
                exc_info=logger.isEnabledFor(logging.DEBUG),
            )
            qs = []
        enrich_point_payload(payload, questions=qs)
        updated += 1
    return points, updated, failed


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
        "-v",
        "--verbose",
        action="store_true",
        help="Debug logging.",
    )
    return p.parse_args()


def main_points() -> None:
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

    with httpx.Client() as client:
        for path in paths:
            data, updated, failed = enrich_points_file(
                path,
                num_questions=n,
                client=client,
                base_url=base_url,
                model=args.chat_model,
                api_key=api_key,
                use_json_object=use_json,
                skip_existing=args.skip_existing,
            )
            total_updated += updated
            total_failed += failed
            dest = (out_dir / path.name) if out_dir is not None else path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(
                json.dumps(data, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            logger.info("Wrote %s (updated_payloads=%d gen_failures=%d)", dest, updated, failed)

    logger.info(
        "Done: files=%d total_updated_payloads=%d total_gen_failures=%d",
        len(paths),
        total_updated,
        total_failed,
    )


if __name__ == "__main__":
    main_points()
