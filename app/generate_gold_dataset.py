"""Generate a gold QA dataset from points_*.json files."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)
_DEFAULT_CHAT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
_DEFAULT_INFERENCE_BASE_URL = "http://192.168.86.179:30180"


def _iter_points_files(data_roots: list[Path], pattern: str) -> list[tuple[str, Path]]:
    """Return sorted (env, file_path) matches for all roots."""
    matches: list[tuple[str, Path]] = []
    for root in data_roots:
        if not root.exists():
            logger.warning("Data root does not exist, skipping: %s", root)
            continue
        env = root.name.removeprefix("data_")
        for path in sorted(root.glob(pattern)):
            if path.is_file():
                matches.append((env, path))
    matches.sort(key=lambda item: (item[0], str(item[1])))
    return matches


def _load_points(path: Path) -> list[dict[str, Any]]:
    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON array")
    return data


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip())


def _sanitize_must_contain(items: list[str]) -> list[str]:
    out: list[str] = []
    for item in items:
        s = _normalize_text(item)
        if not s:
            continue
        s = re.sub(r"^[Qq]\s*:\s*", "", s)
        s = re.sub(r"^[Aa]\s*:\s*", "", s)
        s = s.strip(" -:;,.")
        if not s:
            continue
        # Split overlong entries so must_contain stays semantic, not formatting-bound.
        parts = re.split(r"\s+and\s+|;|,|\n", s)
        for part in parts:
            p = _normalize_text(part).strip(" -:;,.")
            if not p:
                continue
            word_count = len(p.split())
            if 1 <= word_count <= 8 and p not in out:
                out.append(p)
    return out[:5]


def _extract_keywords_fallback(answer: str, *, limit: int = 4) -> list[str]:
    """Fallback must_contain extraction when LLM is disabled/unavailable."""
    text = _normalize_text(answer)
    if not text:
        return []
    parts = re.split(r"[.;,\n]|(?:\bQ:\b)|(?:\bA:\b)", text)
    out: list[str] = []
    for part in parts:
        token = _normalize_text(part)
        if len(token) < 4:
            continue
        if token.lower().startswith("what ") or token.endswith("?"):
            continue
        if token not in out:
            out.append(token)
        if len(out) >= limit:
            break
    if out:
        return _sanitize_must_contain(out)[:limit]
    words = [w for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9+\-_/\.]*", text) if len(w) >= 4]
    return _sanitize_must_contain(words)[:limit]


def _fallback_keywords_from_question(question: str, *, limit: int = 3) -> list[str]:
    tokens = [w.lower() for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9+\-_/\.]*", question)]
    stop = {"what", "which", "where", "when", "why", "how", "does", "is", "are", "the", "can", "for"}
    out: list[str] = []
    for t in tokens:
        if len(t) < 3 or t in stop:
            continue
        if t not in out:
            out.append(t)
        if len(out) >= limit:
            break
    return out


def _llm_must_contain_messages(answer: str) -> tuple[str, str]:
    system = (
        "Extract key factual fragments required to validate an answer in a RAG evaluator. "
        "Return only JSON as {\"must_contain\": [\"...\"]}. Keep items short and literal."
    )
    user = (
        "Answer text:\n"
        f"{answer}\n\n"
        "Rules:\n"
        "- Return 2 to 5 concise fragments\n"
        "- Prefer entities, values, constraints, statuses\n"
        "- No full sentences\n"
    )
    return system, user


def _parse_must_contain_llm_response(content: str) -> list[str]:
    parsed = json.loads(content)
    result = parsed.get("must_contain")
    if not isinstance(result, list):
        return []
    return _sanitize_must_contain([str(item).strip() for item in result if str(item).strip()])


async def _extract_keywords_llm_async(
    answer: str,
    *,
    base_url: str,
    model: str,
    api_key: str | None,
    client: Any,
) -> list[str]:
    # Optional import so non-LLM flow does not require inference deps.
    from client_inference import async_chat_completions, normalize_chat_base_url

    system, user = _llm_must_contain_messages(answer)
    data = await async_chat_completions(
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        base_url=normalize_chat_base_url(base_url),
        model=model,
        api_key=api_key,
        max_tokens=180,
        temperature=0.0,
        response_format={"type": "json_object"},
        client=client,
    )
    content = str(data["choices"][0]["message"]["content"]).strip()
    return _parse_must_contain_llm_response(content)


def _must_contain_llm_dedupe_key(env: str, text: str) -> tuple[str, str]:
    fp = hashlib.sha256(_normalize_text(text).encode("utf-8")).hexdigest()[:24]
    return env, fp


async def _enrich_rows_must_contain_llm(
    rows: list[dict[str, Any]],
    *,
    base_url: str,
    model: str,
    api_key: str | None,
    concurrency: int,
) -> None:
    """Replace heuristic must_contain using concurrent async LLM calls (single_hop rows)."""
    import httpx

    targets: dict[tuple[str, str], str] = {}
    for row in rows:
        case = str(row.get("case_type", ""))
        if case != "single_hop":
            continue
        env = str(row.get("env", ""))
        text = str(row.get("text") or row.get("answer") or "").strip()
        if not text:
            continue
        key = _must_contain_llm_dedupe_key(env, text)
        targets.setdefault(key, text)

    if not targets:
        return

    sem = asyncio.Semaphore(max(1, concurrency))
    keys_in_order = list(targets.keys())

    async with httpx.AsyncClient() as client:

        async def _one(text: str) -> list[str]:
            async with sem:
                try:
                    return await _extract_keywords_llm_async(
                        text,
                        base_url=base_url,
                        model=model,
                        api_key=api_key,
                        client=client,
                    )
                except Exception as exc:
                    logger.warning("must_contain LLM extraction failed: %s", exc)
                    return []

        results = await asyncio.gather(*[_one(targets[k]) for k in keys_in_order])
        llm_by_key = dict(zip(keys_in_order, results, strict=True))

    for row in rows:
        case = str(row.get("case_type", ""))
        if case != "single_hop":
            continue
        env = str(row.get("env", ""))
        text = str(row.get("text") or row.get("answer") or "").strip()
        if not text:
            continue
        key = _must_contain_llm_dedupe_key(env, text)
        llm_out = llm_by_key.get(key) or []
        if llm_out:
            row["must_contain"] = llm_out
        elif not row.get("must_contain"):
            fb = _fallback_keywords_from_question(str(row.get("question", ""))) or ["fact_required"]
            row["must_contain"] = fb


def _generate_noisy_queries(question: str) -> list[str]:
    q = question.strip()
    if not q:
        return []
    base = q.rstrip("?")
    compact = re.sub(r"[^\w\s]", "", base).lower().strip()
    toks = compact.split()
    short = " ".join(toks[:4]).strip()
    variants = [
        compact,
        f"{compact}?",
        short,
        short.replace("what is ", "").replace("what are ", "").strip(),
        compact.replace("work authorization", "work auth"),
        compact.replace("sponsorship", "sponsor"),
    ]
    out: list[str] = []
    for v in variants:
        v = _normalize_text(v)
        if v and v not in out and v != q:
            out.append(v)
    return out


def _pick_canonical_question(questions: list[str]) -> str:
    unique = []
    for q in questions:
        qq = _normalize_text(q)
        if qq and qq not in unique:
            unique.append(qq)
    if not unique:
        return ""
    # Prefer a concise clean query.
    unique.sort(key=lambda q: (len(q.split()), len(q)))
    return unique[0]


def _build_single_hop_rows(
    *,
    env: str,
    source_file: Path,
    points: list[dict[str, Any]],
    include_empty_questions: bool,
    enable_noisy_queries: bool,
    max_paraphrases_per_fact: int,
    must_contain_cache: dict[tuple[str, str], list[str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    max_count = max(1, max_paraphrases_per_fact)
    for point in sorted(points, key=lambda p: str(p.get("id", ""))):
        payload = point.get("payload") if isinstance(point, dict) else None
        if not isinstance(payload, dict):
            continue

        point_id = str(point.get("id", "")).strip()
        answer_text = str(payload.get("text", ""))
        source = str(payload.get("source", ""))
        cache_key = (env, point_id)
        if cache_key not in must_contain_cache:
            must_contain_cache[cache_key] = _extract_keywords_fallback(answer_text)

        questions_raw = payload.get("synthetic_questions")
        questions: list[str] = []
        if isinstance(questions_raw, list):
            questions = [str(q).strip() for q in questions_raw if str(q).strip()]
        if not questions and include_empty_questions:
            questions = [""]
        if not questions:
            continue

        canonical = _pick_canonical_question(questions)
        if not canonical:
            continue
        all_qs: list[tuple[str, str]] = [(canonical, "clean")]
        if enable_noisy_queries and max_count > 1:
            noisy_variants = _generate_noisy_queries(canonical)
            if noisy_variants:
                all_qs.append((noisy_variants[0], "noisy"))
        all_qs = all_qs[:max_count]

        for idx, (question, query_type) in enumerate(all_qs):
            must_contain = must_contain_cache[cache_key]
            if not must_contain:
                must_contain = _fallback_keywords_from_question(question) or ["fact_required"]
            rows.append(
                {
                    "env": env,
                    "source_file": str(source_file),
                    "id": point_id,
                    "question": question,
                    "answer": answer_text,
                    "must_contain": must_contain,
                    "source": source,
                    "doc_type": str(payload.get("doc_type", "")),
                    "section": str(payload.get("section", "")),
                    "chunk_id": str(payload.get("chunk_id", "")),
                    "text": answer_text,
                    "case_type": "single_hop",
                    "required_sources": [],
                    "expected_behavior": "answer",
                    "query_type": query_type,
                    "eval_bucket": "easy_single_hop" if query_type == "clean" else "paraphrase",
                    "_question_index": idx,
                }
            )
    return rows


def _dedup_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    seen: set[tuple[str, str, str]] = set()
    out: list[dict[str, Any]] = []
    dropped = 0
    for row in rows:
        key = (str(row["env"]), str(row["id"]), str(row["question"]))
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        out.append(row)
    return out, dropped


def _write_jsonl(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            out = dict(row)
            out.pop("_question_index", None)
            fh.write(json.dumps(out, ensure_ascii=False) + "\n")


def _write_split_jsonl(rows: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    bucket_map = {
        "easy_single_hop": output_dir / "easy_single_hop.jsonl",
        "paraphrase": output_dir / "paraphrase.jsonl",
    }
    grouped: dict[str, list[dict[str, Any]]] = {k: [] for k in bucket_map}
    for row in rows:
        bucket = str(row.get("eval_bucket", "easy_single_hop"))
        if bucket not in grouped:
            grouped["easy_single_hop"].append(row)
        else:
            grouped[bucket].append(row)
    for bucket, target in bucket_map.items():
        _write_jsonl(grouped[bucket], target)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate gold dataset JSONL from points_*.json (consolidated and/or split eval files)."
    )
    parser.add_argument(
        "--data-roots",
        nargs="+",
        default=["data_dev", "data_qa", "data_prod"],
        help="Base data roots to scan (default: data_dev data_qa data_prod).",
    )
    parser.add_argument(
        "--glob",
        default="**/processed/points_*.json",
        help="Glob pattern under each data root.",
    )
    parser.add_argument(
        "--output",
        default="gold_dataset.jsonl",
        help="Consolidated output JSONL path (skipped if --skip-consolidated-output).",
    )
    parser.add_argument(
        "--skip-consolidated-output",
        action="store_true",
        help="Do not write the consolidated JSONL; requires --split-output-dir (split files only).",
    )
    parser.add_argument(
        "--include-empty-questions",
        action="store_true",
        help="Include one row with empty question if synthetic_questions is missing/empty.",
    )
    parser.add_argument(
        "--no-dedup",
        action="store_true",
        help="Disable dedup on (env, id, question).",
    )
    parser.add_argument(
        "--enable-must-contain-llm",
        action="store_true",
        help="Use LLM extraction for must_contain (fallback heuristics on errors).",
    )
    parser.add_argument(
        "--max-paraphrases-per-fact",
        type=int,
        default=3,
        help="Max queries per fact (canonical + noisy variants).",
    )
    parser.add_argument(
        "--enable-noisy-queries",
        action="store_true",
        help="Generate messy query variants from canonical questions.",
    )
    parser.add_argument(
        "--chat-base-url",
        default=(os.getenv("CHAT_BASE_URL") or _DEFAULT_INFERENCE_BASE_URL).strip(),
        help="Chat base URL for must_contain LLM extraction.",
    )
    parser.add_argument(
        "--chat-model",
        default=(os.getenv("CHAT_MODEL") or _DEFAULT_CHAT_MODEL).strip(),
        help="Chat model for must_contain LLM extraction.",
    )
    parser.add_argument(
        "--chat-api-key",
        default=(os.getenv("CHAT_API_KEY") or "").strip(),
        help="Optional chat API key for must_contain extraction.",
    )
    parser.add_argument(
        "--llm-concurrency",
        type=int,
        default=40,
        help="Max concurrent async LLM calls for must_contain extraction (default: 40).",
    )
    parser.add_argument(
        "--split-output-dir",
        default="",
        help="Directory for split eval files (defaults to output file directory).",
    )
    return parser.parse_args()


def _validate_rows(rows: list[dict[str, Any]]) -> dict[str, int]:
    invalid_single_hop = 0
    for row in rows:
        case_type = str(row.get("case_type", "single_hop"))
        if case_type == "single_hop" and not row.get("must_contain"):
            invalid_single_hop += 1
    return {"invalid_single_hop": invalid_single_hop}


def main() -> None:
    args = parse_args()
    data_roots = [Path(p) for p in args.data_roots]
    output_path = Path(args.output)
    if args.skip_consolidated_output and not str(args.split_output_dir).strip():
        raise SystemExit(
            "--split-output-dir is required when using --skip-consolidated-output."
        )
    split_output_dir = Path(args.split_output_dir) if args.split_output_dir else output_path.parent
    max_paraphrases = max(1, int(args.max_paraphrases_per_fact))

    matches = _iter_points_files(data_roots, args.glob)
    if not matches:
        raise SystemExit("No points files found. Check --data-roots/--glob.")

    all_rows: list[dict[str, Any]] = []
    files_scanned = 0
    points_processed = 0
    must_contain_cache: dict[tuple[str, str], list[str]] = {}

    for env, path in matches:
        points = _load_points(path)
        files_scanned += 1
        points_processed += len(points)
        rows = _build_single_hop_rows(
            env=env,
            source_file=path,
            points=points,
            include_empty_questions=bool(args.include_empty_questions),
            enable_noisy_queries=bool(args.enable_noisy_queries),
            max_paraphrases_per_fact=max_paraphrases,
            must_contain_cache=must_contain_cache,
        )
        all_rows.extend(rows)

    if args.enable_must_contain_llm:
        asyncio.run(
            _enrich_rows_must_contain_llm(
                all_rows,
                base_url=args.chat_base_url,
                model=args.chat_model,
                api_key=(args.chat_api_key or None),
                concurrency=max(1, int(args.llm_concurrency)),
            )
        )

    all_rows.sort(
        key=lambda row: (
            str(row.get("env", "")),
            str(row.get("source_file", "")),
            str(row.get("id", "")),
            int(row.get("_question_index", 0)),
        )
    )

    duplicates_dropped = 0
    if not args.no_dedup:
        all_rows, duplicates_dropped = _dedup_rows(all_rows)

    checks = _validate_rows(all_rows)
    if not args.skip_consolidated_output:
        _write_jsonl(all_rows, output_path)
    _write_split_jsonl(all_rows, split_output_dir)

    consolidated = "skipped" if args.skip_consolidated_output else str(output_path)
    logger.info(
        "Gold dataset written: consolidated=%s split_dir=%s files_scanned=%d points_processed=%d rows_written=%d duplicates_dropped=%d invalid_single_hop=%d",
        consolidated,
        split_output_dir,
        files_scanned,
        points_processed,
        len(all_rows),
        duplicates_dropped,
        checks["invalid_single_hop"],
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    main()
