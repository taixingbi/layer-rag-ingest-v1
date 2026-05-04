"""Gold dataset evaluation against the RAG `/v1/rag/query` API.

Reads JSONL rows (see `docs/gold-dataset.md`), posts each `question`, then scores:
- `must_contain`: each fragment must appear as a substring of the model `answer` (case-insensitive).
- `source` (single-hop): at least one citation `source` equals the gold row `source` when present.
- `required_sources` (multi-hop): every listed source appears in citation `source` values.
- `retrieval_hits` (optional): by default the client sends `include_retrieval_hits: true`. Match gold
  row UUID `id` to each hit's `chunk_id` within `retrieve` / `rerank` stages; per-row ranks and
  summary mean RR + Recall@k (see `--recall-at-k`). Use `--skip-retrieval-hits` for gateways without
  this field.

Example:

  python3 app/rag_gold_eval/run_eval.py \\
    --gold data_dev/gold_dataset/easy_single_hop.jsonl \\
    --rag-base-url http://192.168.86.179:30183 \\
    --collection-base taixing_knowledge \\
    --recall-at-k 5,10,40
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_RAG_BASE_URL = "http://192.168.86.179:30183"
_DEFAULT_COLLECTION_BASE = "taixing_knowledge"

_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\Z",
    re.IGNORECASE,
)


def _parse_recall_ks(raw: str) -> list[int]:
    out: list[int] = []
    for part in (raw or "").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            k = int(part)
        except ValueError:
            continue
        if k > 0 and k not in out:
            out.append(k)
    return sorted(out)


def _gold_chunk_id(row: dict[str, Any]) -> str | None:
    """Gold point id when it is a UUID (matches retrieval_hits chunk_id)."""
    rid = str(row.get("id") or "").strip()
    if not rid or not _UUID_RE.match(rid):
        return None
    return rid.lower()


def _hits_by_stage(retrieval_hits: Any) -> dict[str, list[str]]:
    """Group retrieval_hits by stage; ordered chunk_id lists by ascending rank."""
    if not isinstance(retrieval_hits, list):
        return {}
    by_stage: dict[str, list[tuple[int, str]]] = {}
    for h in retrieval_hits:
        if not isinstance(h, dict):
            continue
        st = str(h.get("stage") or "").strip()
        cid = str(h.get("chunk_id") or "").strip()
        if not st or not cid:
            continue
        try:
            rk = int(h["rank"])
        except (KeyError, TypeError, ValueError):
            continue
        by_stage.setdefault(st, []).append((rk, cid.lower()))
    out: dict[str, list[str]] = {}
    for st, pairs in by_stage.items():
        pairs.sort(key=lambda x: x[0])
        out[st] = [p[1] for p in pairs]
    return out


def _rank_of(chunk_ids: list[str], target: str) -> int | None:
    t = target.lower()
    for i, cid in enumerate(chunk_ids):
        if cid.lower() == t:
            return i + 1
    return None


def _reciprocal_rank(rank: int | None) -> float:
    if rank is None or rank <= 0:
        return 0.0
    return 1.0 / float(rank)


def _hit_at_k(rank: int | None, k: int) -> bool:
    return rank is not None and rank <= k


def _retrieval_row_fields(
    row: dict[str, Any],
    data: dict[str, Any],
    *,
    request_retrieval_hits: bool,
    recall_ks: list[int],
) -> dict[str, Any]:
    """Per-row retrieval metrics; retrieval_scored True only when eval ran on non-empty hits."""
    out: dict[str, Any] = {
        "retrieval_scored": False,
        "retrieval_eval_skipped": None,
        "gold_chunk_id": None,
        "rank_retrieve": None,
        "rank_rerank": None,
        "rr_retrieve": 0.0,
        "rr_rerank": 0.0,
    }
    if not request_retrieval_hits:
        out["retrieval_eval_skipped"] = "skip_retrieval_hits_flag"
        return out

    gold = _gold_chunk_id(row)
    out["gold_chunk_id"] = gold
    if not gold:
        out["retrieval_eval_skipped"] = "no_gold_uuid_id"
        return out

    raw_hits = data.get("retrieval_hits")
    if not isinstance(raw_hits, list):
        out["retrieval_eval_skipped"] = "no_retrieval_hits_in_response"
        return out
    if not raw_hits:
        out["retrieval_eval_skipped"] = "empty_retrieval_hits"
        return out

    by_stage = _hits_by_stage(raw_hits)
    retrieve_ids = by_stage.get("retrieve", [])
    rerank_ids = by_stage.get("rerank", [])

    rank_r = _rank_of(retrieve_ids, gold)
    rank_rr = _rank_of(rerank_ids, gold)
    out["rank_retrieve"] = rank_r
    out["rank_rerank"] = rank_rr
    out["rr_retrieve"] = _reciprocal_rank(rank_r)
    out["rr_rerank"] = _reciprocal_rank(rank_rr)
    out["retrieval_hit_count"] = len(raw_hits)
    out["retrieval_stages"] = sorted(by_stage.keys())

    for kk in recall_ks:
        out[f"hit_retrieve_at_{kk}"] = _hit_at_k(rank_r, kk)
        out[f"hit_rerank_at_{kk}"] = _hit_at_k(rank_rr, kk)

    out["retrieval_scored"] = True
    return out


def _normalize_answer(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip()).lower()


def _must_contain_hits(answer: str, fragments: list[str]) -> tuple[int, int, list[str]]:
    hay = _normalize_answer(answer)
    hits = 0
    missing: list[str] = []
    for frag in fragments:
        f = str(frag).strip()
        if not f:
            continue
        needle = _normalize_answer(f)
        if needle and needle in hay:
            hits += 1
        else:
            missing.append(f)
    total = len([f for f in fragments if str(f).strip()])
    return hits, total, missing


def _citation_sources(citations: Any) -> set[str]:
    out: set[str] = set()
    if not isinstance(citations, list):
        return out
    for c in citations:
        if isinstance(c, dict) and c.get("source"):
            out.add(str(c["source"]))
    return out


def _gold_source_hit(row: dict[str, Any], cite_sources: set[str]) -> bool | None:
    """None = skip check (no gold source)."""
    gold_src = str(row.get("source") or "").strip()
    if not gold_src or gold_src == "multi" or gold_src == "negative":
        return None
    return gold_src in cite_sources


def _required_sources_hit(row: dict[str, Any], cite_sources: set[str]) -> bool | None:
    req = row.get("required_sources")
    if not isinstance(req, list) or not req:
        return None
    needed = {str(s).strip() for s in req if str(s).strip()}
    if not needed:
        return None
    return needed.issubset(cite_sources)


def _iter_jsonl_paths(paths: list[str]) -> list[Path]:
    out: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_file():
            out.append(p)
        elif p.is_dir():
            for child in sorted(p.glob("*.jsonl")):
                if child.is_file():
                    out.append(child)
        else:
            logger.warning("Gold path not found, skipping: %s", raw)
    return out


def _load_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for line_no, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Skip bad JSON %s:%s: %s", path, line_no, exc)
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


async def _rag_query(
    client: Any,
    *,
    base_url: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/v1/rag/query"
    r = await client.post(url, json=payload, timeout=120.0)
    r.raise_for_status()
    return r.json()


async def _evaluate_all(
    rows: list[dict[str, Any]],
    *,
    rag_base_url: str,
    collection_base: str,
    k: int,
    k_max: int,
    concurrency: int,
    limit: int | None,
    request_retrieval_hits: bool,
    recall_ks: list[int],
) -> list[dict[str, Any]]:
    import httpx

    sem = asyncio.Semaphore(max(1, concurrency))
    work = rows if limit is None else rows[: max(0, limit)]

    async with httpx.AsyncClient() as client:

        async def _one(idx: int, row: dict[str, Any]) -> dict[str, Any]:
            question = str(row.get("question") or "").strip()
            out: dict[str, Any] = {
                "index": idx,
                "env": row.get("env"),
                "id": row.get("id"),
                "eval_bucket": row.get("eval_bucket"),
                "case_type": row.get("case_type"),
                "expected_behavior": row.get("expected_behavior"),
                "question": question,
                "ok": False,
                "http_status": None,
                "error": None,
                "must_contain_hits": 0,
                "must_contain_total": 0,
                "must_contain_missing": [],
                "must_contain_pass": False,
                "gold_source_hit": None,
                "required_sources_pass": None,
                "latency_ms_total": None,
            }
            if not question:
                out["error"] = "empty_question"
                return out

            payload: dict[str, Any] = {
                "question": question,
                "collection_base": collection_base,
                "request_id": f"eva-{uuid.uuid4().hex[:12]}",
                "session_id": f"eva-ses-{uuid.uuid4().hex[:10]}",
                "k": k,
                "k_max": k_max,
            }
            if request_retrieval_hits:
                payload["include_retrieval_hits"] = True
            try:
                async with sem:
                    data = await _rag_query(client, base_url=rag_base_url, payload=payload)
            except Exception as exc:
                out["error"] = str(exc)
                return out

            out["ok"] = True
            answer = str(data.get("answer") or "")
            citations = data.get("citations")
            cite_sources = _citation_sources(citations)

            must_list: list[str] = []
            raw_must = row.get("must_contain")
            if isinstance(raw_must, list):
                must_list = [str(x) for x in raw_must]

            hits, total, missing = _must_contain_hits(answer, must_list)
            out["must_contain_hits"] = hits
            out["must_contain_total"] = total
            out["must_contain_missing"] = missing
            out["must_contain_pass"] = total == 0 or (hits == total and total > 0)

            out["gold_source_hit"] = _gold_source_hit(row, cite_sources)
            out["required_sources_pass"] = _required_sources_hit(row, cite_sources)

            lat = data.get("latency_ms")
            if isinstance(lat, dict) and lat.get("total") is not None:
                out["latency_ms_total"] = lat.get("total")

            out["rag_answer_preview"] = answer[:400]
            out["citation_sources"] = sorted(cite_sources)

            retr = _retrieval_row_fields(row, data, request_retrieval_hits=request_retrieval_hits, recall_ks=recall_ks)
            out.update(retr)
            return out

        tasks = [_one(i, row) for i, row in enumerate(work)]
        return await asyncio.gather(*tasks)


def _summarize(results: list[dict[str, Any]], *, recall_ks: list[int]) -> dict[str, Any]:
    n = len(results)
    ok = sum(1 for r in results if r.get("ok"))
    must_pass = sum(1 for r in results if r.get("must_contain_pass"))
    must_scored = sum(1 for r in results if (r.get("must_contain_total") or 0) > 0)

    src_vals = [r.get("gold_source_hit") for r in results if r.get("gold_source_hit") is not None]
    src_pass = sum(1 for v in src_vals if v is True)

    req_vals = [r.get("required_sources_pass") for r in results if r.get("required_sources_pass") is not None]
    req_pass = sum(1 for v in req_vals if v is True)

    errors = [r for r in results if r.get("error")]

    out: dict[str, Any] = {
        "rows": n,
        "rag_calls_ok": ok,
        "rag_calls_failed": n - ok,
        "must_contain_pass": must_pass,
        "must_contain_scored_rows": must_scored,
        "gold_source_checked": len(src_vals),
        "gold_source_pass": src_pass,
        "required_sources_checked": len(req_vals),
        "required_sources_pass": req_pass,
        "errors_sample": errors[:5],
    }

    scored = [r for r in results if r.get("ok") and r.get("retrieval_scored")]
    ns = len(scored)
    out["retrieval_scored_rows"] = ns
    if ns > 0:
        out["mean_rr_retrieve"] = sum(float(r.get("rr_retrieve") or 0) for r in scored) / ns
        out["mean_rr_rerank"] = sum(float(r.get("rr_rerank") or 0) for r in scored) / ns
        found_r = [r for r in scored if r.get("rank_retrieve") is not None]
        found_rr = [r for r in scored if r.get("rank_rerank") is not None]
        out["retrieval_found_retrieve"] = len(found_r)
        out["retrieval_found_rerank"] = len(found_rr)
        if found_r:
            out["mean_rr_retrieve_when_found"] = sum(1.0 / int(r["rank_retrieve"]) for r in found_r) / len(found_r)
        else:
            out["mean_rr_retrieve_when_found"] = 0.0
        if found_rr:
            out["mean_rr_rerank_when_found"] = sum(1.0 / int(r["rank_rerank"]) for r in found_rr) / len(found_rr)
        else:
            out["mean_rr_rerank_when_found"] = 0.0
        for kk in recall_ks:
            hk = f"hit_retrieve_at_{kk}"
            out[f"recall_at_{kk}_retrieve"] = sum(1 for r in scored if r.get(hk) is True) / ns
            hk2 = f"hit_rerank_at_{kk}"
            out[f"recall_at_{kk}_rerank"] = sum(1 for r in scored if r.get(hk2) is True) / ns
    else:
        out["mean_rr_retrieve"] = 0.0
        out["mean_rr_rerank"] = 0.0
        out["retrieval_found_retrieve"] = 0
        out["retrieval_found_rerank"] = 0
        out["mean_rr_retrieve_when_found"] = 0.0
        out["mean_rr_rerank_when_found"] = 0.0
        for kk in recall_ks:
            out[f"recall_at_{kk}_retrieve"] = 0.0
            out[f"recall_at_{kk}_rerank"] = 0.0

    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate gold JSONL against RAG /v1/rag/query.")
    p.add_argument(
        "--gold",
        nargs="+",
        required=True,
        help="Gold JSONL file(s) or directories containing *.jsonl.",
    )
    p.add_argument(
        "--rag-base-url",
        default=(os.getenv("RAG_BASE_URL") or _DEFAULT_RAG_BASE_URL).strip(),
        help="RAG gateway base URL (no /v1 suffix). Default: RAG_BASE_URL or %(default)s",
    )
    p.add_argument(
        "--collection-base",
        default=(os.getenv("RAG_COLLECTION_BASE") or _DEFAULT_COLLECTION_BASE).strip(),
        help="collection_base field for the API. Default: RAG_COLLECTION_BASE or %(default)s",
    )
    p.add_argument("--k", type=int, default=5, help="k for retrieve (default: %(default)s).")
    p.add_argument("--k-max", type=int, default=40, help="k_max (default: %(default)s).")
    p.add_argument(
        "--concurrency",
        type=int,
        default=20,
        help="Max concurrent async RAG requests (default: %(default)s).",
    )
    p.add_argument("--limit", type=int, default=0, help="Max rows to evaluate (0 = all).")
    p.add_argument(
        "--skip-retrieval-hits",
        action="store_true",
        help="Do not send include_retrieval_hits; skip retrieval rank / Recall@k metrics.",
    )
    p.add_argument(
        "--recall-at-k",
        default="5,10,40",
        help="Comma-separated k values for Recall@k on retrieve and rerank lists (default: %(default)s).",
    )
    p.add_argument(
        "--report-json",
        default="",
        help="Write per-row results JSON array to this path.",
    )
    p.add_argument(
        "--summary-json",
        default="",
        help="Write summary object to this path.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    paths = _iter_jsonl_paths(list(args.gold))
    if not paths:
        raise SystemExit("No gold JSONL files resolved from --gold.")

    rows = _load_rows(paths)
    if not rows:
        raise SystemExit("No rows loaded from gold files.")

    limit = None if int(args.limit) <= 0 else int(args.limit)

    recall_ks = _parse_recall_ks(str(args.recall_at_k))
    if not recall_ks:
        recall_ks = [5, 10, 40]

    request_retrieval_hits = not bool(args.skip_retrieval_hits)

    results = asyncio.run(
        _evaluate_all(
            rows,
            rag_base_url=args.rag_base_url,
            collection_base=args.collection_base,
            k=int(args.k),
            k_max=int(args.k_max),
            concurrency=int(args.concurrency),
            limit=limit,
            request_retrieval_hits=request_retrieval_hits,
            recall_ks=recall_ks,
        )
    )
    summary = _summarize(results, recall_ks=recall_ks)

    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.report_json:
        out_path = Path(args.report_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        logger.info("Wrote per-row report: %s", out_path)

    if args.summary_json:
        s_path = Path(args.summary_json)
        s_path.parent.mkdir(parents=True, exist_ok=True)
        s_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        logger.info("Wrote summary: %s", s_path)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    main()
