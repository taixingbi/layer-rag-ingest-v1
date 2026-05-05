# Gold Dataset Runbook

Generate and evaluate an evaluator-ready gold QA dataset (JSONL) from `points_*.json` under each data root.

**Code package:** `app/rag_gold_eval/` (generator + RAG eval).  
**Typical artifact dirs:** `data_<env>/gold_dataset/` (e.g. `data_dev/gold_dataset/`) for gold JSONL, and **`data_<env>/report/`** (e.g. `data_dev/report/`) for optional `run_eval.py` JSON outputs — not the same folder as the Python package.

**Scripts**

| Script | Purpose |
|--------|---------|
| `app/rag_gold_eval/generate_gold_dataset.py` | Build gold JSONL from `points_*.json` |
| `app/rag_gold_eval/run_eval.py` | Call RAG `POST /v1/rag/query` per gold row and score answers |

---

## Generator (`generate_gold_dataset.py`)

### What it does

- Scans `--data-roots` (default: `data_dev`, `data_qa`, `data_prod`).
- Finds `**/processed/points_*.json` (override with `--glob`).
- For each point: reads `payload.synthetic_questions`, uses `payload.text` as `answer` / `text`.
- Emits one JSONL row per selected question variant (canonical and optional noisy query).
- **Split outputs:** `easy_single_hop.jsonl` and `paraphrase.jsonl` (by `eval_bucket`).
- **Optional consolidated file:** all rows in one JSONL (`--output`), unless `--skip-consolidated-output`.
- **`must_contain`:** heuristic extraction by default; with `--enable-must-contain-llm`, an async batch calls the chat API (`async_chat_completions`, shared `httpx.AsyncClient`, bounded by `--llm-concurrency`, default **40**), with heuristics on failure.

### Quick start (recommended)

Split files only under an env folder (no consolidated `data_dev/gold_dataset.jsonl` beside `data_dev/`):

```bash
python3 app/rag_gold_eval/generate_gold_dataset.py \
  --skip-consolidated-output \
  --split-output-dir data_dev/gold_dataset \
  --enable-must-contain-llm \
  --enable-noisy-queries \
  --max-paraphrases-per-fact 2
```

Use `--split-output-dir data_qa/gold_dataset` (or `data_prod/...`) the same way for other envs.

`--skip-consolidated-output` **requires** `--split-output-dir`.

### Output layout

| Artifact | When |
|----------|------|
| `easy_single_hop.jsonl` | Always written; rows with `eval_bucket` = `easy_single_hop` |
| `paraphrase.jsonl` | Always written; rows with `eval_bucket` = `paraphrase` (often empty unless `--enable-noisy-queries` and `--max-paraphrases-per-fact` ≥ 2) |
| Consolidated JSONL (`--output`) | Written unless `--skip-consolidated-output` (default path: `gold_dataset.jsonl` in cwd) |

If `--split-output-dir` is omitted, split files are written next to the consolidated `--output` path (same directory).

With noisy queries enabled, **line count** of the consolidated file (if written) matches **easy_single_hop + paraphrase** (same rows; consolidated is globally sorted).

### Output schema

Each JSONL object includes:

- `env`, `source_file`, `id`, `question`, `answer`, `must_contain`
- `source`, `doc_type`, `section`, `chunk_id`, `text`
- `case_type` — `single_hop` for rows emitted by the generator
- `required_sources` — `[]` for those rows; non-empty for hand-authored **multi-hop** rows
- `expected_behavior` — `answer` for generator rows; may differ for curated negatives
- `query_type` — `clean` or `noisy`
- `eval_bucket` — `easy_single_hop` or `paraphrase` for generator rows

### Generator CLI (summary)

| Flag | Role |
|------|------|
| `--data-roots` | Roots to scan (default: `data_dev data_qa data_prod`) |
| `--glob` | Glob under each root (default: `**/processed/points_*.json`) |
| `--output` | Consolidated JSONL path (default: `gold_dataset.jsonl`) |
| `--skip-consolidated-output` | Omit consolidated file; **must** pass `--split-output-dir` |
| `--split-output-dir` | Directory for `easy_single_hop.jsonl` / `paraphrase.jsonl` |
| `--include-empty-questions` | One row with empty question if no synthetic questions |
| `--no-dedup` | Keep duplicate `(env, id, question)` rows |
| `--enable-must-contain-llm` | LLM `must_contain` extraction (needs inference + `httpx`) |
| `--enable-noisy-queries` | Add one noisy variant when cap allows |
| `--max-paraphrases-per-fact` | Max variants per fact (default `3`; use `2` for clean + noisy only) |
| `--chat-base-url` | Inference root (default: `CHAT_BASE_URL` or script default) |
| `--chat-model` | Model (default: `CHAT_MODEL` or `Qwen/Qwen2.5-7B-Instruct`) |
| `--chat-api-key` | Optional bearer token (`CHAT_API_KEY`) |
| `--llm-concurrency` | Max concurrent async LLM calls (default **40**) |

### Useful one-offs

```bash
# Consolidated + splits under the same folder
python3 app/rag_gold_eval/generate_gold_dataset.py \
  --output data_dev/gold_dataset/gold_dataset.jsonl \
  --split-output-dir data_dev/gold_dataset

# Only some roots
python3 app/rag_gold_eval/generate_gold_dataset.py --data-roots data_dev --skip-consolidated-output --split-output-dir data_dev/gold_dataset
```

---

## RAG evaluation (`run_eval.py`)

Runs **`POST {rag_base_url}/v1/rag/query`** for each gold row’s `question`, then scores the JSON response.

### Scoring rules

1. **`must_contain`** — Each non-empty fragment must appear as a substring of the RAG **`answer`** (case-insensitive, whitespace-normalized). Rows with no `must_contain` entries are not scored on this axis (`must_contain_total` = 0 counts as pass for that check in summaries).
2. **`source` (single-hop)** — If the gold row has a non-empty `source` and it is not `multi` or `negative`, at least one citation must list that `source`.
3. **`required_sources` (multi-hop)** — If the list is non-empty, every listed source must appear among citation `source` values.
4. **`retrieval_hits` (retrieval quality)** — By default the client sends `"include_retrieval_hits": true`. For each gold row whose **`id` is a UUID** (same as point / `chunk_id` in hits), the script finds that id in the ordered **`retrieve`** and **`rerank`** hit lists, then records **1-based rank**, **RR** and **MRR**, plus **Recall@k**, **Precision@k**, **NDCG@k**, and **F1@k** per `--recall-at-k` (default `5,10,40`). Rows with non-UUID `id` (e.g. synthetic multi-hop ids) get `retrieval_eval_skipped` and do not affect retrieval aggregates.
5. **`quality_dimensions` (answer quality, heuristic)** — Per row: `correct`, `faithful`, `complete`, `precise`, `cited` plus `quality_score` (mean of the 5 binary dimensions). Summary includes pass counts/rates and `quality_score_mean`.

Each request sends `collection_base`, `request_id`, `session_id`, `k`, `k_max`, and (unless `--skip-retrieval-hits`) `include_retrieval_hits` as in the RAG API.

### Environment variables

| Variable | Used when |
|----------|-----------|
| `RAG_BASE_URL` | Default `--rag-base-url` (e.g. `http://192.168.86.179:30183`) |
| `RAG_COLLECTION_BASE` | Default `--collection-base` (e.g. `taixing_knowledge`) |

### `run_eval.py` CLI

| Flag | Default | Role |
|------|---------|------|
| `--gold` | (required) | JSONL file(s) or directories of `*.jsonl` |
| `--rag-base-url` | `RAG_BASE_URL` or `http://192.168.86.179:30183` | Gateway base URL (no `/v1` suffix) |
| `--collection-base` | `RAG_COLLECTION_BASE` or `taixing_knowledge` | `collection_base` in JSON body |
| `--k` | `5` | Retrieval `k` |
| `--k-max` | `40` | Retrieval `k_max` |
| `--concurrency` | `20` | Max concurrent async RAG requests |
| `--limit` | `0` | Max rows (`0` = all) |
| `--skip-retrieval-hits` | off | Omit `include_retrieval_hits` from the request; skip retrieval rank / RR/MRR / Recall@k / Precision@k / NDCG@k / F1@k |
| `--recall-at-k` | `5,10,40` | Comma-separated k values for per-row and summary `recall_at_*_*`, `precision_at_*_*`, `ndcg_at_*_*`, `f1_at_*_*` |
| `--report-json` | off | Write full per-row results (JSON array) to this path; parent dirs are created |
| `--summary-json` | off | Write the same summary object as stdout to this path |

### Where output goes

By default **`run_eval.py` does not write any report file** — it only **prints the JSON summary to stdout** (terminal). When you do write files, use each env’s **`data_<env>/report/`** folder (e.g. `data_dev/report/rag_eval_summary.json`) so outputs stay next to that environment’s data.

- **`--summary-json PATH`** — save the summary JSON to a file (same content as stdout).
- **`--report-json PATH`** — save the large per-row array (every row’s scores, previews, retrieval fields); use for debugging failures.
- **Shell redirect** — `python3 app/rag_gold_eval/run_eval.py ... > data_dev/report/rag_eval_summary.stdout.json` captures **stdout only** (summary, not per-row).

### Examples

```bash
# All JSONL under a gold_dataset folder (sorted *.jsonl); summary only on stdout
python3 app/rag_gold_eval/run_eval.py --gold data_dev/gold_dataset/

# Same run, persist summary + per-row report under data_dev/report/
python3 app/rag_gold_eval/run_eval.py --gold data_dev/gold_dataset/ \
  --report-json data_dev/report/rag_eval_report.json \
  --summary-json data_dev/report/rag_eval_summary.json

# QA env: use data_qa paths for gold + report
python3 app/rag_gold_eval/run_eval.py --gold data_qa/gold_dataset/ \
  --summary-json data_qa/report/rag_eval_summary.json

# Single split file, smoke test first 10 rows
python3 app/rag_gold_eval/run_eval.py \
  --gold data_dev/gold_dataset/easy_single_hop.jsonl \
  --limit 10

# Explicit gateway + collection + reports under data_dev/report/
python3 app/rag_gold_eval/run_eval.py \
  --gold data_dev/gold_dataset/paraphrase.jsonl \
  --rag-base-url http://192.168.86.179:30183 \
  --collection-base taixing_knowledge \
  --k 5 \
  --k-max 40 \
  --report-json data_dev/report/rag_eval_paraphrase_report.json \
  --summary-json data_dev/report/rag_eval_paraphrase_summary.json
```

Unless **`--summary-json`** is set, the only output is this **JSON summary on stdout**, including **`retrieval_scored_rows`**, **`mean_rr_retrieve`**, **`mean_rr_rerank`**, **`mrr_retrieve`**, **`mrr_rerank`**, **`retrieval_found_retrieve`**, **`mean_rr_*_when_found`**, and for each k **`recall_at_{k}_*`**, **`precision_at_{k}_*`**, **`ndcg_at_{k}_*`**, **`f1_at_{k}_*`** (for `retrieve` and `rerank`), latency aggregates (**`latency_scored_rows`**, **`latency_ms_mean`**, **`latency_ms_min`**, **`latency_ms_max`**, **`latency_ms_p50`**, **`latency_ms_p95`**, **`latency_ms_p99`**), quality aggregates (**`quality_*_pass`**, **`quality_*_rate`**, **`quality_score_mean`**), `must_contain_*`, `gold_source_*`, `required_sources_*`, and `errors_sample`.

---

## Validation checklist

### Generator

1. Log shows `files_scanned`, `points_processed`, `rows_written`, `duplicates_dropped`, `invalid_single_hop` (**0**). If consolidated was skipped, log shows `consolidated=skipped`.
2. Non-zero line counts in split files (and in consolidated file if written).
3. Spot-check: `id` / `question` / `answer` align with the source `points_*.json` payload.
4. Quality: non-empty `must_contain` where expected; variant count respects `--max-paraphrases-per-fact`.

### RAG eval

1. Gateway reachable from the machine running `run_eval.py`.
2. Summary `rag_calls_failed` is **0** (or inspect `errors_sample` / `--report-json`).
3. `must_contain_pass` and citation checks align with your eval bar (strict `must_contain` vs. acceptable paraphrases).
4. If retrieval metrics are enabled: `retrieval_scored_rows` matches your expectation (UUID `id` rows only); inspect `recall_at_*_retrieve` / `mean_rr_retrieve` for retrieval health.
5. Check latency metrics (`latency_ms_p50`, `latency_ms_p95`, `latency_ms_p99`) for runtime stability.

## Notes

- Rows use `env` = folder name with `data_` stripped (e.g. `data_dev` → `dev`).
- Dedup key: `(env, id, question)` when dedup is enabled on the generator.
- Heuristic `must_contain` is computed once per `(env, point id)` before paraphrase rows; the LLM pass dedupes by normalized answer text per env to limit API calls.
- If a root has no `points_*.json` files, that environment contributes no rows.
