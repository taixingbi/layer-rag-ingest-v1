# Gold Dataset Runbook

Generate an evaluator-ready gold QA dataset (JSONL) from `points_*.json` under each data root.

**Script:** `app/generate_gold_dataset.py`

## What it does

- Scans `--data-roots` (default: `data_dev`, `data_qa`, `data_prod`).
- Finds `**/processed/points_*.json` (override with `--glob`).
- For each point: reads `payload.synthetic_questions`, uses `payload.text` as `answer` / `text`.
- Emits one JSONL row per selected question variant (canonical and optional noisy query).
- **Split outputs:** `easy_single_hop.jsonl` and `paraphrase.jsonl` (by `eval_bucket`).
- **Optional consolidated file:** all rows in one JSONL (`--output`), unless `--skip-consolidated-output`.
- **`must_contain`:** heuristic extraction by default; with `--enable-must-contain-llm`, an async batch calls the chat API (`async_chat_completions`, shared `httpx.AsyncClient`, bounded by `--llm-concurrency`, default **40**), with heuristics on failure.

## Quick start (recommended)

Split files only under an env folder (no consolidated `data_dev/gold_dataset.jsonl` beside `data_dev/`):

```bash
python3 app/generate_gold_dataset.py \
  --skip-consolidated-output \
  --split-output-dir data_dev/gold_dataset \
  --enable-must-contain-llm \
  --enable-noisy-queries \
  --max-paraphrases-per-fact 2

python3 app/generate_gold_dataset.py \
  --skip-consolidated-output \
  --split-output-dir data_qa/gold_dataset \
  --enable-must-contain-llm \
  --enable-noisy-queries \
  --max-paraphrases-per-fact 2
```

`--skip-consolidated-output` **requires** `--split-output-dir`.

## Output layout

| Artifact | When |
|----------|------|
| `easy_single_hop.jsonl` | Always written; rows with `eval_bucket` = `easy_single_hop` |
| `paraphrase.jsonl` | Always written; rows with `eval_bucket` = `paraphrase` (often empty unless `--enable-noisy-queries` and `--max-paraphrases-per-fact` ≥ 2) |
| Consolidated JSONL (`--output`) | Written unless `--skip-consolidated-output` (default path: `gold_dataset.jsonl` in cwd) |

If `--split-output-dir` is omitted, split files are written next to the consolidated `--output` path (same directory).

With noisy queries enabled, **line count** of the consolidated file (if written) matches **easy_single_hop + paraphrase** (same rows, different sort: consolidated is globally sorted).

## Output schema

Each JSONL object includes:

- `env`, `source_file`, `id`, `question`, `answer`, `must_contain`
- `source`, `doc_type`, `section`, `chunk_id`, `text`
- `case_type` — always `single_hop` from this script
- `required_sources` — `[]`
- `expected_behavior` — `answer`
- `query_type` — `clean` or `noisy`
- `eval_bucket` — `easy_single_hop` or `paraphrase`

## CLI flags (summary)

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

## Useful one-offs

```bash
# Consolidated + splits under the same folder
python3 app/generate_gold_dataset.py \
  --output data_dev/gold_dataset/gold_dataset.jsonl \
  --split-output-dir data_dev/gold_dataset

# Only some roots
python3 app/generate_gold_dataset.py --data-roots data_dev --skip-consolidated-output --split-output-dir data_dev/gold_dataset
```

## Validation checklist

1. Run the generator; log should show `files_scanned`, `points_processed`, `rows_written`, `duplicates_dropped`, `invalid_single_hop` (**0**). If consolidated was skipped, log shows `consolidated=skipped`.
2. Non-zero line counts in split files (and in consolidated file if written).
3. Spot-check: `id` / `question` / `answer` align with the source `points_*.json` payload.
4. Quality: non-empty `must_contain`; variant count respects `--max-paraphrases-per-fact`.

## Notes

- Rows use `env` = folder name with `data_` stripped (e.g. `data_dev` → `dev`).
- Dedup key: `(env, id, question)` when dedup is enabled.
- Heuristic `must_contain` is computed once per `(env, point id)` before paraphrase rows; the LLM pass dedupes by normalized answer text per env to limit API calls.
- If a root has no `points_*.json` files, that environment contributes no rows.
