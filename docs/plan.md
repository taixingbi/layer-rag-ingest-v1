# Ingest + Embed Plan (Code-Aligned)

This plan reflects the current implementation in `app/` and `scripts/`.

## End-to-End Pipeline

From repo root, the primary entrypoints are:

- `./scripts/data1.sh` for personal/plain-text ingest
- `./scripts/data2.sh` for GitHub/Markdown ingest

Both scripts run synthetic-question enrichment and smoke validation by default. Set `RUN_SYNTHETIC_QUESTIONS=0` and/or `RUN_SMOKE_VALIDATE=0` to skip stages.

## Stage 1 - Build Chunk Files

Two chunkers are used depending on source shape:

- `app/plain_text_chunks.py` for prose/resume-style text
- `app/markdown_to_chunks.py` for Markdown and GitHub export `.txt`

Outputs are `chunks_*.json` under `<data>/processed/`.

## Stage 2 - Prepare Point Payloads

`app/prepare_payloads.py` converts chunk rows to Qdrant-ready point dictionaries with:

- deterministic `id` (UUID5 over SHA-256 `content_hash` of `payload.text`)
- empty `vector` placeholder (filled later unless already present)
- normalized payload metadata for filtering and traceability

Current payload fields include:

- identity/lineage: `chunk_id`, `chunk_id_parent`, `was_split`, `split_index`
- content: `text`, `embed_text`, `synthetic_questions`
- counters: `token_count`, `embed_token_count`, `synthetic_questions_used`, `synthetic_questions_trimmed`
- filter/audit: `source`, `doc_type`, `section`, `language`, `tags`, `content_hash`, `ingest_run_id`, `ingest_ts`

Implementation notes:

- token counting is whitespace-based (`re.findall(r"\S+", text)`), not model-tokenizer exact
- `embed_text` format is:
  - `[SECTION: <section>]`
  - raw `text`
  - `Q: ...` lines from `synthetic_questions`
- no token-budget trimming is currently applied (`synthetic_questions_trimmed` is always `0` in current builder)

## Stage 3 - Optional Synthetic Question Enrichment

`app/synthetic_questions.py` enriches `points_*.json` rows by calling chat completions and then recomputing:

- `payload.synthetic_questions`
- `payload.embed_text`
- `payload.embed_token_count`
- `payload.synthetic_questions_used` / `payload.synthetic_questions_trimmed`

Operational behavior in code:

- async concurrency with shared `httpx.AsyncClient` and semaphore (`--max-concurrency`, default `40`)
- transient retry policy (`--retry-max-attempts` default `3`, `--retry-base-delay` default `0.5`)
- per-request and per-section latency logging
- failed-item report JSON written under `<data-or-output>/reports/` by default, with EST timestamp suffix
- `--skip-existing` supports incremental enrich runs

## Stage 4 - Embed Missing Vectors

`app/upsert_qdrant.py` loads all matched `points_*.json` and embeds only rows with empty vectors (unless `--skip-embedding` is set).

Embedding source is `payload.embed_text` and requests are sent through `app/client_embeddings.py` integration.

Guardrails:

- if `--skip-embedding` is set for real upsert, script fails when any vector is missing
- vector length is validated against `--vector-size` (default `1024`)

## Stage 5 - Ensure Collection + Indexes

Before upsert, the script:

- resolves collection name from `COLLECTION_NAME` (+ `_<env>` suffix when `ENV` is `dev`/`qa`/`prod`)
- creates collection if missing (unless `--skip-create-collection`)
- creates payload indexes (unless `--skip-indexes`) for:
  - keywords: `source`, `doc_type`, `section`, `content_hash`, `chunk_id_parent`
  - bool: `was_split`
  - integers: `token_count`, `embed_token_count`, `split_index`
  - datetime: `ingest_ts`

## Stage 6 - Batch Upsert

Points are converted to `qdrant_client.models.PointStruct` and upserted in batches (`--batch-size`, default `20`) with `wait=True`.

The command logs running progress and total latency at the end.

Optional in-script smoke execution is available via `upsert_qdrant.py` flags:

- `--run-smoke-validate`
- `--smoke-threshold`
- `--smoke-max-probes`
- `--smoke-strict`

## Stage 7 - Post-Upsert Smoke Validation

`app/smoke_validate.py` validates retrieval quality by generating one probe per `(source, section, doc_type)` group from `points_*.json`.

Probe construction:

- first `synthetic_questions` item when available
- fallback to prefix of `payload.text`

Validation behavior:

- embed probe text using embeddings API
- search Qdrant with deterministic filter (`source`, `section`, `doc_type`)
- pass when top score meets threshold (default `0.75`) and payload remains in expected scope
- write JSON report under `<data-dir>/reports/` by default
- warning-only by default; strict mode exits non-zero (`--strict`)

## Data Set Conventions

- `data1` flow uses `--source-prefix personal`
- `data2` flow uses `--source-prefix repo`

This keeps payload `source` namespaced (for example `personal_profile`, `repo_layer-gateway-embed-v1__design.md`).

## Non-Goals / Not Yet Implemented

The current codebase does **not** implement:

- automatic recursive chunk splitting in `prepare_payloads.py`
- strict model-tokenizer counting for the primary token metrics
- automatic smoke-validation execution inside `upsert_qdrant.py` unless `--run-smoke-validate` is passed

If needed, these can be added as follow-up enhancements.