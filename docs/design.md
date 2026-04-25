# RAG Ingest Pipeline Design

## Purpose

This document describes the concrete architecture of the ingest pipeline in this repository: how raw content becomes filterable, vectorized Qdrant points with deterministic IDs and repeatable behavior.

It complements:

- `README.md` for operator usage
- `docs/plan.md` for stage-level execution plan
- `docs/schema.md` for payload shape examples

## Goals

- Produce idempotent, Qdrant-ready point files from heterogeneous text inputs.
- Keep metadata stable and query-friendly for deterministic filtering.
- Support optional enrichment (`synthetic_questions`) without changing point identity.
- Provide safe operational controls (dry runs, retries, summaries, latency logs).

## Non-Goals

- Retrieval-time ranking/reranking logic.
- Online query orchestration and response composition.
- Automatic long-chunk recursive split.

## System Context

The pipeline is batch-oriented and file-backed:

1. Input files are read from data folders (`data1/raw`, `data2/raw`).
2. Intermediate JSON artifacts are written to `processed/`.
3. Final stage embeds and upserts into a Qdrant collection.

Design choice: files are preserved between stages to make each step debuggable and rerunnable independently.

## Architecture Overview

### Runtime Entry Points

- `scripts/data1.sh`: plain-text corpus pipeline (`data1`)
- `scripts/data2.sh`: GitHub/Markdown corpus pipeline (`data2`)

Both scripts:

- run from any cwd by resolving repo root
- default to running synthetic enrichment (`RUN_SYNTHETIC_QUESTIONS=0` disables)
- default to running smoke validation (`RUN_SMOKE_VALIDATE=0` disables)
- orchestrate chunk -> prepare -> enrich (optional) -> upsert -> smoke validate

### Core Modules

- `app/plain_text_chunks.py`
  - Converts prose-style `.txt` sources into `chunks_*.json`.
- `app/markdown_to_chunks.py`
  - Converts Markdown (and GitHub-export `.txt`) into `chunks_*.json` with heading-aware segmentation.
- `app/prepare_payloads.py`
  - Transforms chunks into Qdrant point dictionaries (`points_*.json`) and writes run summary + ingest manifests.
- `app/synthetic_questions.py`
  - Async chat-enrichment pass for existing `points_*.json`; recomputes `embed_text` fields.
- `app/upsert_qdrant.py`
  - Embeds missing vectors and upserts points into Qdrant, with collection/index bootstrap options.
- `app/smoke_validate.py`
  - Runs post-upsert retrieval smoke checks and writes JSON reports.
- `app/reconcile_qdrant.py`
  - Reconciles active points against ingest manifest IDs and supports soft-delete / retention purge operations.
- `app/rollback_ingest_run.py`
  - Restores lifecycle state to a target ingest run and writes rollback action reports.

## End-to-End Data Flow

### Stage A: Chunk Generation

Input:

- raw text or markdown sources in `<data>/raw`

Output:

- `<data>/processed/chunks_<stem>.json`

Design notes:

- Chunkers do not call chat APIs.
- `synthetic_questions` starts empty in chunk outputs.
- Chunking strategy depends on source format (plain text vs Markdown headings).

### Stage B: Point Preparation (`prepare_payloads.py`)

Input:

- `chunks_*.json`

Output:

- `points_*.json`
- `ingest_prepare_summary.json`
- `ingest_manifest_<ingest_run_id>.json`
- `ingest_manifest_latest.json`

Responsibilities:

- Validate essential fields (`chunk_id`, non-empty `text`).
- Build `payload.embed_text` from section, text, and `synthetic_questions`.
- Compute:
  - `content_hash` = SHA-256 of `payload.text`
  - `id` = UUID5(namespace, `content_hash`) for deterministic point identity
- Attach stable metadata:
  - lineage: `chunk_id_parent`, `was_split`, `split_index`
  - filter fields: `source`, `doc_type`, `section`, `language`, `tags`
  - audit fields: `ingest_run_id`, `ingest_ts`
- lifecycle fields: `lifecycle_status`, `deleted_at`, `deleted_by_run_id`

Important invariant: if `text` is unchanged, `content_hash` and `id` remain unchanged across reruns.

### Stage C: Synthetic Enrichment (`synthetic_questions.py`, optional)

Input:

- `points_*.json` with payload text

Output:

- updated `points_*.json` (in place or to `--output-dir`)
- failed-item report under `reports/` by default

Responsibilities:

- Generate N questions per point via chat completions.
- Recompute `embed_text`, `embed_token_count`, and question counters.
- Keep `id` and `content_hash` stable (derived from `text`, not generated questions).

Operational behavior:

- shared `httpx.AsyncClient`
- bounded concurrency (`--max-concurrency`, default `40`)
- transient retries with exponential backoff + jitter
- per-request and per-section latency logging
- dry-run mode and skip-existing mode for safer incremental runs

### Stage D: Embedding + Upsert (`upsert_qdrant.py`)

Input:

- prepared/enriched `points_*.json`

Output:

- points persisted in Qdrant

Responsibilities:

- Embed only points missing vectors (unless `--skip-embedding`).
- Enforce vector-size consistency (`--vector-size`, default `1024`).
- Resolve collection name from `COLLECTION_NAME` and `ENV`:
  - `ENV=dev|qa|prod` -> append `_<env>` if missing.
- Optionally create collection and payload indexes.
- Upsert in batches (`--batch-size`, default `20`) with `wait=True`.

Optional in-upsert smoke hook:

- `--run-smoke-validate` runs smoke checks immediately after upsert.
- `--smoke-strict` converts smoke failures into non-zero command exit.

### Stage E: Post-Upsert Smoke Validation (`smoke_validate.py`)

Input:

- `points_*.json`
- target Qdrant collection and embedding endpoint

Output:

- smoke report JSON under `<data-dir>/reports/` by default

Responsibilities:

- Select one probe per `(source, section, doc_type)` group.
- Use first synthetic question as probe text when available, otherwise fallback text prefix.
- Embed probes, run filtered vector search, and validate:
  - score meets threshold (default `0.75`)
  - top hit payload remains within expected scope (`source`, `section`, `doc_type`)
- Emit warning-only summary by default; support strict failure mode.

### Stage F: Reconcile + Purge (`reconcile_qdrant.py`)

Responsibilities:

- Load one ingest manifest and compare current active IDs in scope (`collection`/`source`/`doc_type`).
- Write stale candidate report before any mutation.
- Soft-delete stale IDs only when explicitly requested.
- Hard-delete only tombstoned points older than configured retention when explicitly requested.

Safety defaults:

- dry-run semantics by default (apply flags required for mutation)
- artifacts are always written to support auditability

### Stage G: Rollback (`rollback_ingest_run.py`)

Responsibilities:

- Select target `ingest_run_id` by manifest.
- Reactivate IDs from target run.
- Tombstone non-target IDs in same scope.
- Write rollback action report for reproducibility.

## Data Model and Queryability

### Why metadata is embedded in payload

Similarity alone is not enough for production retrieval isolation. Payload metadata enables deterministic filters by corpus, type, section, and ingest lineage.

Indexed filter keys currently include:

- keyword: `source`, `doc_type`, `section`, `content_hash`, `chunk_id_parent`
- bool: `was_split`
- integer: `token_count`, `embed_token_count`, `split_index`
- datetime: `ingest_ts`
- lifecycle keys: `lifecycle_status`, `deleted_at`, `deleted_by_run_id`

### Source namespacing convention

- `data1` uses `--source-prefix personal`
- `data2` uses `--source-prefix repo`

This avoids collisions and enables strict filtering between personal and repository corpora.

## Reliability and Idempotency

### Idempotent identity

- Point ID is deterministic from text hash.
- Re-running the same source text performs upsert updates, not duplicate inserts.

### Failure isolation

- `prepare_payloads.py` tracks per-file failures in its summary instead of crashing entire runs on every bad row.
- `synthetic_questions.py` records failed rows and continues processing remaining points.

### Controlled mutability

- Enrichment mutates embedding input (`embed_text`) but not point identity fields (`id`, `content_hash`).
- This allows iterative improvement of retrieval quality without ID churn.

## Configuration Model

Primary environment controls:

- Qdrant: `QDRANT_URL`, `QDRANT_API_KEY`, `COLLECTION_NAME`, `ENV`
- Embedding API: `EMBEDDINGS_BASE_URL`, `EMBEDDING_MODEL`, `EMBEDDING_API_KEY`
- Chat API: `CHAT_BASE_URL`, `CHAT_MODEL`, `CHAT_API_KEY` (or `INFERENCE_BASE_URL` fallback)

Design principle: CLI flags override env values for run-scoped adjustments without changing shared `.env`.

## Observability

The pipeline emits operator-facing logs for:

- stage completion and counts
- batch upsert progress
- total latency per command
- chat latency per call and per section totals
- synthetic-enrichment failure reports with run args and timestamps
- smoke-validation pass/fail summaries and per-probe report artifacts

This is intentionally lightweight (structured-enough text logs + JSON artifacts) to keep local and CI execution simple.

## Trade-Offs

- **File-based stages over direct streaming:** easier debugging and restartability; costs extra disk I/O.
- **Deterministic IDs from text only:** stable identity; does not reflect enrichment changes in ID.
- **Tokenizer dependency for counting:** token metrics now depend on loading the embedding model tokenizer (`transformers`), which is more accurate but adds startup/dependency cost.
- **Default-on synthetic enrichment in shell wrappers:** better out-of-box retrieval context; more API/runtime cost unless disabled.
- **Default-on smoke validation in shell wrappers:** immediate quality signal after upsert; adds extra API and Qdrant checks.

## Future Enhancements

- Token-budget-aware trimming in `prepare_payloads.py` (currently no trimming).
- Stronger dead-letter handling and retry classification across all networked stages.
