# RAG Ingest Pipeline

Prepare chunk JSON files, enrich metadata/filter fields, embed text, and upsert points into Qdrant.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Environment

Use environment-specific env files at repo root:

- `.env.dev`
- `.env.qa`
- `.env.prod`

`scripts/data1.sh` can auto-load `.env.<env>` based on its env argument (`dev|qa|prod`).

| Variable | Required | Description |
|---|---|---|
| `QDRANT_URL` | yes | Qdrant endpoint URL |
| `QDRANT_API_KEY` | no | API key for hosted Qdrant |
| `INFERENCE_BASE_URL` | no | Chat inference root (without `/v1`) |
| `EMBEDDINGS_BASE_URL` | yes (for embedding step) | Embeddings root (without `/v1`) |
| `COLLECTION_NAME` | yes | Base collection name (for example `taixing_knowledge`) |
| `ENV` | no | Environment label; if `dev`/`qa`/`prod`, collection auto-resolves to `<COLLECTION_NAME>_<env>` |
| `VECTOR_SIZE` | no | Embedding vector dimension for `upsert_qdrant.py` (default: `1024`; override with `--vector-size`) |
| `BATCH_SIZE` | no | Upsert batch size for `upsert_qdrant.py` (default: `20`; override with `--batch-size`) |
| `EMBEDDING_MODEL` | no | Embedding model id (`BAAI/bge-m3` fallback) |

| `CHAT_BASE_URL` | no | Chat API root; if unset, `synthetic_questions.py` uses `INFERENCE_BASE_URL` |
| `CHAT_MODEL` | no | Chat model id for `synthetic_questions.py` |
| `CHAT_API_KEY` | no | Optional Bearer token for chat (`synthetic_questions.py`) |

## Environment-specific data folders

The dataset root is split by environment:

- `data_dev/data1`
- `data_qa/data1`
- `data_prod/data1`

Shell wrappers resolve this automatically:

```bash

# data1 shorthand (auto-loads .env.dev / .env.qa / .env.prod)
./scripts/data1.sh dev
./scripts/data1.sh qa
./scripts/data1.sh prod
```

For manual Python commands, set a base folder and reuse it in paths:

```bash
export DATA_ROOT=data_dev
# export DATA_ROOT=data_qa
# export DATA_ROOT=data_prod
```

Path shorthand in older examples:

- `data1/...` means `"${DATA_ROOT}/data1/..."`

## Data Flow

1) Build chunk JSON from raw text with `plain_text_chunks.py`  
2) Prepare metadata-rich PointStruct payload files (`points_*.json`)  
2b) *(Optional)* Add synthetic questions to points with `synthetic_questions.py` (updates `embed_text`; re-upsert so vectors match)  
3) Embed missing vectors and upsert to Qdrant  
4) Run post-upsert smoke retrieval validation (warning-only by default in shell wrappers)

## Gold Dataset

Code lives under **`app/rag_gold_eval/`**: `generate_gold_dataset.py` (build JSONL) and `run_eval.py` (score RAG answers vs gold). Gold JSONL usually goes under **`data_<env>/gold_dataset/`**; optional **`run_eval.py`** JSON outputs go under **`data_<env>/report/`** (see `docs/gold-dataset.md`).

Build a consolidated gold QA dataset from all environment points files:

```bash
python3 app/rag_gold_eval/generate_gold_dataset.py
```

Default behavior:
- scans `data_dev`, `data_qa`, `data_prod`
- matches `**/processed/points_*.json`
- writes a consolidated `gold_dataset.jsonl` in the current directory (unless `--skip-consolidated-output`)
- writes split eval files `easy_single_hop.jsonl` and `paraphrase.jsonl` next to that output or under `--split-output-dir`
- outputs one JSONL row per synthetic question in each point
- dedups on `(env, id, question)`

Gold row fields:
- `env`, `source_file`, `id`, `question`, `answer`, `must_contain`
- `source`, `doc_type`, `section`, `chunk_id`, `text`
- `case_type`, `required_sources`, `expected_behavior`

Useful flags:

```bash
# splits only under data_dev/gold_dataset (no data_dev/gold_dataset.jsonl)
python3 app/rag_gold_eval/generate_gold_dataset.py \
  --skip-consolidated-output \
  --split-output-dir data_dev/gold_dataset

# custom consolidated output path (splits default beside that file unless --split-output-dir)
python3 app/rag_gold_eval/generate_gold_dataset.py --output data_dev/gold_dataset/gold_dataset.jsonl

# custom roots and pattern
python3 app/rag_gold_eval/generate_gold_dataset.py \
  --data-roots data_dev data_qa \
  --glob "**/processed/points_*.json"

# include points with missing/empty synthetic_questions
python3 app/rag_gold_eval/generate_gold_dataset.py --include-empty-questions

# keep duplicates
python3 app/rag_gold_eval/generate_gold_dataset.py --no-dedup

# evaluator-ready output (recommended)
python3 app/rag_gold_eval/generate_gold_dataset.py \
  --skip-consolidated-output \
  --split-output-dir data_dev/gold_dataset \
  --enable-must-contain-llm \
  --enable-noisy-queries \
  --max-paraphrases-per-fact 2
```

Detailed runbook: `docs/gold-dataset.md` (includes **RAG eval** with `run_eval.py`).

Split eval files: `easy_single_hop.jsonl`, `paraphrase.jsonl` (see `--split-output-dir` and `--skip-consolidated-output`).

RAG eval over gold JSONL (requires a running RAG gateway):

```bash
python3 app/rag_gold_eval/run_eval.py --gold data_dev/gold_dataset/ --limit 20

# optional: write summary next to dev data
python3 app/rag_gold_eval/run_eval.py --gold data_dev/gold_dataset/ \
  --summary-json data_dev/report/rag_eval_summary.json
```

## Example: full pipeline (`data1`)

From repo root:

**Data** (plain text pipeline):

```bash
# data1 shorthand (auto-loads matching .env.<env>)
./scripts/data1.sh dev
./scripts/data1.sh qa
./scripts/data1.sh prod

```

Shell wrapper [`scripts/data1.sh`](scripts/data1.sh) runs synthetic questions and smoke validation by default; set `RUN_SYNTHETIC_QUESTIONS=0` and/or `RUN_SMOKE_VALIDATE=0` to skip stages. Optional lifecycle reconcile can be enabled with `RUN_RECONCILE=1` (default dry-run; use `RECONCILE_APPLY_SOFT_DELETE=1` to mutate). It resolves dataset paths by `DATA_ENV` (`dev|qa|prod`) using `data_<env>/data1`; `DATA_ROOT` can be set to override the base folder. See [data1.md](docs/data1.md).

Run chunking, prepare, then upsert (adjust paths as needed):

```bash
python3 app/plain_text_chunks.py "${DATA_ROOT}/data1"
python3 app/prepare_payloads.py --data-dir "${DATA_ROOT}/data1/processed" --output-dir "${DATA_ROOT}/data1/processed" --pattern "chunks_*.json" --source-prefix personal
```

Variants:

```bash
# chunks + points only (skip upsert): run the first two commands above

# selected sources only: run plain_text_chunks in single-file mode per stem, or keep only those .txt files under data1/raw

# validate upsert parsing only (existing points)
python3 app/upsert_qdrant.py --data-dir "${DATA_ROOT}/data1/processed" --pattern "points_*.json" --dry-run --skip-embedding
```

### 1) Build chunks from text

Directory mode (recommended):

```bash
python3 app/plain_text_chunks.py data1
```

Single-file mode:

```bash
python3 app/plain_text_chunks.py data1/raw/resume.txt data1/processed/chunks_resume.json
python3 app/plain_text_chunks.py data1/raw/qa.txt data1/processed/chunks_qa.json
python3 app/plain_text_chunks.py data1/raw/profile.txt data1/processed/chunks_profile.json
```

`plain_text_chunks.py` does not call the chat API; every chunk has `synthetic_questions: []`. For LLM questions on prepared payloads, use **`synthetic_questions.py`** on **`points_*.json`** after **`prepare_payloads.py`** (see §2b).

### 2) Prepare metadata + filter payload files

```bash
python3 app/prepare_payloads.py --data-dir data1/processed --output-dir data1/processed --pattern "chunks_*.json" --source-prefix personal
```

This writes:
- `data1/processed/points_resume.json`
- `data1/processed/points_qa.json`
- `data1/processed/points_profile.json`
- `data1/processed/ingest_prepare_summary.json`
- `data1/processed/ingest_manifest_<ingest_run_id>.json`
- `data1/processed/ingest_manifest_latest.json`

Each point includes filter-ready payload fields such as:
- `source`, `doc_type`, `section`, `language`, `tags`
- `document_id`, `chunk_id`
- `was_split`, `split_index`, `token_count`, `embed_token_count`
- `content_hash`, `ingest_run_id`, `ingest_ts`
- lifecycle: `lifecycle_status` (`active|deleted`), `deleted_at`, `deleted_by_run_id`

Point identity contract (v2):
- canonical key: `v2|source=<source>|document_id=<document_id>|chunk_id=<chunk_id>`
- `id` is UUID5 of that key (idempotent reruns for same doc/chunk)

### 2b) Add synthetic questions to `points_*.json` (optional)

Use **`app/synthetic_questions.py`** when points already exist but `payload.synthetic_questions` is empty, or to regenerate questions. The script calls the chat API per point, fills `synthetic_questions`, and recomputes **`embed_text`** / **`embed_token_count`** the same way as `prepare_payloads.py`. Point **`id`** and **`content_hash`** stay the same (they are derived from `text` only).

```bash
# default: --data-dir data, pattern points_*.json, 3 questions each, overwrites files in place
python3 app/synthetic_questions.py --data-dir data1/processed --questions-per-chunk 3

# write to another directory; skip points that already have enough questions
python3 app/synthetic_questions.py --data-dir data1/processed --output-dir data1/processed_enriched --skip-existing --questions-per-chunk 3

# list files and payload counts only (no inference, no writes)
python3 app/synthetic_questions.py --data-dir data1/processed --dry-run
```

Optional chat flags: `--chat-base-url`, `--chat-model`, `--chat-api-key`, `--no-json-object-mode` (same env defaults as elsewhere). After changing **`embed_text`**, run **`upsert_qdrant.py`** again so stored vectors match the new embedding input (unless your upsert always re-embeds from payload).

With `--source-prefix personal`, `payload.source` is namespaced (for example `personal_profile`) for clean filtering.

### 3) Upsert to Qdrant

Dry run parse validation (no network write):

```bash
python3 app/upsert_qdrant.py --data-dir data1/processed --pattern "points_*.json" --dry-run --skip-embedding
```

Normal run (embed + upsert):

```bash
python3 app/upsert_qdrant.py --data-dir data1/processed --pattern "points_*.json"
```

Optional: run smoke validation from inside upsert:

```bash
python3 app/upsert_qdrant.py --data-dir data1/processed --pattern "points_*.json" --run-smoke-validate
```

### 4) Smoke validation (`smoke_validate.py`)

Run retrieval smoke checks after upsert using one probe per `(source, section, doc_type)` group. Probe text uses the first synthetic question when available, otherwise falls back to chunk text prefix. Search is filtered by `source`, `section`, and `doc_type`.

```bash
# warning-only by default (exits 0, writes report)
python3 app/smoke_validate.py --data-dir data1/processed --pattern "points_*.json"

# strict mode (non-zero exit on failures)
python3 app/smoke_validate.py --data-dir data1/processed --pattern "points_*.json" --strict

# optional: enable LLM judge as secondary rescue signal for borderline failures
python3 app/smoke_validate.py \
  --data-dir data1/processed \
  --pattern "points_*.json" \
  --threshold 0.65 \
  --judge-enabled \
  --judge-rescue-floor 0.58
```

Common flags: `--threshold` (default `0.75`), `--max-probes`, `--report-path`, `--strict`, `--judge-enabled`, `--judge-rescue-floor`, `--chat-base-url`, `--chat-model`, `--chat-api-key`.

### 5) Lifecycle reconcile / purge / rollback

Reconcile (dry-run by default) compares active points in a scope vs the current manifest IDs and writes:
- `stale_candidates_<run_id>.json`
- `delete_actions_<run_id>.json`

```bash
# preview stale points (no mutation)
python3 app/reconcile_qdrant.py \
  --manifest-path data1/processed/ingest_manifest_latest.json \
  --scope-key collection \
  --dry-run

# apply soft-delete to stale points
python3 app/reconcile_qdrant.py \
  --manifest-path data1/processed/ingest_manifest_latest.json \
  --scope-key collection \
  --delete-mode soft \
  --apply-soft-delete

# hard purge tombstoned points older than retention
python3 app/reconcile_qdrant.py \
  --manifest-path data1/processed/ingest_manifest_latest.json \
  --scope-key collection \
  --delete-mode hard \
  --retention-days 30 \
  --apply-hard-delete
```

Rollback restores one target ingest run and tombstones non-target points in scope:

```bash
# preview rollback
python3 app/rollback_ingest_run.py \
  --target-run-id run_20260425_180000_EST \
  --manifest-dir data1/processed \
  --scope-key collection \
  --dry-run

# apply rollback
python3 app/rollback_ingest_run.py \
  --target-run-id run_20260425_180000_EST \
  --manifest-dir data1/processed \
  --scope-key collection \
  --apply
```

### 6) GitHub Actions lifecycle automation

Workflows under `.github/workflows`:
- `lifecycle-reconcile.yml`: nightly scheduled dry-run reconcile + manual soft-delete apply (approval + `confirm_apply=YES`).
- `lifecycle-phase2.yml`: manual hard purge / rollback dry-run and apply actions (approval + `confirm_apply=YES`).

### 7) ID migration rollout (v1 -> v2)

Dry-run-first sequence:
1. Run prepare/upsert with current identity IDs.
2. Run reconcile preview and verify stale candidates before any mutation.
3. Apply soft-delete only after preview is reviewed.
4. Purge old tombstones after retention window.

Validation checklist:
- same doc rerun -> no duplicate inserts (same `id`)
- same text in two docs -> distinct `id` values
- chunk movement in one doc -> only affected chunk IDs change
- reconcile dry-run shows expected stale v1 candidates before apply

## Collection naming rule

- Base name: `COLLECTION_NAME`
- If `ENV` is `dev`, `qa`, or `prod`, scripts auto-use `<COLLECTION_NAME>_<env>`
  - examples: `taixing_knowledge_dev`, `taixing_knowledge_qa`, `taixing_knowledge_prod`
- CLI `--collection` overrides the base name.
