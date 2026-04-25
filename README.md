# RAG Ingest Pipeline

Prepare chunk JSON files, enrich metadata/filter fields, embed text, and upsert points into Qdrant.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Environment

Create `.env` at repo root.

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

## Data Flow

1) Build chunk JSON from raw text (see below: `plain_text_chunks.py` for prose/resume-style sources, `markdown_to_chunks.py` for Markdown and GitHub-exported `.txt`)  
2) Prepare metadata-rich PointStruct payload files (`points_*.json`)  
2b) *(Optional)* Add synthetic questions to points with `synthetic_questions.py` (updates `embed_text`; re-upsert so vectors match)  
3) Embed missing vectors and upsert to Qdrant  
4) Run post-upsert smoke retrieval validation (warning-only by default in shell wrappers)

## Example: full pipeline (`data1`)

From repo root:

**Data** (plain text pipeline):

```bash
./scripts/data1.sh
./scripts/data2.sh
```

Shell wrappers [`scripts/data1.sh`](scripts/data1.sh) and [`scripts/data2.sh`](scripts/data2.sh) run synthetic questions and smoke validation by default; set `RUN_SYNTHETIC_QUESTIONS=0` and/or `RUN_SMOKE_VALIDATE=0` to skip stages. Optional lifecycle reconcile can be enabled with `RUN_RECONCILE=1` (default dry-run; use `RECONCILE_APPLY_SOFT_DELETE=1` to mutate). See [data1.md](data1.md) / [data2.md](data2.md).

Run chunking, prepare, then upsert (adjust paths as needed):

```bash
python3 app/plain_text_chunks.py data1
python3 app/prepare_payloads.py --data-dir data1/processed --output-dir data1/processed --pattern "chunks_*.json" --source-prefix personal
```

Variants:

```bash
# chunks + points only (skip upsert): run the first two commands above

# selected sources only: run plain_text_chunks in single-file mode per stem, or keep only those .txt files under data1/raw

# validate upsert parsing only (existing points)
python3 app/upsert_qdrant.py --data-dir data1/processed --pattern "points_*.json" --dry-run --skip-embedding
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

#### Markdown and GitHub-exported docs (`markdown_to_chunks.py`)

Use this when sources are Markdown (or `.txt` from `github_tree_to_txt.py` that still contain Markdown). It splits on ATX headings (`#` … `######`), strips the GitHub export preamble (`source:` / `path_in_archive:` / `---`), and packs paragraphs to a target size. Output matches `plain_text_chunks.py` so `prepare_payloads.py` is unchanged.

Directory mode (`<root>/raw/*.{txt,md}` → `<root>/processed/chunks_<stem>.json`):

```bash
python3 app/markdown_to_chunks.py data2
```

Single-file mode:

```bash
python3 app/markdown_to_chunks.py data2/raw/layer-gateway-embed-v1__design.md.txt data2/processed/chunks_layer-gateway-embed-v1__design.md.json
```

Optional: `--min-chunk-chars` (default `400`), `--max-chunk-chars` (default `2800`). Chunks always have `synthetic_questions: []`. **`plain_text_chunks.py` is the same** (no in-script LLM calls). After **`prepare_payloads.py`**, run **`synthetic_questions.py`** on **`points_*.json`** if you want LLM questions.

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
- `was_split`, `split_index`, `token_count`, `embed_token_count`
- `content_hash`, `ingest_run_id`, `ingest_ts`
- lifecycle: `lifecycle_status` (`active|deleted`), `deleted_at`, `deleted_by_run_id`

### 2b) Add synthetic questions to `points_*.json` (optional)

Use **`app/synthetic_questions.py`** when points already exist but `payload.synthetic_questions` is empty (for example after `markdown_to_chunks.py`), or to regenerate questions. The script calls the chat API per point, fills `synthetic_questions`, and recomputes **`embed_text`** / **`embed_token_count`** the same way as `prepare_payloads.py`. Point **`id`** and **`content_hash`** stay the same (they are derived from `text` only).

```bash
# default: --data-dir data, pattern points_*.json, 3 questions each, overwrites files in place
python3 app/synthetic_questions.py --data-dir data1/processed --questions-per-chunk 3

python3 app/synthetic_questions.py --data-dir data2/processed --questions-per-chunk 3

# write to another directory; skip points that already have enough questions
python3 app/synthetic_questions.py --data-dir data2/processed --output-dir data2/processed_enriched --skip-existing --questions-per-chunk 3

# list files and payload counts only (no inference, no writes)
python3 app/synthetic_questions.py --data-dir data2/processed --dry-run
```

Optional chat flags: `--chat-base-url`, `--chat-model`, `--chat-api-key`, `--no-json-object-mode` (same env defaults as elsewhere). After changing **`embed_text`**, run **`upsert_qdrant.py`** again so stored vectors match the new embedding input (unless your upsert always re-embeds from payload).

## GitHub docs (`data2`)

1. List GitHub folder URLs (one per line) in `data2/raw/repo.txt`, for example  
   `https://github.com/<org>/<repo>/tree/<branch>/docs`
2. Download each tree as zip, extract text-like files, write `.txt` under `--out-dir` (default `data2/raw/github_docs_txt` or pass `--out-dir data2/raw` to drop files next to `repo.txt`).
3. Chunk with **`markdown_to_chunks.py`** (not `plain_text_chunks.py` for these exports).
4. Run **`prepare_payloads.py`** then **`upsert_qdrant.py`** (see commands below).

```bash
python3 app/github_tree_to_txt.py --repo-list data2/raw/repo.txt --out-dir data2/raw
python3 app/markdown_to_chunks.py data2
python3 app/prepare_payloads.py --data-dir data2/processed --output-dir data2/processed --pattern "chunks_*.json" --source-prefix repo
# optional: add synthetic questions to points before upsert
python3 app/synthetic_questions.py --data-dir data2/processed --questions-per-chunk 3
python3 app/upsert_qdrant.py --data-dir data2/processed --pattern "points_*.json"
```

For Markdown-heavy `data2`, use the command block above (chunk with `markdown_to_chunks.py`, then prepare, optional `synthetic_questions.py`, then upsert).

With those `prepare_payloads.py` flags, `payload.source` is namespaced for clean filtering:
- `personal_*` for personal context (`data1`)
- `repo_*` for repository context (`data2`)

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

## Collection naming rule

- Base name: `COLLECTION_NAME`
- If `ENV` is `dev`, `qa`, or `prod`, scripts auto-use `<COLLECTION_NAME>_<env>`
  - examples: `taixing_knowledge_dev`, `taixing_knowledge_qa`, `taixing_knowledge_prod`
- CLI `--collection` overrides the base name.
