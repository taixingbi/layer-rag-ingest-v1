# Ingest + Embed Plan

This plan describes a robust, idempotent ingest pipeline for resume chunk data.

## Stage 1 - Normalize and Validate

- Load raw JSON and validate required fields.
- Normalize whitespace and strip control characters.
- Fail fast with per-chunk error reporting; do not silently drop bad records.

## Stage 2 - Token Counting (No Split for Current Dataset)

- Tokenize each chunk using `tiktoken` (`cl100k`) as a close proxy for `bge-m3`.
- If exact tokenizer parity is needed, use `transformers.AutoTokenizer`.
- For current resume data, all chunks are under 1500 tokens, so no split is required.
- For future chunks over 1500 tokens:
  - Apply recursive character splitting with 150-token overlap.
  - Preserve parent lineage (`3_0 -> 3_0_a`, `3_0_b`).
  - Re-validate each child is under 1500 tokens.
- Flag any chunk under 20 tokens as degenerate.

## Stage 3 - `embed_text` Construction

- Build `embed_text` by concatenating section header, chunk text, and synthetic questions.

```text
[SECTION: PROFESSIONAL EXPERIENCE]
{text}
Q: What infrastructure did the engineer build at Saks?
Q: What was the P95 latency maintained?
...
```

- Re-count tokens on `embed_text`.
- If augmented content exceeds 1500 tokens, trim synthetic questions from the bottom up until it fits.
- Never truncate the core `text`.
- Store both:
  - `text` (clean display content)
  - `embed_text` (augmented embedding content)

## Stage 4 - Content Hashing and Dedup

- Compute SHA-256 over the raw `text` field to produce `content_hash`.
- Generate deterministic UUID5 from `content_hash` for stable Qdrant `point_id`.
- Re-ingesting the same document produces the same ID, making Qdrant upserts idempotent.

## Stage 4b - Metadata Normalization and Filter Strategy

- Add canonical metadata to every chunk before PointStruct mapping:
  - `source` (document source id)
  - `doc_type` (`resume`, `profile`, `qa`, etc.)
  - `section` (already present; keep normalized)
  - `language` (for example `en`)
  - `tags` (optional list for domain/user filters)
  - `ingest_run_id`, `ingest_ts` (traceability/audit)
- Define filter semantics:
  - exact keyword filters: `source`, `doc_type`, `section`, `content_hash`, `chunk_id_parent`
  - boolean filters: `was_split`
  - numeric range filters: `token_count`, `embed_token_count`, `split_index`
  - time filters: `ingest_ts`
- Keep metadata values stable and low-cardinality where possible (especially for `section`, `doc_type`) to improve filter performance.
- Ensure metadata generation is deterministic for re-runs of the same source.

## Stage 5 - Qdrant Payload Mapping

```python
PointStruct(
  id=uuid5(NAMESPACE, content_hash),
  vector=[],  # filled in stage 6
  payload={
    "chunk_id": "3_0",
    "chunk_id_parent": None,       # "3_0" if this is a split child
    "was_split": False,
    "split_index": None,           # 0, 1, 2... if split
    "section": "PROFESSIONAL EXPERIENCE",
    "doc_type": "resume",
    "language": "en",
    "tags": ["candidate", "backend", "ai"],
    "text": "...",                 # original, stored for retrieval display
    "embed_text": "...",           # augmented, what was actually embedded
    "char_count": 2901,
    "token_count": 687,            # exact, from tokenizer
    "embed_token_count": 712,      # token count of embed_text
    "content_hash": "abc123...",
    "synthetic_questions": [...],
    "source": "resume_taixing_bi",
    "ingest_run_id": "run_20260422_100000",
    "ingest_ts": "2026-04-22T..."
  }
)
```

## Stage 6 - Batch Embed

- Collect all `embed_text` values.
- Make one batched call to `bge-m3` through the vLLM `/v1/embeddings` endpoint.
- Attach returned vectors back to `PointStruct` entries by index.
- Avoid serial per-chunk embedding calls.

## Stage 7 - Qdrant Upsert

- Upsert to `taixing_knowledge_dev` with `wait=True`.
- Log per chunk: `chunk_id`, `token_count`, `was_split`, `point_id`, `upsert_status`.
- Collect and surface failures without crashing the full batch.

## Stage 8 - Smoke Validation

- After upsert, run one query per section using the first synthetic question as the probe.
- Assert each section scores above a minimum similarity threshold (for example, `0.75`).
- Log warnings for failed sections to catch embedding or ingest quality issues early.
- Apply deterministic filters during smoke checks (at minimum `source`, `section`, `doc_type`) so validation measures the intended slice of data.