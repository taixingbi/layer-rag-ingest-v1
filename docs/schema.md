# Ingest Data Schema (Stage-by-Stage)

This document shows the canonical shape of data as it moves through the ingest and embedding pipeline.

## Metadata and Filters (Canonical)

Store metadata in each Qdrant payload so retrieval can combine semantic similarity with deterministic filters.

Recommended metadata keys:

- `source`: logical document source (for example `resume_taixing_bi`)
- `doc_type`: high-level type (for example `resume`, `profile`, `qa`)
- `section`: section label used for chunking/retrieval grouping
- `language`: language code (for example `en`)
- `tags`: optional tag list for product/domain filters
- `ingest_ts`: ingest timestamp
- `ingest_run_id`: run identifier for traceability

Recommended filterable keys:

- exact-match (`keyword`): `source`, `doc_type`, `section`, `content_hash`, `chunk_id_parent`
- boolean: `was_split`
- numeric range: `token_count`, `embed_token_count`, `split_index`
- time/range: `ingest_ts`

## Stage 1 - Raw Input

```json
{
  "chunk_id": "0001",
  "section": "ROOT",
  "text": "Taixing Bi (Open to Remote / Hybrid / NYC)...",
  "synthetic_questions": [
    "What is Taixing Bi's email address?",
    "Is Taixing Bi open to remote work?"
  ]
}
```

## Stage 2 - After Normalize + Token Count

```json
{
  "chunk_id": "0001",
  "chunk_id_parent": null,
  "was_split": false,
  "split_index": null,
  "section": "ROOT",
  "text": "Taixing Bi (Open to Remote / Hybrid / NYC)...",
  "token_count": 38,
  "synthetic_questions": [
    "What is Taixing Bi's email address?",
    "Is Taixing Bi open to remote work?"
  ],
  "validation": {
    "status": "ok",
    "flags": []
  }
}
```

## Stage 2b - Split Child (Only If Over 1500 Tokens)

```json
{
  "chunk_id": "0004a",
  "chunk_id_parent": "0004",
  "was_split": true,
  "split_index": 0,
  "section": "PROFESSIONAL EXPERIENCE",
  "text": "...(first half)...",
  "token_count": 498,
  "synthetic_questions": [
    "What infrastructure did the engineer build?"
  ],
  "validation": {
    "status": "ok",
    "flags": []
  }
}
```

## Stage 3 - After `embed_text` Construction

```json
{
  "chunk_id": "0001",
  "chunk_id_parent": null,
  "was_split": false,
  "split_index": null,
  "section": "ROOT",
  "text": "Taixing Bi (Open to Remote / Hybrid / NYC)...",
  "embed_text": "[SECTION: ROOT]\nTaixing Bi (Open to Remote / Hybrid / NYC)...\nQ: What is Taixing Bi's email address?\nQ: Is Taixing Bi open to remote work?",
  "token_count": 38,
  "embed_token_count": 52,
  "synthetic_questions_used": 2,
  "synthetic_questions_trimmed": 0,
  "synthetic_questions": [
    "What is Taixing Bi's email address?",
    "Is Taixing Bi open to remote work?"
  ],
  "validation": {
    "status": "ok",
    "flags": []
  }
}
```

## Stage 4 - After Content Hashing

```json
{
  "chunk_id": "0001",
  "chunk_id_parent": null,
  "was_split": false,
  "split_index": null,
  "section": "ROOT",
  "text": "Taixing Bi (Open to Remote / Hybrid / NYC)...",
  "embed_text": "[SECTION: ROOT]\nTaixing Bi...\nQ: What is Taixing Bi's email address?",
  "token_count": 38,
  "embed_token_count": 52,
  "synthetic_questions_used": 2,
  "synthetic_questions_trimmed": 0,
  "synthetic_questions": [...],
  "content_hash": "e3b0c44298fc1c149afbf4c8996fb924...",
  "point_id": "550e8400-e29b-41d4-a716-446655440000",
  "validation": {
    "status": "ok",
    "flags": []
  }
}
```

## Stage 5 - Qdrant PointStruct (Pre-Embed)

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "vector": [],
  "payload": {
    "chunk_id": "0001",
    "chunk_id_parent": null,
    "was_split": false,
    "split_index": null,
    "section": "ROOT",
    "doc_type": "resume",
    "language": "en",
    "tags": ["candidate", "backend", "ai"],
    "text": "Taixing Bi (Open to Remote / Hybrid / NYC)...",
    "embed_text": "[SECTION: ROOT]\nTaixing Bi...\nQ: What is Taixing Bi's email address?",
    "token_count": 38,
    "embed_token_count": 52,
    "synthetic_questions_used": 2,
    "synthetic_questions_trimmed": 0,
    "synthetic_questions": [...],
    "content_hash": "e3b0c44298fc1c149afbf4c8996fb924...",
    "source": "resume_taixing_bi",
    "ingest_run_id": "run_20260422_100000",
    "ingest_ts": "2026-04-22T10:00:00Z"
  }
}
```

## Stage 6 - After Batch Embed (Vector Attached)

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "vector": [0.0231, -0.0412, 0.0178, "...1024 dims..."],
  "payload": {
    "chunk_id": "0001",
    "chunk_id_parent": null,
    "was_split": false,
    "split_index": null,
    "section": "ROOT",
    "doc_type": "resume",
    "language": "en",
    "tags": ["candidate", "backend", "ai"],
    "text": "Taixing Bi (Open to Remote / Hybrid / NYC)...",
    "embed_text": "[SECTION: ROOT]\nTaixing Bi...\nQ: What is Taixing Bi's email address?",
    "token_count": 38,
    "embed_token_count": 52,
    "synthetic_questions_used": 2,
    "synthetic_questions_trimmed": 0,
    "synthetic_questions": [...],
    "content_hash": "e3b0c44298fc1c149afbf4c8996fb924...",
    "source": "resume_taixing_bi",
    "ingest_run_id": "run_20260422_100000",
    "ingest_ts": "2026-04-22T10:00:00Z"
  }
}
```

## Stage 7 - Upsert Result Log (Per Chunk)

```json
{
  "chunk_id": "0001",
  "point_id": "550e8400-e29b-41d4-a716-446655440000",
  "token_count": 38,
  "embed_token_count": 52,
  "was_split": false,
  "upsert_status": "ok",
  "error": null
}
```

## Stage 8 - Smoke Validation (Per Section)

```json
{
  "section": "ROOT",
  "probe_question": "What is Taixing Bi's email address?",
  "top_result_chunk_id": "0001",
  "top_result_score": 0.94,
  "passed": true,
  "threshold": 0.75,
  "warning": null
}
```

Example retrieval filter used with smoke/query:

```json
{
  "must": [
    {"key": "source", "match": {"value": "resume_taixing_bi"}},
    {"key": "section", "match": {"value": "ROOT"}},
    {"key": "doc_type", "match": {"value": "resume"}}
  ],
  "must_not": [
    {"key": "was_split", "match": {"value": true}}
  ]
}
```

## Full Ingest Run Summary

```json
{
  "ingest_run_id": "run_20260422_100000",
  "source": "resume_taixing_bi",
  "collection": "taixing_knowledge_dev",
  "ingest_ts": "2026-04-22T10:00:00Z",
  "stats": {
    "chunks_input": 6,
    "chunks_split": 0,
    "chunks_total_upserted": 6,
    "chunks_failed": 0,
    "avg_token_count": 201,
    "max_token_count": 687,
    "avg_embed_token_count": 223
  },
  "validation": {
    "sections_passed": 6,
    "sections_failed": 0,
    "warnings": []
  },
  "chunk_results": [
    {
      "chunk_id": "0001",
      "point_id": "550e8400-...",
      "upsert_status": "ok"
    }
  ]
}
```