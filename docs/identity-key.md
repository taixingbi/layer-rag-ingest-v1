# Point Identity Key

This document explains the current point identity model and how to operate it safely.

## Identity model

Point identity is deterministic and document-aware:

- canonical key: `key|source=<source>|document_id=<document_id>|chunk_id=<chunk_id>`
- point `id`: UUID5 of the canonical key
- `content_hash`: content fingerprint for drift detection (not primary identity)

## Identity-related fields

Prepared payloads include:

- `source`
- `document_id`
- `chunk_id`
- `content_hash`

Chunk builders (`plain_text_chunks.py`, `markdown_to_chunks.py`) emit `document_id` in `chunks_*.json`.

## Manifests and reconcile

`prepare_payloads.py` writes manifests with authoritative point IDs and ownership metadata.

Reconcile behavior:

- compares active collection IDs against manifest IDs
- stale detection is `active_collection_ids - manifest_ids`
- preview first, then optional apply

## Safe operation sequence

1. **Backup / snapshot current collection**
   - required rollback safety before large ingest updates
2. **Prepare + upsert**
   - build points from canonical identity key
3. **Reconcile dry-run preview**
   - inspect stale candidates against latest manifest
4. **Apply soft-delete (optional staged cleanup)**
   - only after preview is confirmed
5. **Retention purge**
   - hard-delete tombstoned points older than retention

## Commands

### 1) Build/prepare/upsert

```bash
python3 app/prepare_payloads.py \
  --data-dir data2/processed \
  --output-dir data2/processed \
  --pattern "chunks_*.json" \
  --source-prefix repo

python3 app/upsert_qdrant.py --data-dir data2/processed --pattern "points_*.json"
```

Optional strict approach:

- upsert into a fresh collection name, validate, then switch reader config to the new collection

### 2) Reconcile dry-run (no mutation)

```bash
python3 app/reconcile_qdrant.py \
  --manifest-path data2/processed/ingest_manifest_latest.json \
  --scope-key collection \
  --dry-run
```

### 3) Apply soft-delete for stale candidates

```bash
python3 app/reconcile_qdrant.py \
  --manifest-path data2/processed/ingest_manifest_latest.json \
  --scope-key collection \
  --delete-mode soft \
  --apply-soft-delete
```

### 4) Purge old tombstones (retention-gated)

```bash
python3 app/reconcile_qdrant.py \
  --manifest-path data2/processed/ingest_manifest_latest.json \
  --scope-key collection \
  --delete-mode hard \
  --retention-days 30 \
  --apply-hard-delete
```

## Validation checklist

- Same doc + same chunk rerun -> same `id` (update, no duplicate insert)
- Same text in different docs -> different `id` when identity key differs
- Chunk movement in one doc -> only affected chunk IDs change
- Reconcile dry-run shows expected stale set before apply
- Rollback path remains reproducible from snapshot + manifests/action reports

## Notes

- Keep dry-run as the default operational habit.
- Do not apply delete/purge until stale previews look correct.
- Keep manifests and action reports for auditability and rollback traceability.
