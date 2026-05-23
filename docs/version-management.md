# Version Management

This document defines practical version-management rules for the ingest pipeline.

## Scope

Versioning in this repo applies to:

- payload/data contract changes
- identity key contract changes
- script CLI behavior changes
- workflow/runbook changes

## Principles

- Prefer backward-compatible changes by default.
- Version contracts, not just code.
- Keep roll-forward and rollback paths explicit.
- Use dry-run checks before mutating operations.

## 1) Data Contract Versioning

When payload shape changes:

1. Add new fields as optional first.
2. Keep existing readers tolerant of missing new fields.
3. Only remove fields after docs + runbooks are updated and old artifacts are retired.

Recommended compatibility approach:

- writer: can emit new fields
- reader: must not fail when field is absent

## 2) Identity Key Versioning

Point IDs are deterministic and computed from canonical identity input.  
If canonical key format changes, treat it as a contract change.

Rules:

- preserve deterministic ID generation within a given contract
- document canonical key format in docs
- define cutover plan before changing identity logic

Current canonical identity input:

- `source`
- `document_id`
- `chunk_id`

## 3) CLI and Script Versioning

For `app/*.py` and `scripts/*.sh` changes:

- additive flags are preferred over breaking flag renames
- if behavior changes materially, document before/after examples
- keep defaults safe (`dry-run` where possible for destructive flows)

## 4) Workflow/Operations Versioning

For `.github/workflows` and operational commands:

- require explicit apply confirmations for mutating paths
- keep environment-gated approvals for apply jobs
- store artifacts for traceability (`manifest`, `stale_candidates`, `delete_actions`, `rollback_actions`)

## 5) Documentation Version Discipline

When behavior changes, update in same change set:

- `README.md`
- `docs/design.md`
- `docs/plan.md`
- runbook file (`data1.md`)

Avoid “code says one thing, docs say another.”

## 6) Change Categories

Use this simple classification:

- **Patch**: bug fix, no contract change.
- **Minor**: additive fields/flags, backward compatible.
- **Major**: identity/schema/behavior changes requiring migration or cutover.

## 7) Release Checklist

Before applying to production-like collections:

1. Run syntax/lint checks.
2. Run ingest dry-run and reconcile dry-run.
3. Verify manifest output and stale preview.
4. Confirm rollback path (snapshot + action reports).
5. Update docs/runbooks.

## 8) Recommended Naming

For run artifacts:

- `ingest_manifest_<run_id>.json` (immutable run artifact)
- `ingest_manifest_latest.json` (operational pointer)
- `stale_candidates_<run_id>.json`
- `delete_actions_<run_id>.json`
- `rollback_actions_<run_id>.json`

## 9) Anti-Patterns

- changing identity key logic without a cutover plan
- mutating live data without dry-run preview
- removing fields immediately without compatibility window
- shipping behavior changes without runbook/doc updates

## 10) RAG Version Contract (Document/Chunk/Embedding)

Treat these three axes as independently versioned:

- `document_version`: raw source revision for a logical document
- `chunk_version`: chunking strategy/version (split policy, thresholds, parser behavior)
- `embedding_version`: embedding model/config version

Recommended payload keys (per point):

- `source`
- `document_id`
- `chunk_id`
- `document_version`
- `chunk_version`
- `embedding_version`
- `ingest_run_id`
- `lifecycle_status` (`active|deleted`)
- `deleted_at`
- `deleted_by_run_id`

### Identity Strategy

Use deterministic UUID5 IDs with explicit identity schema version, and include version axes when you want immutable coexistence across generations.

Example canonical string:

`idv=v3|source=<source>|document_id=<document_id>|document_version=<document_version>|chunk_version=<chunk_version>|embedding_version=<embedding_version>|chunk_id=<chunk_id>`

Rules:

- same canonical string => same ID (idempotent reruns)
- changing any version axis should produce a different ID when immutable coexistence is desired
- never silently reuse IDs across incompatible vector spaces

### Query Routing Contract

Retrieval must target an explicit active version set, not "latest by accident".

Recommended retrieval filters include:

- `source` scope
- `embedding_version` (required)
- optionally `chunk_version` and/or `document_version` during migration windows
- `lifecycle_status=active`

## 11) Cutover Playbook (prepare -> upsert -> reconcile -> rollback)

Use this sequence for safe production updates:

1. **Prepare new version artifacts**
   - build points with new version tags
   - write immutable `ingest_manifest_<run_id>.json`
2. **Upsert new version**
   - upsert without deleting old version
   - run smoke validation against new version filter
3. **Switch retrieval**
   - update query filter/alias to new version
   - monitor quality/latency/error metrics
4. **Reconcile old active points**
   - run reconcile dry-run first
   - apply soft-delete for superseded points after review
5. **Retention purge**
   - hard-delete tombstones only after retention period
6. **Rollback readiness**
   - keep rollback action path validated for target `ingest_run_id`

## 12) Immutability Policy

Operational policy for this repo:

- do not blindly overwrite embeddings in-place for active versions
- ingest new versions as additive writes
- perform explicit cutover and lifecycle cleanup after validation
