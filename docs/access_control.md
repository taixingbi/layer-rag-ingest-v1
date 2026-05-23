# Access control (ACL)

During **prepare payloads**, optional policies from `access_control.json` are attached to each point as `payload.access`. Downstream **retrieval and filtering** (who may see which chunks) is enforced by the RAG gateway or other services that read those fields; this repo only **embeds** the policy on vectors.

## Requester roles (reference matrix)

For documents whose ACL lists the roles below, treat these **requester** identities as allowed to access the content (when your gateway implements the same role names).

| Requester | Allowed |
|-----------|---------|
| public | ✅ |
| hr | ✅ |
| recruiter | ✅ |
| engineer | ✅ |
| admin | ✅ |

Adjust the `roles` arrays in `access_control.json` per document or source if a subset should apply (for example, internal-only content might omit `public`).

## Where the file lives

Paths are **per environment** under `data1`:

| Environment | data1 (personal) |
|-------------|------------------|
| dev | `data_dev/data1/raw/access_control.json` |
| qa | `data_qa/data1/raw/access_control.json` |
| prod | `data_prod/data1/raw/access_control.json` |

[`scripts/data1.sh`](../scripts/data1.sh) passes the matching file explicitly: `--access-control-file "$DATASET_ROOT/raw/access_control.json"` with `DATASET_ROOT="${DATA_ROOT}/data1"`.

If you omit `--access-control-file`, [`app/prepare_payloads.py`](../app/prepare_payloads.py) still resolves `<dataset-root>/raw/access_control.json` from `--data-dir` (the dataset root is the parent of `processed/`).

## JSON shape

Top level: object whose keys are **lookup keys**; values are **policy objects**.

Policy object fields (all optional lists of strings; empty lists are omitted when normalized):

- **`roles`** — logical roles (e.g. `hr`, `engineer`, `admin`, `public`).
- **`groups`** — group identifiers.
- **`teams`** — team identifiers.

Example (data1-style sources with `personal` prefix):

```json
{
  "personal_profile": {
    "roles": ["admin", "hr", "recruiter", "engineer", "public"],
    "groups": ["engineering"],
    "teams": ["rag-platform"]
  }
}
```

## Lookup order

For each chunk, [`_resolve_access_policy`](../app/prepare_payloads.py) tries, in order:

1. `{source}:{document_id}`
2. `{source}` only
3. `{document_id}` only

The first key that exists in the map wins. If none match, no `payload.access` is set for that point.

## Operational notes

- Keep **dev / qa / prod** files in sync only when you intend the same policy; copying whole `data_dev` trees onto `data_qa` will overwrite that env’s `access_control.json`.
- Use a **distinct Qdrant collection** per environment (`COLLECTION_NAME` + `ENV` in `.env.*`) so dev ingest does not overwrite production vectors.
- After changing `access_control.json`, re-run **chunks → prepare_payloads** (and upsert) so points pick up the new `payload.access`.
