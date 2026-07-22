# Deed Validation Tool

Split-pane interface for language experts to validate OCR-digitised registry
deeds: scanned PDF on the right, editable English metadata on the left.
PostgreSQL-backed, with search-first navigation, per-record locking with
reviewer indicators, admin-managed accounts, and export of the corrected
dataset in the original input Excel layout.

## Quick start / local demo (Docker — recommended)

```bash
DB_PASSWORD=demo docker compose up -d --build        # macOS/Linux
# Windows PowerShell:  $env:DB_PASSWORD="demo"; docker compose up -d --build
```
Open http://localhost:8000 — the sample data auto-ingests on first boot
(watch it: `docker compose logs -f app`). Requires Docker Desktop only.
Stop with `docker compose down` (data persists); full reset:
`docker compose down -v`.

For later batches: drop files into `data/` and restart, or run
`docker compose exec app python ingest.py "data/<file>.xlsx" data/<folder>`.
Put nginx + TLS in front for production.

## Quick start (bare)

```bash
pip install -r requirements.txt
export DATABASE_URL=postgresql://deeds:deeds@localhost:5432/deeds
python ingest.py "data/data-85-94/DEED 85-94.xlsx" data/data-85-94
uvicorn app:app --port 8000
```

**Seed logins** (password `sarvam123` — rotate immediately):
`expert1`–`expert3` (expert), `admin` (admin: Progress, Accounts, export).

## Loading documents

`python ingest.py <metadata.xlsx> <folder-with-pdfs>`

- One `documents` row per Excel row; each metadata value becomes an editable
  `fields` row (compound party strings are parsed into per-party
  Name / Relation / Relation name / Address).
- PDFs are matched by `R0xx_{regno}_{year}_{book}` in the filename (tolerates
  +-1 year drift between execution and registration year) and copied to
  `static/scans/`.
- Re-runnable: existing deed numbers are skipped, so ingesting a new batch is
  just running the command again with the new files.

### Adding the raw orissa_deeds export (read-only GCS access)

`grounding/grounding_good_partial.jsonl` + `ocr/ocr_dataset.jsonl`, both
under `GCS_RAW_PREFIX` (default `ocr_outputs/orissa_deeds`, same bucket as
`GCS_BUCKET`), are read directly — no download, no local conversion step,
and no write access to the bucket needed. `ingest_gcs_raw()` in
`ingest_json.py` loads deed metadata + OCR text into Postgres immediately;
each deed's scan PDF is stitched from its raw page images the first time
someone opens it (`gcs_store.fetch_or_build_pdf`) and cached locally after
that, the same lazy pattern already used for pre-made `sample_1000` PDFs.

To load a new batch: Admin → Reingest (or `POST /api/admin/reingest`), or
locally: `python -c "from ingest_json import ingest_gcs_raw; ingest_gcs_raw()"`.
Safe to re-run — existing deed_numbers are skipped everywhere, so nothing
is ever reset or duplicated.

If you don't have GCS read access either and are working from local copies
of the dataset files, `convert_orissa_raw.py <orissa_deeds_dir> data/batch_2`
reshapes them into per-deed folders (`grounding.json` + `ocr.jsonl` +
`<reg_no>.pdf`, PDFs built up front instead of on first view) that
`ingest_json.py data/batch_2` or `ingest_dir()` can load the normal way.

## Extracting corrected data

Admin -> Progress -> "Download corrected dataset", or:
```bash
curl -o corrected.xlsx "https://<host>/api/export?token=<admin-token>"
# or offline, straight from the DB:
python export.py
```
The workbook mirrors the input: one sheet per book (BOOK-1, BOOK_3, BOOK_4)
with the original column names/order and party-details strings reassembled
from the corrected per-party fields. A VALIDATION_STATUS column is appended
(filter to `validated` for finished rows), and an Audit sheet lists every
corrected field with its OCR value, corrected value, and who validated.

## Authentication

- Admin creates accounts (Accounts tab); no self-signup.
- Passwords hashed with bcrypt; login issues an opaque session token stored
  in the `sessions` table, expiring after 12 h (SESSION_HOURS in db.py).
- Every request resolves token -> user server-side; expert vs admin is
  enforced in the API, not the UI.
- For production add: HTTPS (nginx), rate limiting on /api/login, and a
  password-change endpoint. If the org later wants SSO, swap the login
  endpoint for OIDC — the session table and the rest of the app are unchanged.

## Concurrency & persistence

- Opening a record locks it (locked_by/locked_at); the search table shows
  who holds it. Locks expire after 30 min of inactivity (editing refreshes).
- Queue claims use FOR UPDATE SKIP LOCKED — safe under any number of
  simultaneous experts.
- ocr_value is immutable; corrections go to current_value; every change
  appends to edit_log (old, new, who, when). Autosave fires on field blur.

## Production checklist

- nginx in front: TLS termination, serve static/, proxy /api to the app.
- Nightly pg_dump + rsync of static/scans/ to a second disk.
- Rotate seed passwords; set a strong DB_PASSWORD.
- Add /api/health monitoring and log rotation.
- At full-corpus scale: ingestion reconciliation report (unmatched PDFs/rows)
  and Postgres full-text search if party-name search slows down.
