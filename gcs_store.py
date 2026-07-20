"""
gcs_store.py — optional Google Cloud Storage source for deed data.

If these environment variables are set, the app reads deeds directly from the
bucket instead of the local data/ folder:

    GCS_CREDENTIALS_JSON   full contents of the service-account key JSON
    GCS_BUCKET             e.g. classification-vision
    GCS_PREFIX             e.g. ocr_outputs/orissa_deeds/sample_1000

If they are not set, everything falls back to the local data/ folder and the
app behaves exactly as before. PDFs are streamed from GCS on demand and cached
on local disk (the cache is just a cache — losing it on restart is fine).
"""

import json
import os
from pathlib import Path

_client = None
_bucket = None


def enabled():
    return bool(os.environ.get("GCS_BUCKET"))


def _get_bucket():
    global _client, _bucket
    if _bucket is not None:
        return _bucket
    from google.cloud import storage
    from google.oauth2 import service_account
    creds_json = os.environ.get("GCS_CREDENTIALS_JSON")
    if creds_json:
        info = json.loads(creds_json)
        creds = service_account.Credentials.from_service_account_info(info)
        _client = storage.Client(credentials=creds, project=info.get("project_id"))
    else:
        _client = storage.Client()  # ambient credentials (e.g. on GCP)
    _bucket = _client.bucket(os.environ["GCS_BUCKET"])
    return _bucket


def _prefix():
    p = os.environ.get("GCS_PREFIX", "").strip("/")
    return p + "/" if p else ""


def list_deed_ids():
    """Enumerate deed folders. Prefers index.csv (needs only objects.get);
    falls back to listing the bucket (needs objects.list)."""
    ids = _ids_from_index()
    if ids:
        return ids
    bucket = _get_bucket()
    prefix = _prefix()
    out = set()
    it = bucket.list_blobs(prefix=prefix, delimiter="/")
    for _ in it:            # must consume pages for prefixes to populate
        pass
    for p in it.prefixes:
        name = p[len(prefix):].strip("/")
        if name:
            out.add(name)
    return sorted(out)


def _ids_from_index():
    """Read reg_nos from index.csv in the bucket, if readable."""
    import csv
    import io
    try:
        raw = read_text("index.csv")
    except Exception:
        return None
    if not raw:
        return None
    try:
        rows = list(csv.DictReader(io.StringIO(raw)))
        ids = [str(r["reg_no"]).strip() for r in rows if r.get("reg_no")]
        return sorted(set(ids)) or None
    except Exception:
        return None


def read_text(rel_path):
    """Read a text object under the prefix; None if missing."""
    bucket = _get_bucket()
    blob = bucket.blob(_prefix() + rel_path)
    if not blob.exists():
        return None
    return blob.download_as_text()


def fetch_pdf(reg_no, cache_dir="static/scans"):
    """Return a local path to <reg_no>.pdf, downloading from GCS if needed."""
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    local = cache / f"{reg_no}.pdf"
    if local.exists() and local.stat().st_size > 0:
        return local
    bucket = _get_bucket()
    blob = bucket.blob(f"{_prefix()}{reg_no}/{reg_no}.pdf")
    if not blob.exists():
        return None
    blob.download_to_filename(str(local))
    return local
