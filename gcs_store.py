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


# ---------------------------------------------------------------------------
# Raw orissa_deeds dataset (grounding_good_partial.jsonl + ocr_dataset.jsonl +
# per-page images), read-only. Unlike sample_1000, deeds here have no
# pre-made <reg_no>.pdf in the bucket — only individual page images — so a
# PDF is stitched on first view and cached locally, same lazy pattern as
# fetch_pdf() above. Only ever needs object READ, never LIST or WRITE.
# ---------------------------------------------------------------------------

def _raw_prefix():
    return os.environ.get("GCS_RAW_PREFIX", "ocr_outputs/orissa_deeds").strip("/")


def raw_prefixes():
    """Bucket-root-relative prefixes for the raw orissa_deeds export.
    GCS_RAW_PREFIXES (comma-separated) takes precedence; falls back to
    GCS_RAW_PREFIX for backward compatibility."""
    multi = os.environ.get("GCS_RAW_PREFIXES", "").strip()
    if multi:
        return [p.strip().strip("/") for p in multi.split(",") if p.strip()]
    return [_raw_prefix()]


def blob_stat(abs_path):
    """Cheap existence + size check (HEAD only, no download)."""
    bucket = _get_bucket()
    blob = bucket.blob(abs_path)
    if not blob.exists():
        return {"exists": False, "size_bytes": None}
    blob.reload()
    return {"exists": True, "size_bytes": blob.size}


def read_text_abs(abs_path):
    """Read a text object by bucket-root-relative path (ignores GCS_PREFIX,
    unlike read_text() above). None if missing."""
    bucket = _get_bucket()
    blob = bucket.blob(abs_path)
    if not blob.exists():
        return None
    return blob.download_as_text()


def _pages_for_reg_no(reg_no):
    """Find {prefix, pages:[[page, image_rel_path], ...]} for one reg_no by
    streaming ocr/ocr_dataset.jsonl line-by-line, without ever holding the
    whole dataset (or any other deed's entries) in memory.

    This replaces an earlier design that built ONE big reg_no -> pages dict
    for the entire raw dataset and kept it resident for the process's
    lifetime, cached to a JSON file on local disk to avoid rebuilding it
    every time. That disk cache lives on Render's local filesystem, which is
    wiped on every deploy — so after any redeploy, the very first raw PDF
    view had to re-download and json-parse the *entire* ocr_dataset.jsonl
    (across every raw prefix) into memory in one shot before it could look
    up a single deed. With ~10k+ documents behind it, that one-time rebuild
    was almost certainly the actual trigger for the OOM restarts, separate
    from (and larger than) the per-document image-stitch fix already
    shipped. Streaming + caching only the tiny per-deed result keeps peak
    memory to roughly this one deed's page list, regardless of dataset size,
    at the cost of a network scan on a not-yet-cached deed's first view."""
    cache_file = Path("static/.raw_pages") / f"{reg_no}.json"
    if cache_file.exists():
        try:
            return json.loads(cache_file.read_text())
        except Exception:
            pass
    bucket = _get_bucket()
    for prefix in raw_prefixes():
        blob = bucket.blob(f"{prefix}/ocr/ocr_dataset.jsonl")
        if not blob.exists():
            continue
        pages = []
        try:
            with blob.open("rt") as fh:   # streams from GCS; no full-text buffer
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        o = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if str(o.get("reg_no") or "") == reg_no and o.get("image"):
                        pages.append([o.get("page", 0), o.get("image")])
        except Exception as e:
            print(f"[gcs-raw] scan failed for {prefix}: {e}")
            continue
        if pages:
            pages.sort(key=lambda x: x[0] or 0)
            result = {"prefix": prefix, "pages": pages}
            try:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                cache_file.write_text(json.dumps(result))
            except Exception:
                pass
            return result
    return None


def premade_pdf_exists(reg_no):
    """Cheap existence check (no download) for a pre-made <reg_no>.pdf."""
    bucket = _get_bucket()
    blob = bucket.blob(f"{_prefix()}{reg_no}/{reg_no}.pdf")
    return blob.exists()


def pages_entry(reg_no):
    """Public wrapper around the per-deed page lookup, for callers that just
    need to know whether/how many raw pages a deed has."""
    return _pages_for_reg_no(reg_no)


def fetch_page_image(reg_no, page_num, cache_dir="static/scans/pages"):
    """Return (bytes, content_type) for ONE page image of reg_no. Caches
    that single page to local disk so re-viewing the same deed doesn't
    re-hit GCS. Never loads any other page or deed into memory — this is
    what replaced the old build-a-whole-PDF-per-deed approach: the frontend
    now requests pages one at a time and displays them as a sequence of
    images instead of a stitched document, so peak memory here is just this
    one page's bytes, regardless of how many pages the deed has."""
    entry = _pages_for_reg_no(reg_no)
    if not entry:
        return None, None
    match = next((rel for pg, rel in entry["pages"] if str(pg) == str(page_num)), None)
    if not match:
        return None, None
    cache = Path(cache_dir) / reg_no
    ext = Path(match).suffix or ".jpg"
    local = cache / f"{page_num}{ext}"
    if local.exists() and local.stat().st_size > 0:
        data = local.read_bytes()
        return data, _content_type(data[:8])
    bucket = _get_bucket()
    blob = bucket.blob(f"{entry['prefix']}/{match}")
    if not blob.exists():
        return None, None
    data = blob.download_as_bytes()
    try:
        cache.mkdir(parents=True, exist_ok=True)
        local.write_bytes(data)
    except Exception:
        pass
    return data, _content_type(data[:8])


def _content_type(head):
    if head[:2] == b"\xff\xd8":
        return "image/jpeg"
    if head[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    return "application/octet-stream"
