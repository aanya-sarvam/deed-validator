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
import re
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


_PAGE_NUM_RE = re.compile(r"(\d+)(?=\.\w+$)")


def _pages_via_probe(reg_no, max_pages=60, max_consecutive_misses=5):
    """Fallback page-discovery method for when 'list' permission isn't
    available (confirmed to be the case here): guess sequential filenames
    matching the confirmed page_NNN.jpg naming convention
    ({prefix}/grounding/images/{reg_no}/page_NNN.jpg, 3-digit zero-padded,
    1-indexed) and check each one's existence individually.
    blob.exists() is a metadata check on a KNOWN path — it only needs
    'storage.objects.get' (read), not 'storage.objects.list' — so this
    finds every real page file without ever listing the folder. This is
    ground truth the same way listing would be: it doesn't matter whether
    ocr_dataset.jsonl has an entry for a page or not, only whether the file
    is actually there.

    Stops after several consecutive misses (so a short document doesn't
    cost 60 checks), but tolerates a few isolated gaps rather than stopping
    at the very first missing page number, in case a page was skipped or
    renamed."""
    bucket = _get_bucket()
    for prefix in raw_prefixes():
        found = []
        misses = 0
        pg = 1
        while pg <= max_pages and misses < max_consecutive_misses:
            rel = f"grounding/images/{reg_no}/page_{pg:03d}.jpg"
            try:
                exists = bucket.blob(f"{prefix}/{rel}").exists()
            except Exception as e:
                print(f"[gcs-raw] probe failed {prefix}/{rel}: {e}")
                break
            if exists:
                found.append([pg, prefix, rel])
                misses = 0
            else:
                misses += 1
            pg += 1
        if found:
            return {"pages": found}
    return None


def _pages_via_listing(reg_no):
    """Primary page-discovery method: list the reg_no's image folder
    directly (confirmed layout: {prefix}/grounding/images/{reg_no}/page_NNN.jpg)
    instead of trusting ocr_dataset.jsonl to enumerate pages. This is the
    ground truth — it doesn't matter whether OCR ran on every page or
    whether the JSONL has an entry for each one; if the file is sitting in
    the folder, it counts. Needs 'storage.objects.list' on the service
    account; returns None (not an error) if that's unavailable or the
    folder doesn't exist under this prefix, so the caller can fall back to
    the JSONL-based scan."""
    bucket = _get_bucket()
    combined = {}
    any_listable = False
    for prefix in raw_prefixes():
        folder = f"{prefix}/grounding/images/{reg_no}/"
        try:
            blobs = list(bucket.list_blobs(prefix=folder))
        except Exception as e:
            print(f"[gcs-raw] could not list {folder}: {e}")
            continue
        any_listable = True
        for b in blobs:
            name = b.name[len(folder):]
            if not name or name.endswith("/"):
                continue
            m = _PAGE_NUM_RE.search(name)
            if not m:
                continue
            pg = int(m.group(1))
            rel = f"grounding/images/{reg_no}/{name}"
            if pg not in combined:
                combined[pg] = (prefix, rel)
    if not any_listable or not combined:
        return None
    pages = sorted(
        ([pg, prefix, rel] for pg, (prefix, rel) in combined.items()),
        key=lambda x: x[0])
    return {"pages": pages}


def _pages_for_reg_no(reg_no):
    """Find pages:[[page, prefix, image_rel_path], ...] for one reg_no.
    Three-tier fallback, each one ground-truth (independent of whether
    ocr_dataset.jsonl has an entry for a given page), tried in order:
      1. _pages_via_listing  — list the folder directly (needs 'list'
         permission on the service account).
      2. _pages_via_probe    — no 'list' needed: guess sequential
         page_NNN.jpg filenames and check each one's existence
         individually. This is what actually runs in this deployment,
         since the service account here only has read/get, not list.
      3. JSONL scan (below)  — last resort if neither of the above finds
         anything (e.g. a batch whose images live under a different path
         than the confirmed grounding/images/<reg_no>/ convention).
    Either way, never holds a whole dataset (or any other deed's entries)
    in memory.

    Why fall back to the JSONL scan at all, instead of only listing: some
    deployments' service accounts may only have read/get permission (not
    list), or the images for a given batch might live under a different
    path than the confirmed grounding/images/<reg_no>/ layout — in either
    case listing silently returns nothing useful and we still want an
    answer from the JSONL rather than reporting no pages at all.

    IMPORTANT (JSONL fallback only): this checks ALL configured raw
    prefixes and merges whatever pages each one has for this reg_no — it
    does NOT stop at the first prefix with a match. A deed's pages can be
    split across more than one ingested batch (e.g. an earlier ~5k-doc
    batch and a later ~5k-doc batch both containing entries for
    overlapping reg_nos, each with a different subset of that deed's
    pages). An earlier version returned as soon as it found ANY pages in
    the first prefix it checked, which silently served a partial page set
    (e.g. 3 of 6, or 6 of 10) with no error at all — indistinguishable from
    a correctly-loaded short document. But even merging all prefixes, the
    JSONL scan can still under-report if the JSONL itself doesn't have an
    entry for every image file that actually exists — which is exactly
    why the folder listing above is tried first.

    This also replaces an earlier design that built ONE big reg_no -> pages
    dict for the entire raw dataset and kept it resident for the process's
    lifetime, cached to a JSON file on local disk to avoid rebuilding it
    every time. That disk cache lives on Render's local filesystem, which is
    wiped on every deploy — so after any redeploy, the very first raw PDF
    view had to re-download and json-parse the *entire* ocr_dataset.jsonl
    (across every raw prefix) into memory in one shot before it could look
    up a single deed. With ~10k+ documents behind it, that one-time rebuild
    was almost certainly a trigger for earlier OOM restarts. Streaming +
    caching only the tiny per-deed result keeps peak memory to roughly this
    one deed's page list, regardless of dataset size, at the cost of a
    network scan (now across all prefixes) on a not-yet-cached deed's first
    view."""
    cache_file = Path("static/.raw_pages") / f"{reg_no}.json"
    if cache_file.exists():
        try:
            return json.loads(cache_file.read_text())
        except Exception:
            pass

    listed = _pages_via_listing(reg_no)
    if listed:
        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(listed))
        except Exception:
            pass
        return listed

    probed = _pages_via_probe(reg_no)
    if probed:
        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(probed))
        except Exception:
            pass
        return probed

    bucket = _get_bucket()
    combined = {}   # page_num -> (prefix, image_rel_path); first prefix to
                    # report a given page number wins if it's ever duplicated
    for prefix in raw_prefixes():
        blob = bucket.blob(f"{prefix}/ocr/ocr_dataset.jsonl")
        if not blob.exists():
            continue
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
                        pg = o.get("page", 0)
                        if pg not in combined:
                            combined[pg] = (prefix, o.get("image"))
        except Exception as e:
            print(f"[gcs-raw] scan failed for {prefix}: {e}")
            continue
    if not combined:
        return None
    pages = sorted(
        ([pg, prefix, rel] for pg, (prefix, rel) in combined.items()),
        key=lambda x: x[0] or 0)
    result = {"pages": pages}
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(result))
    except Exception:
        pass
    return result


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
    match = next(((prefix, rel) for pg, prefix, rel in entry["pages"]
                  if str(pg) == str(page_num)), None)
    if not match:
        return None, None
    prefix, rel = match
    cache = Path(cache_dir) / reg_no
    ext = Path(rel).suffix or ".jpg"
    local = cache / f"{page_num}{ext}"
    if local.exists() and local.stat().st_size > 0:
        data = local.read_bytes()
        return data, _content_type(data[:8])
    bucket = _get_bucket()
    blob = bucket.blob(f"{prefix}/{rel}")
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
