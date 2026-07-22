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


_raw_page_index = None


def _normalize_page_index(data):
    """Accept legacy reg_no -> [[page, path], ...] caches from single-prefix
    deployments and normalize to reg_no -> {prefix, pages}."""
    if not data:
        return data
    sample = next(iter(data.values()), None)
    if sample and isinstance(sample, list):
        pfx = _raw_prefix()
        return {k: {"prefix": pfx, "pages": v} for k, v in data.items()}
    return data


def _load_raw_page_index():
    """reg_no -> {prefix, pages:[[page, image_rel_path], ...]}, built once
    from ocr/ocr_dataset.jsonl under every raw prefix and cached (in memory
    + on local disk, since the source files are large and re-downloading on
    every cold start / PDF view would be wasteful). First prefix wins when
    a reg_no appears in more than one dataset."""
    global _raw_page_index
    if _raw_page_index is not None:
        return _raw_page_index
    cache_file = Path("static/.raw_page_index.json")
    if cache_file.exists():
        try:
            _raw_page_index = _normalize_page_index(json.loads(cache_file.read_text()))
            return _raw_page_index
        except Exception:
            pass
    index = {}
    for prefix in raw_prefixes():
        raw = read_text_abs(f"{prefix}/ocr/ocr_dataset.jsonl")
        if not raw:
            continue
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            reg_no = str(o.get("reg_no") or "")
            img = o.get("image")
            if reg_no and img:
                if reg_no not in index:
                    index[reg_no] = {"prefix": prefix, "pages": []}
                elif index[reg_no]["prefix"] != prefix:
                    continue
                index[reg_no]["pages"].append([o.get("page", 0), img])
    for entry in index.values():
        entry["pages"].sort(key=lambda x: x[0] or 0)
    _raw_page_index = index
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(index))
    except Exception:
        pass
    return index


def fetch_or_build_pdf(reg_no, cache_dir="static/scans"):
    """Return a local PDF for reg_no: prefer a pre-made <reg_no>.pdf
    (sample_1000-style batches, via fetch_pdf), else stitch one from that
    deed's raw page images. None if neither source has anything."""
    p = fetch_pdf(reg_no, cache_dir)
    if p:
        return p
    entry = _load_raw_page_index().get(reg_no)
    if not entry:
        return None
    from PIL import Image
    import io
    bucket = _get_bucket()
    prefix = entry["prefix"]
    pages = entry["pages"]
    imgs = []
    for _page_no, rel in pages:
        try:
            blob = bucket.blob(f"{prefix}/{rel}")
            if not blob.exists():
                continue
            data = blob.download_as_bytes()
            imgs.append(Image.open(io.BytesIO(data)).convert("RGB"))
        except Exception as e:
            print(f"[gcs-raw] page fetch failed {reg_no}/{rel}: {e}")
    if not imgs:
        return None
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    local = cache / f"{reg_no}.pdf"
    imgs[0].save(local, save_all=True, append_images=imgs[1:])
    return local
