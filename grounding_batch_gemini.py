#!/usr/bin/env python3
"""Run the Odia grounding/transcription prompt over deed page images via the
Gemini Batch API (Vertex AI).

For each deed it sends that deed's page images plus a per-deed prompt built from
`prompt.py` (build_user_prompt): the prompt injects the deed's known metadata
values as "targets" to LOCATE on the page and TRANSCRIBE verbatim in the source
script. The model returns a JSON array of FieldResult objects (one per target).

Inputs
  * pages_manifest.csv       (from pdf_to_png.py): reg_no -> page_dir, n_pages
  * deed_metadata_sample.csv (from backfill_subset): per-deed tabular metadata
  * <reg_no>.json  (optional, next to the PDF): full API response with the rich
                    seller/buyer/property blobs -- used for extra name/place targets

Outputs (under --out-dir)
  * batch_input_<tag>.jsonl  : the request JSONL (built on local SSD, uploaded to GCS)
  * batch_meta_<tag>.jsonl   : sidecar mapping key -> deed + targets (for merge)
  * grounding_results.jsonl  : one row per deed with the parsed FieldResult list
  * grounding_fields.csv     : flattened one-row-per-field view

Usage
  python3 grounding_batch_gemini.py --n 20 --dry-run     # build JSONL only
  python3 grounding_batch_gemini.py --n 20               # small test submit
  python3 grounding_batch_gemini.py --all                # full run
  python3 grounding_batch_gemini.py --poll projects/.../batchPredictionJobs/123
  python3 grounding_batch_gemini.py --all --model gemini-3.1-flash
"""
import argparse
import base64
import csv
import html
import io
import json
import os
import re
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from PIL import Image
Image.MAX_IMAGE_PIXELS = None

# prompt.py lives next to this script.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prompt as grounding_prompt  # noqa: E402

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
GCP_PROJECT = "vision-projects-463307"
GCP_LOCATION = "global"
GCS_BUCKET = "classification-vision"
DEFAULT_MODEL = "gemini-3.1-pro-preview"
DEFAULT_THINKING = "medium"   # Gemini 3.x thinking level: low | medium | high ('' off)

BASE = Path("/home/suhani_sarvam_ai/home/orissa-data-gen")
PAGES_MANIFEST = BASE / "pages_manifest.csv"
SELECTION_CSV = BASE / "sample_3k_selection.csv"
DEED_METADATA = Path("/home/suhani_sarvam_ai/orissa-bulk/out/deed_metadata_sample.csv")
OUTPUT_DIR = BASE / "batch_grounding"
# The image-embedded input JSONL is large & transient -> build it on roomy SSD.
WORK_DIR = Path("/mnt/localssd/orissa-data-gen-work/grounding")

MAX_DIMENSION = 1536   # keep fine Odia glyphs legible (vs 768 for classification)
JPEG_QUALITY = 85

PAGES_PER_CALL = 3     # <= this many page images per API request (a deed splits into chunks)
MAX_DEED_PAGES = 20    # deeds with MORE than this many pages are skipped entirely

# Which tabular metadata columns become grounding targets, and their "type" hint.
# type is a free-form hint passed to the model (see prompt.py HOW TO MATCH).
TABULAR_TARGETS = [
    ("deed_type", "Deed type", "deed_type"),
    ("district", "District", "place"),
    ("office", "Registration office", "place"),
    ("registration_date", "Registration date", "date"),
    ("presentation_date", "Presentation date", "date"),
    ("consideration_amount", "Consideration amount", "amount"),
    ("old_reg_no", "Old registration no", "number"),
]

_GENERIC = {"", "-", "na", "n/a", "nil", "none", "null", "others", "0"}


# ----------------------------------------------------------------------------
# Image helpers
# ----------------------------------------------------------------------------
def resize_image(image_path: Path) -> bytes:
    img = Image.open(image_path)
    w, h = img.size
    if max(w, h) > MAX_DIMENSION:
        if w >= h:
            img = img.resize((MAX_DIMENSION, int(h * MAX_DIMENSION / w)), Image.LANCZOS)
        else:
            img = img.resize((int(w * MAX_DIMENSION / h), MAX_DIMENSION), Image.LANCZOS)
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY)
    return buf.getvalue()


def page_files(page_dir: str, max_pages: int) -> list[Path]:
    pages = sorted(Path(page_dir).glob("page_*.png"))
    if max_pages and len(pages) > max_pages:
        pages = pages[:max_pages]
    return pages


def _page_num(path: Path) -> int:
    """page_007.png -> 7."""
    stem = path.stem.split("_")[-1]
    return int(stem) if stem.isdigit() else 0


def chunk(seq: list, size: int):
    """Yield (chunk_index, items) in fixed-size groups."""
    for ci, start in enumerate(range(0, len(seq), size)):
        yield ci, seq[start:start + size]


# ----------------------------------------------------------------------------
# Metadata -> targets
# ----------------------------------------------------------------------------
def _is_generic(v: str) -> bool:
    return str(v).strip().lower() in _GENERIC


def targets_from_tabular(meta: dict) -> list[dict]:
    out = []
    for col, label, typ in TABULAR_TARGETS:
        val = (meta.get(col) or "").strip()
        if _is_generic(val):
            continue
        out.append({"id": col, "label": label, "value": val, "type": typ})
    return out


def _clean_blob(s) -> str:
    """Normalise a semi-structured API string: unescape HTML, collapse whitespace."""
    s = html.unescape("" if s is None else str(s))
    return re.sub(r"\s+", " ", s).strip()


# seller/buyer/property come back as ONE semi-structured string each (not JSON lists).
# We send the whole blob to the model with a "format" hint and let it split + locate +
# transcribe each sub-value (see prompt.py "LIST / COMPOSITE FIELDS").
_COMPOSITE_SPECS = [
    ("seller_details", "Seller details (list of parties)", "party_list", "sellerDetails",
     "numbered parties -> '1-<NAME>  ( RELATION : )<rel>  ( RELATION NAME : )<guardian>  "
     "( ADDRESS : )<addr> ,2-<NAME> ...'; sub-values: name, relation_name, address"),
    ("buyer_details", "Buyer details (list of parties)", "party_list", "buyerDetails",
     "numbered parties -> '1-<NAME>  ( RELATION : )<rel>  ( RELATION NAME : )<guardian>  "
     "( ADDRESS : )<addr> ,2-<NAME> ...'; sub-values: name, relation_name, address"),
    ("property_details", "Property details (list of plots)", "property_list", "propertyDetails",
     "numbered plots -> '1- Village : <v>  Khata : <k> Plot : <p> Area: <a> Total Area: <ta> "
     "Boundary : <b> ,2- ...'; sub-values: village, khata, plot, area"),
]


def targets_from_raw_json(raw) -> list[dict]:
    """Emit up to three COMPOSITE targets (seller/buyer/property) carrying the raw
    semi-structured blob verbatim. The model does the splitting/locating; see
    prompt.py. Returns [] if the deed has no usable party/property text.
    """
    out: list[dict] = []
    if isinstance(raw, list):
        raw = raw[0] if raw else {}
    if not isinstance(raw, dict):
        return out
    for fid, label, typ, key, fmt in _COMPOSITE_SPECS:
        val = _clean_blob(raw.get(key))
        if _is_generic(val):
            continue
        out.append({"id": fid, "label": label, "value": val, "type": typ, "format": fmt})
    return out


def build_targets(meta: dict, raw) -> list[dict]:
    return targets_from_tabular(meta) + (targets_from_raw_json(raw) if raw is not None else [])


# ----------------------------------------------------------------------------
# Request building
# ----------------------------------------------------------------------------
def response_schema() -> dict:
    """Vertex-style response schema mirroring prompt.FieldResult (array of)."""
    return {
        "type": "ARRAY",
        "items": {
            "type": "OBJECT",
            "properties": {
                "id": {"type": "STRING"},
                "item_index": {"type": "INTEGER"},
                "attr": {"type": "STRING"},
                "english_value": {"type": "STRING"},
                "found": {"type": "BOOLEAN"},
                "odia_text": {"type": "STRING"},
                "script": {"type": "STRING"},
                "page": {"type": "INTEGER"},
                "confidence": {"type": "NUMBER"},
                "latin_readback": {"type": "STRING"},
                "notes": {"type": "STRING"},
            },
            "required": ["id", "item_index", "attr", "english_value", "found",
                         "odia_text", "script", "page", "confidence",
                         "latin_readback", "notes"],
        },
    }


def build_request(key: str, image_bytes_list: list[bytes], prompt_text: str,
                  strict_schema: bool, thinking_level: str = "") -> dict:
    parts = []
    for b in image_bytes_list:
        parts.append({"inline_data": {"mime_type": "image/jpeg",
                                      "data": base64.b64encode(b).decode("ascii")}})
    parts.append({"text": prompt_text})
    gen_cfg = {"responseMimeType": "application/json"}
    if strict_schema:
        gen_cfg["responseSchema"] = response_schema()
    if thinking_level:
        gen_cfg["thinkingConfig"] = {"thinkingLevel": thinking_level.upper()}
    return {
        "key": key,
        "request": {
            "contents": [{"parts": parts, "role": "user"}],
            "generationConfig": gen_cfg,
        },
    }


# ----------------------------------------------------------------------------
# Loading inputs
# ----------------------------------------------------------------------------
def load_csv_map(path: Path, key="reg_no") -> dict:
    out = {}
    if not path.exists():
        return out
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[r[key].strip()] = r
    return out


def load_deed_set(path: Path) -> set:
    """One reg_no per line (blank lines / '#' comments ignored)."""
    keep = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                keep.add(s)
    return keep


def raw_json_for(pdf_path: str):
    if not pdf_path:
        return None
    jp = (pdf_path[:-4] if pdf_path.endswith(".pdf") else pdf_path) + ".json"
    if os.path.exists(jp):
        try:
            with open(jp, encoding="utf-8") as f:
                return json.load(f)
        except Exception:  # noqa: BLE001
            return None
    return None


# ----------------------------------------------------------------------------
# Client / poll
# ----------------------------------------------------------------------------
def make_client():
    from google import genai
    print(f"Using Vertex AI (project={GCP_PROJECT}, location={GCP_LOCATION})")
    return genai.Client(vertexai=True, project=GCP_PROJECT, location=GCP_LOCATION)


_DONE_STATES = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED",
                "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}


def poll_job(client, job_name: str, poll_interval: int = 20):
    t0 = time.time()
    while True:
        job = client.batches.get(name=job_name)
        state = job.state.name
        print(f"  [{time.time()-t0:.0f}s] {state}", flush=True)
        if state in _DONE_STATES:
            return job
        time.sleep(poll_interval)


def poll_all(client, job_names: list, poll_interval: int = 30):
    """Poll several jobs until all reach a terminal state; return the job objects."""
    t0 = time.time()
    final = {}
    while len(final) < len(job_names):
        for nm in job_names:
            if nm in final:
                continue
            job = client.batches.get(name=nm)
            if job.state.name in _DONE_STATES:
                final[nm] = job
                print(f"  [{time.time()-t0:.0f}s] {nm.split('/')[-1]} -> {job.state.name} "
                      f"({len(final)}/{len(job_names)} done)", flush=True)
        if len(final) < len(job_names):
            time.sleep(poll_interval)
    return [final[nm] for nm in job_names]


def shard_jsonl(path: Path, per_shard: int, max_bytes: int = 900_000_000) -> list:
    """Split a JSONL into shards capped by request count AND total bytes (<0.9 GB
    to stay under Vertex's 1 GB input limit). Returns the shard file paths."""
    shards, out, count, nbytes, idx = [], None, 0, 0, 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            lb = len(line.encode("utf-8"))
            if out is None or count >= per_shard or (nbytes + lb) > max_bytes:
                if out:
                    out.close()
                sp = path.with_name(f"{path.stem}_s{idx}{path.suffix}")
                out = open(sp, "w", encoding="utf-8")
                shards.append(sp)
                idx += 1
                count = nbytes = 0
            out.write(line)
            count += 1
            nbytes += lb
    if out:
        out.close()
    return shards


def parse_json_array(text: str):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"raw_response": text, "parse_error": True}


# ----------------------------------------------------------------------------
# Per-deed request builder (runs in a worker process; image resize is CPU-bound)
# ----------------------------------------------------------------------------
def _build_deed_requests(task: dict) -> dict:
    """Build all chunk requests for ONE deed. Returns request/meta JSON strings
    (or a skip reason). Designed to run in a ProcessPoolExecutor worker."""
    reg = task["reg"]
    pinfo = task["pinfo"]
    meta = task["meta"]
    o = task["opts"]
    res = {"reg": reg, "requests": [], "metas": [], "n_pages": 0,
           "skip": None, "render_err": 0}

    raw = raw_json_for(task["pdf_path"])
    targets = build_targets(meta, raw)
    if not targets:
        res["skip"] = "no_targets"
        return res
    pfs = page_files(pinfo["page_dir"], o["max_pages"])
    if not pfs:
        res["skip"] = "no_pages"
        return res
    res["n_pages"] = len(pfs)
    if len(pfs) > o["max_deed_pages"]:
        res["skip"] = "too_long"
        return res

    deed_type = meta.get("deed_type") or pinfo.get("book_label", "")
    n_chunks = (len(pfs) + o["pages_per_call"] - 1) // o["pages_per_call"]
    for ci, chunk_pfs in chunk(pfs, o["pages_per_call"]):
        try:
            imgs = [resize_image(p) for p in chunk_pfs]
        except Exception:  # noqa: BLE001
            res["render_err"] += 1
            continue
        page_offset = ci * o["pages_per_call"]
        key = reg if n_chunks == 1 else f"{reg}__c{ci}"
        prompt_text = grounding_prompt.build_user_prompt(
            targets, n_pages=len(imgs), deed_type=deed_type, page_offset=page_offset)
        res["requests"].append(json.dumps(
            build_request(key, imgs, prompt_text, o["strict_schema"], o["thinking_level"])))
        res["metas"].append(json.dumps({
            "key": key,
            "reg_no": reg,
            "book": pinfo.get("book", ""),
            "book_label": pinfo.get("book_label", ""),
            "page_dir": pinfo.get("page_dir", ""),
            "chunk_index": ci,
            "n_chunks": n_chunks,
            "page_offset": page_offset,
            "page_numbers": [_page_num(p) for p in chunk_pfs],
            "pages_sent": len(imgs),
            "deed_type": deed_type,
            "targets": targets,
        }, ensure_ascii=False))
    return res


# ----------------------------------------------------------------------------
# Submit
# ----------------------------------------------------------------------------
def cmd_submit(args):
    pages = load_csv_map(args.pages_manifest)
    metas = load_csv_map(args.metadata)
    selection = load_csv_map(args.selection)
    if not pages:
        sys.exit(f"No pages manifest at {args.pages_manifest} -- run pdf_to_png.py first.")
    if not metas:
        print(f"WARNING: no metadata at {args.metadata} -- targets will be empty until "
              f"backfill_subset has run.", flush=True)

    reg_nos = [r for r in pages if r in metas] if metas else []
    if args.deeds:
        keep = load_deed_set(args.deeds)
        reg_nos = [r for r in reg_nos if r in keep]
        print(f"--deeds filter: {len(keep):,} listed -> {len(reg_nos):,} usable "
              f"(have pages+metadata)", flush=True)
    if args.exclude_deeds:
        drop = load_deed_set(args.exclude_deeds)
        before = len(reg_nos)
        reg_nos = [r for r in reg_nos if r not in drop]
        print(f"--exclude-deeds: {len(drop):,} listed -> dropped {before - len(reg_nos):,}, "
              f"{len(reg_nos):,} remain", flush=True)
    reg_nos.sort()
    if not args.all:
        reg_nos = reg_nos[: args.n]
    tag = args.tag or ("all" if args.all else str(args.n))
    print(f"Deeds with pages+metadata: {len(reg_nos):,} (tag={tag}, model={args.model}, "
          f"thinking={args.thinking_level or 'off'})", flush=True)
    if not reg_nos:
        sys.exit("Nothing to submit (need both rendered pages AND metadata).")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    jsonl_path = WORK_DIR / f"batch_input_{tag}.jsonl"
    meta_path = OUTPUT_DIR / f"batch_meta_{tag}.jsonl"

    st = {"deeds_used": 0, "requests": 0, "skip_no_targets": 0, "skip_no_pages": 0,
          "skip_too_long": 0, "skip_render_err": 0, "pages_total": 0}
    too_long = []   # (reg_no, n_pages) skipped for exceeding MAX_DEED_PAGES
    opts = {"max_pages": args.max_pages, "max_deed_pages": args.max_deed_pages,
            "pages_per_call": args.pages_per_call, "strict_schema": args.strict_schema,
            "thinking_level": args.thinking_level}
    tasks = [{"reg": reg, "pinfo": pages[reg], "meta": metas.get(reg, {}),
              "pdf_path": (selection.get(reg, {}) or {}).get("pdf_path", ""), "opts": opts}
             for reg in reg_nos]

    workers = args.build_workers or min(16, os.cpu_count() or 8)
    print(f"Building requests with {workers} worker processes "
          f"(<= {args.pages_per_call} pages/call, skip > {args.max_deed_pages}p)...", flush=True)
    t0 = time.time()
    with open(jsonl_path, "w") as fj, open(meta_path, "w", encoding="utf-8") as fm:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_build_deed_requests, t): t["reg"] for t in tasks}
            for done, fut in enumerate(as_completed(futs), 1):
                res = fut.result()
                st["skip_render_err"] += res.get("render_err", 0)
                if res["skip"] == "no_targets":
                    st["skip_no_targets"] += 1
                elif res["skip"] == "no_pages":
                    st["skip_no_pages"] += 1
                elif res["skip"] == "too_long":
                    st["skip_too_long"] += 1
                    too_long.append((res["reg"], res["n_pages"]))
                if res["requests"]:
                    fj.write("\n".join(res["requests"]) + "\n")
                    fm.write("\n".join(res["metas"]) + "\n")
                    st["deeds_used"] += 1
                    st["requests"] += len(res["requests"])
                    st["pages_total"] += res["n_pages"]
                if done % 200 == 0 or done == len(tasks):
                    print(f"  [{done}/{len(tasks)}] deeds={st['deeds_used']} "
                          f"requests={st['requests']} {time.time()-t0:.0f}s", flush=True)

    # Persist the skipped-too-long list + a stats sidecar for bookkeeping.
    stats_path = OUTPUT_DIR / f"submit_stats_{tag}.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump({**st, "max_deed_pages": args.max_deed_pages,
                   "pages_per_call": args.pages_per_call,
                   "too_long": [{"reg_no": r, "n_pages": n} for r, n in too_long]},
                  f, ensure_ascii=False, indent=2)

    size_mb = jsonl_path.stat().st_size / 1024 / 1024
    print(f"\nPrepared {st['requests']} requests across {st['deeds_used']} deeds "
          f"({st['pages_total']} page-images) in {time.time()-t0:.0f}s")
    print(f"  skipped: too_long(>{args.max_deed_pages}p)={st['skip_too_long']} "
          f"no_targets={st['skip_no_targets']} no_pages={st['skip_no_pages']} "
          f"render_err={st['skip_render_err']}")
    print(f"JSONL: {jsonl_path} ({size_mb:.1f} MB)\nMeta:  {meta_path}\nStats: {stats_path}")

    if args.dry_run:
        print("\n--dry-run set: files written, NOT submitting.")
        return

    # Vertex batch has a HARD 1 GB limit on the GCS input JSONL, so split into
    # shards that stay safely under it (by request count AND bytes).
    shard_paths = shard_jsonl(jsonl_path, args.shard_size)
    print(f"\nSplit into {len(shard_paths)} shard(s) "
          f"(<= {args.shard_size} requests / <0.9 GB each) to respect the 1 GB input limit.")

    from google.cloud import storage as gcs
    gcs_client = gcs.Client(project=GCP_PROJECT)
    bucket = gcs_client.bucket(GCS_BUCKET)
    client = make_client()
    ts = int(time.time())
    job_names = []
    for si, sp in enumerate(shard_paths):
        blob_name = f"batch_inputs/grounding_{tag}_{ts}_s{si}.jsonl"
        mb = sp.stat().st_size / 1e6
        print(f"Uploading shard {si+1}/{len(shard_paths)} ({mb:.0f} MB) -> "
              f"gs://{GCS_BUCKET}/{blob_name}", flush=True)
        bucket.blob(blob_name).upload_from_filename(str(sp))
        job = client.batches.create(
            model=args.model, src=f"gs://{GCS_BUCKET}/{blob_name}",
            config={"display_name": f"grounding-{tag}-s{si}"})
        print(f"  job: {job.name}", flush=True)
        job_names.append(job.name)
        try:
            sp.unlink()
        except OSError:
            pass
    try:
        jsonl_path.unlink()
    except OSError:
        pass

    jobs_file = OUTPUT_DIR / f"batch_jobs_{tag}.json"
    with open(jobs_file, "w", encoding="utf-8") as f:
        json.dump({"jobs": job_names, "meta": str(meta_path), "tag": tag}, f, indent=2)
    print(f"\n{len(job_names)} job(s) created. Saved -> {jobs_file}")
    print(f"To poll later: python3 grounding_batch_gemini.py --poll {jobs_file}\n\nPolling now...")
    jobs = poll_all(client, job_names)
    retrieve_results(client, jobs, meta_path, tag=tag)


# ----------------------------------------------------------------------------
# Poll / retrieve
# ----------------------------------------------------------------------------
def cmd_poll(args):
    client = make_client()
    p = Path(args.poll)
    meta_path = None
    tag = ""
    if p.exists() and p.suffix == ".json":
        d = json.load(open(p, encoding="utf-8"))
        names = d.get("jobs", [])
        meta_path = Path(d["meta"]) if d.get("meta") else None
        tag = d.get("tag", "")
    else:
        names = [s.strip() for s in args.poll.split(",") if s.strip()]
    if meta_path is None:
        metas = sorted(OUTPUT_DIR.glob("batch_meta_*.jsonl"))
        meta_path = metas[-1] if metas else None
    if not tag and meta_path:
        stem = Path(meta_path).stem
        if stem.startswith("batch_meta_"):
            tag = stem[len("batch_meta_"):]
    print(f"Polling {len(names)} job(s)...")
    jobs = poll_all(client, names)
    retrieve_results(client, jobs, meta_path, tag=tag)


def _download_job_text(client, job) -> str:
    if job.dest and job.dest.gcs_uri:
        from google.cloud import storage as gcs
        gcs_uri = job.dest.gcs_uri.rstrip("/")
        print(f"Downloading results from {gcs_uri}...")
        gcs_client = gcs.Client(project=GCP_PROJECT)
        parts = gcs_uri.replace("gs://", "").split("/", 1)
        bucket = gcs_client.bucket(parts[0])
        prefix = parts[1] + "/" if len(parts) > 1 else ""
        text = ""
        for blob in bucket.list_blobs(prefix=prefix):
            if blob.name.endswith(".jsonl"):
                text += blob.download_as_text() + "\n"
        return text
    if job.dest and job.dest.file_name:
        print(f"Downloading results from {job.dest.file_name}...")
        return client.files.download(file=job.dest.file_name).decode("utf-8")
    return ""


def _accumulate(text, meta, by_deed):
    for line in text.splitlines():
        if not line.strip():
            continue
        parsed = json.loads(line)
        key = parsed.get("key", "")
        cm = meta.get(key, {"key": key})
        reg = cm.get("reg_no") or key.split("__c")[0]
        d = by_deed[reg]
        d["chunks"] += 1
        d.setdefault("book_label", cm.get("book_label", ""))
        d.setdefault("deed_type", cm.get("deed_type", ""))
        d.setdefault("n_chunks", cm.get("n_chunks", 1))
        page_numbers = cm.get("page_numbers", [])
        if parsed.get("response"):
            resp_text = ""
            for cand in parsed["response"].get("candidates", []):
                for part in cand.get("content", {}).get("parts", []):
                    resp_text += part.get("text", "")
            fields = parse_json_array(resp_text)
            if isinstance(fields, list):
                for fld in fields:
                    if not isinstance(fld, dict):
                        continue
                    # Map the model's request-local page index to the real deed page.
                    local = fld.get("page") or 0
                    fld["page_local"] = local
                    if fld.get("found") and 1 <= local <= len(page_numbers):
                        fld["page"] = page_numbers[local - 1]
                    elif not fld.get("found"):
                        fld["page"] = 0
                    d["fields"].append(fld)
            else:
                d["errors"] += 1  # unparseable chunk response
        elif "error" in parsed:
            d["errors"] += 1


_FIELDS_CSV_HEADER = ["reg_no", "book_label", "field_id", "item_index", "attr",
                      "english_value", "found", "odia_text", "script", "page",
                      "confidence", "latin_readback", "notes"]


def _results_from_by_deed(by_deed) -> list:
    """Collapse chunk-level fields per deed: keep the best occurrence of each
    (id, item_index, attr) - prefer found=true, then highest confidence."""
    results = []
    for reg, d in by_deed.items():
        best = {}
        for fld in d["fields"]:
            k = (fld.get("id", ""), fld.get("item_index", 0), fld.get("attr", ""))
            cand = (bool(fld.get("found")), float(fld.get("confidence") or 0.0))
            if k not in best or cand > best[k][0]:
                best[k] = (cand, fld)
        merged = [v[1] for v in best.values()]
        results.append({
            "key": reg, "reg_no": reg,
            "book_label": d.get("book_label", ""), "deed_type": d.get("deed_type", ""),
            "n_chunks": d.get("n_chunks", d["chunks"]), "chunks_returned": d["chunks"],
            "chunk_errors": d["errors"],
            "status": "ok" if merged else "error",
            "fields": merged,
        })
    results.sort(key=lambda r: r["reg_no"])
    return results


def _write_fields_csv(results, out_csv):
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(_FIELDS_CSV_HEADER)
        for r in results:
            fields = r.get("fields")
            if not isinstance(fields, list):
                continue
            for fld in fields:
                if not isinstance(fld, dict):
                    continue
                w.writerow([
                    r.get("reg_no", r.get("key", "")), r.get("book_label", ""), fld.get("id", ""),
                    fld.get("item_index", ""), fld.get("attr", ""),
                    fld.get("english_value", ""),
                    fld.get("found", ""), fld.get("odia_text", ""), fld.get("script", ""),
                    fld.get("page", ""), fld.get("confidence", ""),
                    fld.get("latin_readback", ""), fld.get("notes", ""),
                ])


def write_grounding_outputs(results, tag="", additive=True) -> list:
    """Write a per-tag snapshot AND merge (keyed by reg_no) into the canonical
    grounding_results.jsonl / grounding_fields.csv. Additive merge means a later
    run adds/updates its own deeds without wiping deeds from an earlier run -- this
    is what prevents the "second run clobbered the first" problem. Returns the full
    merged result list."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if tag:
        snap = OUTPUT_DIR / f"grounding_results_{tag}.jsonl"
        with open(snap, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    canonical = OUTPUT_DIR / "grounding_results.jsonl"
    by_reg = {}
    if additive and canonical.exists():
        with open(canonical, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    by_reg[r.get("reg_no") or r.get("key")] = r
    for r in results:
        by_reg[r.get("reg_no") or r.get("key")] = r
    merged_all = [by_reg[k] for k in sorted(by_reg)]

    with open(canonical, "w", encoding="utf-8") as f:
        for r in merged_all:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    _write_fields_csv(merged_all, OUTPUT_DIR / "grounding_fields.csv")
    return merged_all


def retrieve_results(client, jobs, meta_path, tag=""):
    if not isinstance(jobs, list):
        jobs = [jobs]
    meta = {}
    if meta_path and Path(meta_path).exists():
        with open(meta_path, encoding="utf-8") as f:
            for line in f:
                m = json.loads(line)
                meta[m["key"]] = m

    # Collect chunk-level responses across ALL shard jobs, grouped by deed (reg_no).
    from collections import defaultdict
    by_deed = defaultdict(lambda: {"fields": [], "errors": 0, "chunks": 0})
    n_ok_jobs = 0
    for job in jobs:
        state = getattr(job, "state", None)
        sname = state.name if state else "?"
        if sname != "JOB_STATE_SUCCEEDED":
            err = getattr(job, "error", None)
            print(f"  job {job.name.split('/')[-1]} state={sname} error={err} -- skipped")
            continue
        n_ok_jobs += 1
        _accumulate(_download_job_text(client, job), meta, by_deed)
    print(f"Merged {n_ok_jobs}/{len(jobs)} succeeded job(s).")

    results = _results_from_by_deed(by_deed)
    merged_all = write_grounding_outputs(results, tag=tag, additive=True)

    ok = [r for r in results if r.get("status") == "ok"]
    errs = [r for r in results if r.get("status") == "error"]
    print(f"\n{'='*60}\nGROUNDING BATCH COMPLETE\n{'='*60}")
    print(f"This run (tag={tag or '?'}): OK {len(ok)}, Errors {len(errs)} "
          f"({len(results)} deeds)")
    print(f"Canonical total: {len(merged_all):,} deeds "
          f"-> {OUTPUT_DIR/'grounding_fields.csv'}")
    if ok:
        found = Counter()
        total = Counter()
        for r in ok:
            for fld in (r.get("fields") or []):
                if isinstance(fld, dict):
                    fid = fld.get("id", "?")
                    if fld.get("attr"):
                        fid = f"{fid}.{fld.get('attr')}"
                    total[fid] += 1
                    if fld.get("found"):
                        found[fid] += 1
        print("\nPer-field found-rate (this run):")
        for fid, tot in total.most_common():
            print(f"  {fid:22s} {found[fid]:>5}/{tot:<5} ({100*found[fid]/max(tot,1):.0f}%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="number of deeds (ignored with --all)")
    ap.add_argument("--all", action="store_true", help="process all deeds")
    ap.add_argument("--tag", default="", help="label for output/meta/job files "
                    "(default: 'all' with --all, else the deed count)")
    ap.add_argument("--dry-run", action="store_true", help="build JSONL only, do not submit")
    ap.add_argument("--poll", type=str, help="poll an existing batch job by name")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--thinking-level", dest="thinking_level", default=DEFAULT_THINKING,
                    help="Gemini 3.x thinking level: low | medium | high ('' to disable)")
    ap.add_argument("--max-pages", type=int, default=0, help="cap pages per deed (0 = all)")
    ap.add_argument("--pages-per-call", type=int, default=PAGES_PER_CALL,
                    help="max page images per API request (deed splits into chunks)")
    ap.add_argument("--max-deed-pages", type=int, default=MAX_DEED_PAGES,
                    help="skip deeds with more than this many pages")
    ap.add_argument("--build-workers", type=int, default=0,
                    help="processes for parallel image resize/build (0 = auto)")
    ap.add_argument("--shard-size", type=int, default=800,
                    help="max requests per batch job/shard (Vertex input file limit is 1 GB)")
    ap.add_argument("--strict-schema", action="store_true",
                    help="attach a responseSchema (structured output) to each request")
    ap.add_argument("--pages-manifest", type=Path, default=PAGES_MANIFEST)
    ap.add_argument("--metadata", type=Path, default=DEED_METADATA)
    ap.add_argument("--selection", type=Path, default=SELECTION_CSV)
    ap.add_argument("--deeds", type=Path, default=None,
                    help="restrict to reg_nos in this file (one per line)")
    ap.add_argument("--exclude-deeds", dest="exclude_deeds", type=Path, default=None,
                    help="skip reg_nos in this file (one per line), e.g. already-grounded deeds")
    args = ap.parse_args()

    if args.poll:
        cmd_poll(args)
    else:
        cmd_submit(args)


if __name__ == "__main__":
    main()
