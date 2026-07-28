#!/usr/bin/env python3
"""Realtime (non-batch) Gemini grounding for prompt refinement.

Intended workflow
-----------------
1. We already have a diverse 2500-deed GCS sample
   (``data/mismatches/gcs_diverse_sample.json``).
2. Pick ~10 of those for realtime Gemini + prompt iteration.
3. Once the prompt looks solid, run ``grounding_batch_gemini.py`` on the
   full 2500 (``--deeds data/mismatches/gcs_diverse_sample_reg_nos.txt``).

Pulls page images + metadata directly from GCS (no Render).

Required env
------------
  GCS_BUCKET                 e.g. classification-vision
  GCS_CREDENTIALS_JSON       service-account JSON (full contents)
  GCS_RAW_PREFIX             default: ocr_outputs/orissa_deeds
  GCP_PROJECT                default: vision-projects-463307
  (Vertex ADC / same SA must be able to call Gemini on the project)

Examples
--------
  # Default: 10 diverse deeds FROM the existing 2500 GCS sample:
  python3 grounding_realtime_gemini.py --n 10 --out-dir data/prompt_refine

  # Re-run the same 10 after editing prompt.py:
  python3 grounding_realtime_gemini.py \\
      --deeds data/prompt_refine/sample_reg_nos.txt

  # Dry-run (pages + prompts only, no Gemini):
  python3 grounding_realtime_gemini.py --n 10 --dry-run

  # Later — batch the full 2500 with the refined prompt:
  python3 grounding_batch_gemini.py --deeds data/mismatches/gcs_diverse_sample_reg_nos.txt
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

Image.MAX_IMAGE_PIXELS = None

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gcs_store  # noqa: E402
import grounding_batch_gemini as batch  # noqa: E402
import prompt as grounding_prompt  # noqa: E402
from compare_metadata import grounding_book_no, grounding_scalars  # noqa: E402
from sarvam_client import normalize_reg_no  # noqa: E402

GCP_PROJECT = os.environ.get("GCP_PROJECT", batch.GCP_PROJECT)
GCP_LOCATION = os.environ.get("GCP_LOCATION", batch.GCP_LOCATION)
DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", batch.DEFAULT_MODEL)
DEFAULT_THINKING = os.environ.get("GEMINI_THINKING", batch.DEFAULT_THINKING)

MAX_DIMENSION = batch.MAX_DIMENSION
JPEG_QUALITY = batch.JPEG_QUALITY
PAGES_PER_CALL = batch.PAGES_PER_CALL
MAX_DEED_PAGES = batch.MAX_DEED_PAGES


# ----------------------------------------------------------------------------
# GCS helpers
# ----------------------------------------------------------------------------
def _require_gcs():
    if not os.environ.get("GCS_BUCKET"):
        sys.exit(
            "ERROR: set GCS_BUCKET (and usually GCS_CREDENTIALS_JSON). "
            "This tool talks to GCS directly — Render is not used."
        )
    if not gcs_store.enabled():
        sys.exit("ERROR: gcs_store.enabled() is false — check GCS_BUCKET.")


def iter_raw_groundings(limit: int | None = None):
    """Yield grounding dicts from the raw orissa_deeds JSONL on GCS (streamed)."""
    n = 0
    bucket = gcs_store._get_bucket()
    for prefix in gcs_store.raw_prefixes():
        path = f"{prefix}/grounding/grounding_good_partial.jsonl"
        print(f"[gcs] streaming {path} …", flush=True)
        blob = bucket.blob(path)
        if not blob.exists():
            print(f"[gcs] missing {path}", flush=True)
            continue
        with blob.open("rt", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    g = json.loads(line)
                except json.JSONDecodeError:
                    continue
                reg = normalize_reg_no(g.get("reg_no") or g.get("key") or "")
                if not reg:
                    continue
                g.setdefault("reg_no", reg)
                yield g
                n += 1
                if limit is not None and n >= limit:
                    return


def page_blobs_for(reg_no: str, max_pages: int = 0) -> list[tuple[int, bytes]]:
    """Download page images for a deed from GCS. Returns [(page_num, jpeg_bytes)]."""
    entry = gcs_store.pages_entry(reg_no)  # {"pages": [[pg, prefix, rel], ...]}
    pages = (entry or {}).get("pages") or []
    if not pages:
        return []
    if max_pages and len(pages) > max_pages:
        pages = pages[:max_pages]
    out = []
    for pg, _prefix, _rel in pages:
        raw, _ctype = gcs_store.fetch_page_image(reg_no, int(pg))
        if not raw:
            continue
        out.append((int(pg), _to_jpeg(raw)))
    return out


def _to_jpeg(raw: bytes) -> bytes:
    img = Image.open(io.BytesIO(raw))
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


def meta_from_grounding(g: dict) -> dict:
    """Tabular meta + party/property blobs derived from a grounding record."""
    scalars = grounding_scalars(g)
    meta = {
        "reg_no": g.get("reg_no"),
        "book_label": g.get("book_label", ""),
        "deed_type": scalars.get("deed_type") or g.get("deed_type") or "",
        "district": scalars.get("district") or "",
        "office": scalars.get("office") or "",
        "registration_date": scalars.get("registration_date") or "",
        "presentation_date": scalars.get("presentation_date") or "",
        "execution_date": scalars.get("execution_date") or "",
        "consideration_amount": scalars.get("consideration_amount") or "",
        "old_reg_no": scalars.get("old_reg_no") or "",
        # listed_on may appear in some exports; targets_from_tabular maps it
        # to execution_date and never sends id=listed_on.
        "listed_on": scalars.get("listed_on") or g.get("listed_on") or "",
    }
    return meta


def raw_party_blobs_from_grounding(g: dict) -> dict:
    """Rebuild seller/buyer/property English blobs from grounding fields when
    the raw IGR-style strings are not stored on the grounding record."""
    # Prefer any already-attached API-style blobs
    for key in ("sellerDetails", "buyerDetails", "propertyDetails"):
        if g.get(key):
            return {
                "sellerDetails": g.get("sellerDetails") or "",
                "buyerDetails": g.get("buyerDetails") or "",
                "propertyDetails": g.get("propertyDetails") or "",
            }

    by_id: dict[str, list] = defaultdict(list)
    for f in g.get("fields") or []:
        if not isinstance(f, dict):
            continue
        fid = f.get("id") or ""
        if fid in ("seller_details", "buyer_details", "property_details"):
            by_id[fid].append(f)

    def _party_blob(rows: list) -> str:
        by_item: dict[int, dict] = defaultdict(dict)
        for r in rows:
            try:
                ii = int(r.get("item_index") or 0)
            except (TypeError, ValueError):
                continue
            if ii <= 0:
                continue
            attr = (r.get("attr") or "").strip()
            eng = (r.get("english_value") or "").strip()
            if attr and eng:
                by_item[ii][attr] = eng
        parts = []
        for ii in sorted(by_item):
            d = by_item[ii]
            name = d.get("name", "")
            rel = d.get("relation", "Others")
            rname = d.get("relation_name", "")
            addr = d.get("address", "")
            parts.append(
                f"{ii}-{name}   ( RELATION : )  {rel}  (  RELATION NAME : )  "
                f"{rname}  (  ADDRESS : )  {addr} "
            )
        return ",".join(parts)

    def _prop_blob(rows: list) -> str:
        by_item: dict[int, dict] = defaultdict(dict)
        for r in rows:
            try:
                ii = int(r.get("item_index") or 0)
            except (TypeError, ValueError):
                continue
            if ii <= 0:
                continue
            attr = (r.get("attr") or "").strip()
            eng = (r.get("english_value") or "").strip()
            if attr and eng:
                by_item[ii][attr] = eng
        parts = []
        for ii in sorted(by_item):
            d = by_item[ii]
            parts.append(
                f"{ii}- Village : {d.get('village', '')}  "
                f"Khata : {d.get('khata', '')} Plot : {d.get('plot', '')} "
                f"Area: {d.get('area', '')} Total Area: {d.get('total_area', d.get('area', ''))} "
                f"Boundary : {d.get('boundary', '')} "
            )
        return ",".join(parts)

    return {
        "sellerDetails": _party_blob(by_id.get("seller_details", [])),
        "buyerDetails": _party_blob(by_id.get("buyer_details", [])),
        "propertyDetails": _prop_blob(by_id.get("property_details", [])),
    }


# ----------------------------------------------------------------------------
# Sample N deeds from the existing 2500 GCS diverse sample (or whole corpus)
# ----------------------------------------------------------------------------
DEFAULT_GCS_SAMPLE_JSON = Path("data/mismatches/gcs_diverse_sample.json")
DEFAULT_GCS_SAMPLE_REGS = Path("data/mismatches/gcs_diverse_sample_reg_nos.txt")


def _load_sample_pool(sample_json: Path) -> list[dict]:
    """Load the 2500-deed GCS sample index (reg_no + book/district metadata)."""
    if not sample_json.exists():
        sys.exit(f"ERROR: sample pool not found: {sample_json}")
    data = json.loads(sample_json.read_text(encoding="utf-8"))
    deeds = data.get("deeds") if isinstance(data, dict) else data
    if not isinstance(deeds, list) or not deeds:
        sys.exit(f"ERROR: no deeds in {sample_json}")
    return deeds


def pick_regs_from_pool(pool: list[dict], n: int, seed: int = 42) -> list[str]:
    """Diverse pick of N reg_nos from the 2500 sample (book × district)."""
    random.seed(seed)
    buckets: dict[tuple, list] = defaultdict(list)
    for row in pool:
        reg = normalize_reg_no(row.get("registration_no") or row.get("reg_no"))
        if not reg:
            continue
        book = row.get("book_no") or 0
        try:
            book = int(book)
        except (TypeError, ValueError):
            book = 0
        label = (row.get("book_label") or "").upper() or "?"
        dist = (row.get("district") or "UNKNOWN").upper()
        buckets[(book, label, dist)].append(reg)

    keys = sorted(buckets.keys(), key=lambda k: (-len(buckets[k]), k))
    for k in keys:
        random.shuffle(buckets[k])

    picked: list[str] = []
    seen: set[str] = set()
    while len(picked) < n and any(buckets[k] for k in keys):
        for k in keys:
            if len(picked) >= n:
                break
            while buckets[k]:
                reg = buckets[k].pop()
                if reg in seen:
                    continue
                seen.add(reg)
                picked.append(reg)
                break
    print(f"[sample] pool={len(pool)} buckets={len(buckets)} picked={len(picked)}",
          flush=True)
    return picked


def sample_diverse(n: int, seed: int = 42, pool_scan: int = 5000) -> list[dict]:
    """Fallback: diverse pick by scanning the raw GCS grounding JSONL."""
    random.seed(seed)
    buckets: dict[tuple, list] = defaultdict(list)
    scanned = 0
    for g in iter_raw_groundings(limit=pool_scan):
        scanned += 1
        book = grounding_book_no(g) or 0
        label = (g.get("book_label") or "").upper() or "?"
        scalars = grounding_scalars(g)
        dist = (scalars.get("district") or "UNKNOWN").upper()
        buckets[(book, label, dist)].append(g)
    print(f"[sample] scanned={scanned} buckets={len(buckets)}", flush=True)

    keys = sorted(buckets.keys(), key=lambda k: (-len(buckets[k]), k))
    for k in keys:
        random.shuffle(buckets[k])
    picked = []
    seen_reg = set()
    while len(picked) < n and any(buckets[k] for k in keys):
        for k in keys:
            if len(picked) >= n:
                break
            while buckets[k]:
                g = buckets[k].pop()
                reg = g["reg_no"]
                if reg in seen_reg:
                    continue
                entry = gcs_store.pages_entry(reg)
                pages = (entry or {}).get("pages") or []
                if not pages:
                    continue
                if len(pages) > MAX_DEED_PAGES:
                    continue
                g["_n_pages"] = len(pages)
                seen_reg.add(reg)
                picked.append(g)
                break
    return picked


def load_groundings_for_regs(regs: list[str]) -> list[dict]:
    want = set(normalize_reg_no(r) for r in regs if r)
    found = {}
    for g in iter_raw_groundings():
        reg = g["reg_no"]
        if reg in want and reg not in found:
            entry = gcs_store.pages_entry(reg)
            pages = (entry or {}).get("pages") or []
            g["_n_pages"] = len(pages)
            found[reg] = g
            if len(found) >= len(want):
                break
    out = []
    for r in regs:
        reg = normalize_reg_no(r)
        if reg in found:
            out.append(found[reg])
        else:
            print(f"[warn] reg_no not in GCS grounding jsonl: {reg}", flush=True)
    return out


def sample_from_gcs_pool(sample_json: Path, n: int, seed: int = 42) -> list[dict]:
    """Pick N diverse reg_nos from the 2500 sample, load grounding from GCS."""
    pool = _load_sample_pool(sample_json)
    regs = pick_regs_from_pool(pool, n=n, seed=seed)
    print(f"[sample] loading grounding for {len(regs)} regs from GCS jsonl…",
          flush=True)
    return load_groundings_for_regs(regs)


# ----------------------------------------------------------------------------
# Realtime Gemini call
# ----------------------------------------------------------------------------
def make_client():
    from google import genai
    print(f"Using Vertex AI realtime (project={GCP_PROJECT}, location={GCP_LOCATION})")
    return genai.Client(vertexai=True, project=GCP_PROJECT, location=GCP_LOCATION)


def call_gemini(client, model: str, image_bytes_list: list[bytes], prompt_text: str,
                thinking_level: str, strict_schema: bool) -> list | dict:
    from google.genai import types

    parts = []
    for b in image_bytes_list:
        parts.append(types.Part.from_bytes(data=b, mime_type="image/jpeg"))
    parts.append(types.Part.from_text(text=prompt_text))

    cfg_kwargs = {
        "response_mime_type": "application/json",
    }
    if strict_schema:
        cfg_kwargs["response_schema"] = batch.response_schema()
    if thinking_level:
        cfg_kwargs["thinking_config"] = types.ThinkingConfig(
            thinking_level=thinking_level.upper())

    resp = client.models.generate_content(
        model=model,
        contents=[types.Content(role="user", parts=parts)],
        config=types.GenerateContentConfig(
            system_instruction=grounding_prompt.SYSTEM_INSTRUCTION,
            **cfg_kwargs,
        ),
    )
    text = getattr(resp, "text", None) or ""
    if not text and getattr(resp, "candidates", None):
        for cand in resp.candidates:
            content = getattr(cand, "content", None)
            if not content:
                continue
            for part in getattr(content, "parts", None) or []:
                text += getattr(part, "text", None) or ""
    return batch.parse_json_array(text)


def run_one_deed(client, g: dict, opts: dict) -> dict:
    reg = g["reg_no"]
    meta = meta_from_grounding(g)
    raw = raw_party_blobs_from_grounding(g)
    targets = batch.build_targets(meta, raw)
    if not targets:
        return {"reg_no": reg, "status": "skip_no_targets", "fields": []}

    pages = page_blobs_for(reg, max_pages=opts["max_pages"] or 0)
    if not pages:
        return {"reg_no": reg, "status": "skip_no_pages", "fields": []}
    if len(pages) > opts["max_deed_pages"]:
        return {"reg_no": reg, "status": "skip_too_long",
                "n_pages": len(pages), "fields": []}

    deed_type = meta.get("deed_type") or g.get("book_label") or ""
    all_fields = []
    errors = 0
    n_chunks = (len(pages) + opts["pages_per_call"] - 1) // opts["pages_per_call"]
    t0 = time.time()

    for ci in range(n_chunks):
        chunk = pages[ci * opts["pages_per_call"]:(ci + 1) * opts["pages_per_call"]]
        page_numbers = [p for p, _ in chunk]
        imgs = [b for _, b in chunk]
        prompt_text = grounding_prompt.build_user_prompt(
            targets, n_pages=len(imgs), deed_type=deed_type,
            page_offset=ci * opts["pages_per_call"])

        # Persist prompt for inspection / prompt-iteration diffs
        prompt_dir = Path(opts["out_dir"]) / "prompts"
        prompt_dir.mkdir(parents=True, exist_ok=True)
        (prompt_dir / f"{reg}__c{ci}.txt").write_text(
            grounding_prompt.SYSTEM_INSTRUCTION + "\n\n" + prompt_text,
            encoding="utf-8")

        if opts["dry_run"]:
            continue

        try:
            fields = call_gemini(
                client, opts["model"], imgs, prompt_text,
                opts["thinking_level"], opts["strict_schema"])
        except Exception as e:  # noqa: BLE001
            print(f"  [{reg} c{ci}] ERROR {type(e).__name__}: {e}", flush=True)
            errors += 1
            continue

        if not isinstance(fields, list):
            errors += 1
            all_fields.append({"parse_error": True, "raw": fields})
            continue

        for fld in fields:
            if not isinstance(fld, dict):
                continue
            local = fld.get("page") or 0
            fld["page_local"] = local
            if fld.get("found") and 1 <= local <= len(page_numbers):
                fld["page"] = page_numbers[local - 1]
            elif not fld.get("found"):
                fld["page"] = 0
            all_fields.append(fld)

    # Collapse duplicates across chunks (same as batch)
    best = {}
    for fld in all_fields:
        if not isinstance(fld, dict) or "id" not in fld:
            continue
        k = (fld.get("id", ""), fld.get("item_index", 0), fld.get("attr", ""))
        cand = (bool(fld.get("found")), float(fld.get("confidence") or 0.0))
        if k not in best or cand > best[k][0]:
            best[k] = (cand, fld)
    merged = [v[1] for v in best.values()]

    return {
        "reg_no": reg,
        "book_label": g.get("book_label", ""),
        "deed_type": deed_type,
        "n_pages": len(pages),
        "n_chunks": n_chunks,
        "chunk_errors": errors,
        "status": "ok" if merged and not opts["dry_run"] else (
            "dry_run" if opts["dry_run"] else "error"),
        "targets_sent": [
            {"id": t["id"], "type": t.get("type"), "value": t.get("value")}
            for t in targets
        ],
        "omitted_check": sorted(grounding_prompt.OMITTED_TARGET_IDS),
        "elapsed_s": round(time.time() - t0, 1),
        "fields": merged,
    }


def summarize(results: list[dict]) -> dict:
    found = Counter()
    total = Counter()
    for r in results:
        for fld in r.get("fields") or []:
            if not isinstance(fld, dict) or "id" not in fld:
                continue
            fid = fld.get("id", "?")
            if fld.get("attr"):
                fid = f"{fid}.{fld.get('attr')}"
            total[fid] += 1
            if fld.get("found"):
                found[fid] += 1
    rates = {
        fid: {
            "found": found[fid],
            "total": tot,
            "pct": round(100 * found[fid] / max(tot, 1), 1),
        }
        for fid, tot in total.most_common()
    }
    return {
        "n_results": len(results),
        "ok": sum(1 for r in results if r.get("status") == "ok"),
        "per_field": rates,
    }


def main():
    ap = argparse.ArgumentParser(
        description="Realtime Gemini grounding: refine prompt on ~10 of the "
                    "2500 GCS-sampled deeds (no Render)")
    ap.add_argument("--n", type=int, default=10,
                    help="how many deeds to take from the 2500 GCS sample")
    ap.add_argument("--from-sample", type=Path, default=DEFAULT_GCS_SAMPLE_JSON,
                    help="2500-deed GCS sample JSON to pick --n from "
                         f"(default: {DEFAULT_GCS_SAMPLE_JSON})")
    ap.add_argument("--deeds", type=Path, default=None,
                    help="file of reg_nos (one per line); skips sampling "
                         "(use after the first run to re-test the same 10)")
    ap.add_argument("--from-corpus", action="store_true",
                    help="ignore --from-sample; pick --n from the full GCS "
                         "grounding jsonl instead of the 2500 sample")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pool-scan", type=int, default=5000,
                    help="with --from-corpus: how many grounding lines to scan")
    ap.add_argument("--out-dir", type=Path, default=Path("data/prompt_refine"))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--thinking-level", default=DEFAULT_THINKING)
    ap.add_argument("--pages-per-call", type=int, default=PAGES_PER_CALL)
    ap.add_argument("--max-pages", type=int, default=0)
    ap.add_argument("--max-deed-pages", type=int, default=MAX_DEED_PAGES)
    ap.add_argument("--strict-schema", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="download pages + write prompts only, no Gemini calls")
    args = ap.parse_args()

    _require_gcs()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.deeds:
        regs = [
            ln.strip() for ln in args.deeds.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.startswith("#")
        ]
        print(f"[deeds] {len(regs)} reg_nos from {args.deeds}", flush=True)
        groundings = load_groundings_for_regs(regs)
    elif args.from_corpus:
        print(f"[sample] selecting {args.n} from full GCS corpus…", flush=True)
        groundings = sample_diverse(args.n, seed=args.seed, pool_scan=args.pool_scan)
    else:
        print(f"[sample] selecting {args.n} from 2500-deed GCS sample "
              f"({args.from_sample})…", flush=True)
        groundings = sample_from_gcs_pool(args.from_sample, n=args.n, seed=args.seed)

    reg_list = [g["reg_no"] for g in groundings]
    (args.out_dir / "sample_reg_nos.txt").write_text(
        "\n".join(reg_list) + ("\n" if reg_list else ""), encoding="utf-8")
    with open(args.out_dir / "sample_groundings.json", "w", encoding="utf-8") as f:
        # Strip bulky fields list for the index; full grounding kept per-deed below
        slim = []
        for g in groundings:
            slim.append({
                "reg_no": g.get("reg_no"),
                "book_label": g.get("book_label"),
                "deed_type": g.get("deed_type"),
                "n_fields": len(g.get("fields") or []),
                "n_pages": g.get("_n_pages"),
            })
        json.dump(slim, f, ensure_ascii=False, indent=2)
    print(f"[sample] {len(groundings)} deeds -> {args.out_dir / 'sample_reg_nos.txt'}",
          flush=True)

    opts = {
        "model": args.model,
        "thinking_level": args.thinking_level,
        "pages_per_call": args.pages_per_call,
        "max_pages": args.max_pages,
        "max_deed_pages": args.max_deed_pages,
        "strict_schema": args.strict_schema,
        "dry_run": args.dry_run,
        "out_dir": str(args.out_dir),
    }

    client = None if args.dry_run else make_client()
    results = []
    for i, g in enumerate(groundings, 1):
        print(f"\n[{i}/{len(groundings)}] {g['reg_no']} "
              f"book={g.get('book_label')} pages~{g.get('_n_pages', '?')}",
              flush=True)
        # Keep a copy of grounding for offline review
        (args.out_dir / f"{g['reg_no']}_grounding.json").write_text(
            json.dumps(g, ensure_ascii=False, indent=2), encoding="utf-8")
        res = run_one_deed(client, g, opts)
        results.append(res)
        print(f"  status={res['status']} fields={len(res.get('fields') or [])} "
              f"targets={len(res.get('targets_sent') or [])} "
              f"err={res.get('chunk_errors', 0)} {res.get('elapsed_s', 0)}s",
              flush=True)
        # Assert omitted ids never appear in what we sent
        sent_ids = {t["id"] for t in res.get("targets_sent") or []}
        bad = sent_ids & grounding_prompt.OMITTED_TARGET_IDS
        if bad:
            print(f"  WARNING: omitted ids leaked into targets: {bad}", flush=True)

    out_jsonl = args.out_dir / "realtime_results.jsonl"
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Flatten CSV for quick scanning
    csv_path = args.out_dir / "realtime_fields.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(batch._FIELDS_CSV_HEADER)
        for r in results:
            for fld in r.get("fields") or []:
                if not isinstance(fld, dict) or "id" not in fld:
                    continue
                w.writerow([
                    r.get("reg_no", ""), r.get("book_label", ""), fld.get("id", ""),
                    fld.get("item_index", ""), fld.get("attr", ""),
                    fld.get("english_value", ""),
                    fld.get("found", ""), fld.get("odia_text", ""), fld.get("script", ""),
                    fld.get("page", ""), fld.get("confidence", ""),
                    fld.get("latin_readback", ""), fld.get("notes", ""),
                ])

    summary = summarize(results)
    (args.out_dir / "realtime_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{'=' * 60}\nREALTIME GROUNDING DONE\n{'=' * 60}")
    print(f"results: {out_jsonl}")
    print(f"fields:  {csv_path}")
    print(f"summary: {args.out_dir / 'realtime_summary.json'}")
    print(f"ok={summary['ok']}/{summary['n_results']}")
    if summary["per_field"]:
        print("\nPer-field found-rate:")
        for fid, info in list(summary["per_field"].items())[:40]:
            print(f"  {fid:28s} {info['found']:>4}/{info['total']:<4} ({info['pct']:.0f}%)")


if __name__ == "__main__":
    main()
