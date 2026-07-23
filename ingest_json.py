"""
ingest_json.py — load deeds from the grounding.json + ocr.jsonl format.

INPUT FORMAT (one folder per deed, named by reg_no)
---------------------------------------------------
    <reg_no>/
        <reg_no>.pdf       the scanned deed
        grounding.json     structured metadata fields extracted by the model
        ocr.jsonl          full-page OCR text, one JSON line per page

grounding.json:
    { "reg_no": "...", "book_label": "...", "deed_type": "...", ...,
      "fields": [ { "id": "seller_details", "attr": "name", "item_index": 1,
                    "field": "display name", "english_value": "...",
                    "odia_text": "...", "latin_readback": "...",
                    "found": true, "confidence": 0.9, "page": 1,
                    "notes": "..." }, ... ] }

ocr.jsonl (per line):
    { "page": 3, "char_len": 1407, "audit": ..., "text": "full page text" }
    (OCR may cover only some pages of a deed.)

MAPPING INTO THE PORTAL
-----------------------
- One documents row per deed (deed_number = reg_no).
- Each grounding field becomes an editable fields row:
    section  : "Deed details" for scalars, else "Seller 1", "Buyer 2",
               "Property 1", ... (from id + item_index)
    label    : the field's display name ("Name", "Address", "Deed type", ...)
    english  : english_value  -> current_value (editable, ocr_value immutable)
    odia     : odia_text      -> odia_value (editable)
    The full original field object is preserved in src_block so corrected
    output can be exported in exactly the input shape.
- ocr.jsonl pages, joined in page order, populate the Full text tab.

USAGE
    python ingest_json.py <data_dir>        # scan for all deed folders
    python ingest_json.py <deed_folder>     # load a single deed folder
"""

import json
import re
import shutil
import sys
from pathlib import Path

from db import init_db, connect

_YEAR_RE = re.compile(r"(19|20)\d{2}")
_SHORT_DATE_RE = re.compile(r"\b\d{1,2}[./\-]\d{1,2}[./\-](\d{2})\b")


def _year_from_text(v):
    """Pull a year out of a date string. Handles 4-digit (22-May-2000,
    26/11/2013) and 2-digit (21/8/98, 3.8.98) years."""
    s = str(v or "")
    m = _YEAR_RE.search(s)
    if m:
        return int(m.group(0))
    m = _SHORT_DATE_RE.search(s)
    if m:
        yy = int(m.group(1))
        # registry deeds: 00–30 -> 2000s, else 1900s
        return 2000 + yy if yy <= 30 else 1900 + yy
    return None


def _year_from_fields(fields):
    """Pull the year out of the registration_date (fallback presentation_date),
    checking both the English and Odia values."""
    for fid in ("registration_date", "presentation_date"):
        for f in fields:
            if f.get("id") == fid:
                for v in (f.get("english_value"), f.get("odia_text")):
                    y = _year_from_text(v)
                    if y:
                        return y
    return None

SECTION_NAMES = {
    "seller_details": "Seller",
    "buyer_details": "Buyer",
    "property_details": "Property",
}
SECTION_PLURALS = {"Seller": "Sellers", "Buyer": "Buyers", "Property": "Properties"}

ATTR_LABELS = {
    "name": "Name", "relation_name": "Relation name", "address": "Address",
    "village": "Village", "khata": "Khata", "plot": "Plot", "area": "Area",
}


def _pretty_attr(attr):
    return ATTR_LABELS.get(attr, (attr or "value").replace("_", " ").title())


def _merge_enabled():
    import os
    # ON by default; set MERGE_PARTY_FIELDS=0 to keep per-item fields
    return os.environ.get("MERGE_PARTY_FIELDS", "1").lower() not in ("0", "false", "no")


def _build_field_rows(fields):
    """Turn grounding fields into portal field rows.

    Default: one row per grounding field (Seller 1 / Seller 2 ... sections).
    With MERGE_PARTY_FIELDS=1: list fields (seller/buyer/property) are MERGED —
    one row per attribute with the items' values comma-separated, under a
    single section like "Buyers (5)". Original per-item blocks kept in
    src_block so export can split corrections back into per-item fields.
    Returns list of dicts: section, label, english, odia, src_block, page.
    """
    if not _merge_enabled():
        rows = []
        for f in fields:
            section, label = _section_and_label(f)
            rows.append({
                "section": section, "label": label,
                "english": f.get("english_value") or "",
                "odia": f.get("odia_text") or "",
                "src_block": f, "page": f.get("page"),
            })
        return rows

    rows = []
    groups = {}          # (id, attr) -> list of item blocks
    group_order = []     # first-appearance order of (id, attr)
    counts = {}          # id -> max item_index seen

    for f in fields:
        fid = f.get("id", "field")
        attr = (f.get("attr") or "").strip()
        idx = f.get("item_index") or 0
        if fid in SECTION_NAMES and idx:
            key = (fid, attr)
            if key not in groups:
                groups[key] = []
                group_order.append(key)
            groups[key].append(f)
            counts[fid] = max(counts.get(fid, 0), idx)
        else:
            rows.append({
                "section": "Deed details",
                "label": f.get("field") or fid.replace("_", " ").title(),
                "english": f.get("english_value") or "",
                "odia": f.get("odia_text") or "",
                "src_block": f,
                "page": f.get("page"),
            })

    for (fid, attr) in group_order:
        items = sorted(groups[(fid, attr)], key=lambda x: x.get("item_index") or 0)
        n = counts.get(fid, len(items))
        english = ", ".join((i.get("english_value") or "").strip() for i in items)
        odia = ", ".join((i.get("odia_text") or "").strip() for i in items)
        base = SECTION_NAMES[fid]
        rows.append({
            "section": f"{SECTION_PLURALS[base]} ({n})" if n != 1 else base,
            "label": _pretty_attr(attr),
            "english": english,
            "odia": odia,
            "src_block": {"group": True, "id": fid, "attr": attr,
                          "items": items},
            "page": items[0].get("page") if items else None,
        })
    return rows


def _section_and_label(f):
    fid = f.get("id", "field")
    attr = (f.get("attr") or "").strip()
    idx = f.get("item_index") or 0
    if fid in SECTION_NAMES and idx:
        section = f"{SECTION_NAMES[fid]} {idx}"
        label = ATTR_LABELS.get(attr, attr.replace("_", " ").title() or "Value")
    else:
        section = "Deed details"
        label = f.get("field") or fid.replace("_", " ").title()
    return section, label


def _load_ocr_text(deed_dir):
    """Join ocr.jsonl pages (page order) into the Full text tab content."""
    p = Path(deed_dir) / "ocr.jsonl"
    if not p.exists():
        return None
    pages = []
    for line in open(p, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            o = json.loads(line)
        except json.JSONDecodeError:
            continue
        pages.append((o.get("page", 0), o.get("text", "")))
    pages.sort()
    parts = [f"— Page {pg} —\n{txt}".strip() for pg, txt in pages if txt]
    return "\n\n".join(parts) or None


def find_pdf(data_dir, reg_no):
    """Locate <reg_no>.pdf under data_dir (used for scan repair too)."""
    data = Path(data_dir)
    direct = data / str(reg_no) / f"{reg_no}.pdf"
    if direct.exists():
        return direct
    for p in data.rglob(f"{reg_no}.pdf"):
        return p
    return None


def load_deed(deed_dir, scans_dir="static/scans"):
    """Load one deed folder (grounding.json [+ ocr.jsonl] [+ pdf])."""
    deed_dir = Path(deed_dir)
    gpath = deed_dir / "grounding.json"
    if not gpath.exists():
        raise SystemExit(f"No grounding.json in {deed_dir}")
    g = json.load(open(gpath, encoding="utf-8"))
    reg_no = str(g.get("reg_no") or deed_dir.name)
    g.setdefault("reg_no", reg_no)

    pdf_name = None
    pdf_src = deed_dir / f"{reg_no}.pdf"
    if pdf_src.exists():
        Path(scans_dir).mkdir(parents=True, exist_ok=True)
        pdf_name = f"{reg_no}.pdf"
        shutil.copy(pdf_src, Path(scans_dir) / pdf_name)

    full_text = _load_ocr_text(deed_dir)
    con = connect()
    try:
        ok = _insert_from_grounding(con, g, full_text, pdf_name)
        con.commit()
        if ok:
            print(f"loaded {reg_no}: pdf={'yes' if pdf_name else 'no'}, "
                  f"ocr={'yes' if full_text else 'no'}")
        else:
            print(f"{reg_no} already present — skipping.")
        return bool(ok)
    finally:
        con.close()


def ingest_dir(data_dir, scans_dir="static/scans", init=True):
    """Scan data_dir for deed folders (any folder containing grounding.json)."""
    if init:
        init_db()
    data = Path(data_dir)
    gfiles = sorted(data.rglob("grounding.json"))
    loaded = skipped = 0
    for g in gfiles:
        try:
            if load_deed(g.parent, scans_dir):
                loaded += 1
            else:
                skipped += 1
        except Exception as e:
            print(f"failed {g.parent.name}: {e}", flush=True)
    print(f"\ndone: {loaded} loaded, {skipped} already present")


def _insert_from_grounding(con, g, full_text, pdf_name):
    """Shared insert used by both local and GCS paths."""
    reg_no = str(g.get("reg_no") or "").strip()
    if not reg_no:
        return None
    if con.execute("SELECT 1 FROM documents WHERE deed_number=%s",
                   (reg_no,)).fetchone():
        return False
    src_meta = {k: v for k, v in g.items() if k != "fields"}
    doc_id = con.execute(
        "INSERT INTO documents (deed_number, deed_type, year, pdf_file, status, "
        "digitized_text, digitized_status, src_meta) "
        "VALUES (%s,%s,%s,%s,'pending',%s,%s,%s) RETURNING id",
        (reg_no, g.get("deed_type"), _year_from_fields(g.get("fields", [])),
         pdf_name, full_text,
         "ready" if full_text else "not_started",
         json.dumps(src_meta))).fetchone()["id"]
    rows = []
    for i, r in enumerate(_build_field_rows(g.get("fields", []))):
        rows.append((doc_id, r["section"], r["label"], r["english"], r["english"],
                     r["odia"], len(r["english"]) > 60, i, "text",
                     (r["src_block"].get("id") if isinstance(r["src_block"], dict) else None),
                     json.dumps(r["src_block"]), r["page"]))
    if rows:
        with con.cursor() as cur:
            cur.executemany(
                "INSERT INTO fields (document_id, section, label, ocr_value, "
                "current_value, odia_value, multiline, position, field_kind, "
                "layout_tag, src_block, page_num) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)", rows)
    return True


def _ocr_lines_to_text(raw):
    """ocr.jsonl content (string) -> joined Full-text content."""
    if not raw:
        return None
    pages = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            o = json.loads(line)
        except json.JSONDecodeError:
            continue
        pages.append((o.get("page", 0), o.get("text", "")))
    pages.sort()
    parts = [f"— Page {pg} —\n{txt}".strip() for pg, txt in pages if txt]
    return "\n\n".join(parts) or None


def ingest_gcs(init=True, progress=None):
    """Ingest every deed directly from the GCS bucket (see gcs_store.py).
    PDFs are NOT downloaded here — they stream on first view."""
    import gcs_store
    if init:
        init_db()
    ids = gcs_store.list_deed_ids()
    print(f"[gcs] {len(ids)} deed folders in bucket", flush=True)
    con = connect()
    loaded = skipped = failed = 0
    try:
        # Skip anything already in the DB *before* touching GCS — on a
        # reingest, most or all of these ids are already loaded, and
        # fetching grounding.json + ocr.jsonl for each one just to discard
        # it is what was making "check for new deeds" take forever.
        existing = {r["deed_number"] for r in con.execute(
            "SELECT deed_number FROM documents").fetchall()}
        new_ids = [i for i in ids if i not in existing]
        skipped += len(ids) - len(new_ids)
        print(f"[gcs] {len(existing)} already in DB, "
              f"{len(new_ids)} to check", flush=True)
        for n, reg_no in enumerate(new_ids, 1):
            try:
                graw = gcs_store.read_text(f"{reg_no}/grounding.json")
                if not graw:
                    failed += 1
                    continue
                g = json.loads(graw)
                full_text = _ocr_lines_to_text(
                    gcs_store.read_text(f"{reg_no}/ocr.jsonl"))
                ok = _insert_from_grounding(con, g, full_text, f"{reg_no}.pdf")
                if ok:
                    loaded += 1
                elif ok is False:
                    skipped += 1
                else:
                    failed += 1
                if n % 50 == 0:
                    con.commit()
                    print(f"[gcs] {n}/{len(new_ids)} new ids processed "
                          f"({loaded} loaded)", flush=True)
                    if progress:
                        progress(n, len(new_ids), loaded)
            except Exception as e:
                failed += 1
                print(f"[gcs] failed {reg_no}: {e}", flush=True)
        con.commit()
    finally:
        con.close()
    print(f"[gcs] done: {loaded} loaded, {skipped} already present, "
          f"{failed} failed/empty", flush=True)
    return loaded


def ingest_gcs_raw(init=True, progress=None):
    """Ingest the raw orissa_deeds export directly from GCS — reads
    grounding/grounding_good_partial.jsonl and ocr/ocr_dataset.jsonl (under
    gcs_store.GCS_RAW_PREFIX, default 'ocr_outputs/orissa_deeds'), no local
    copies needed. Only ever does object READS on the bucket — never lists
    or writes to it. pdf_file is set to '<reg_no>.pdf' whenever that deed
    has page images, purely as a "this deed has a scan" flag (used for the
    has_pdf column and the viewer's no-scan message) — no PDF is actually
    built for these. The viewer instead serves the raw page images directly,
    one request per page (gcs_store.fetch_page_image via /api/documents/
    {id}/page/{n}), and displays them as a sequence in the browser. Safe to
    re-run: existing deed_numbers are skipped, so this is exactly how you
    add a new batch without resetting anything."""
    import gcs_store
    if init:
        init_db()
    raw_prefix = gcs_store._raw_prefix()
    print(f"[gcs-raw] reading dataset from gs://.../{raw_prefix}", flush=True)
    graw = gcs_store.read_text_abs(f"{raw_prefix}/grounding/grounding_good_partial.jsonl")
    if not graw:
        print(f"[gcs-raw] grounding_good_partial.jsonl not found under {raw_prefix}/grounding/",
              flush=True)
        return 0
    ocr_raw = gcs_store.read_text_abs(f"{raw_prefix}/ocr/ocr_dataset.jsonl")

    # reg_no -> [(page, text), ...] and reg_no -> has-any-pages, built once
    # in memory for this ingest pass (not persisted — only the lighter
    # page-image index in gcs_store is cached, for PDF viewing later).
    pages_by_deed = {}
    if ocr_raw:
        for line in ocr_raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            reg_no = str(o.get("reg_no") or "")
            if reg_no:
                pages_by_deed.setdefault(reg_no, []).append(
                    (o.get("page", 0), o.get("text", "")))
    for v in pages_by_deed.values():
        v.sort(key=lambda x: x[0])

    con = connect()
    loaded = skipped = failed = 0
    try:
        existing = {r["deed_number"] for r in
                    con.execute("SELECT deed_number FROM documents").fetchall()}
        print(f"[gcs-raw] {len(existing)} deeds already in DB (fast skip, no per-line query)",
              flush=True)
        lines = [l for l in graw.splitlines() if l.strip()]
        print(f"[gcs-raw] {len(lines)} deeds in grounding file", flush=True)
        for n, line in enumerate(lines, 1):
            try:
                g = json.loads(line)
                reg_no = str(g.get("reg_no") or "").strip()
                if not reg_no:
                    failed += 1
                    continue
                if reg_no in existing:
                    skipped += 1
                    continue
                pages = pages_by_deed.get(reg_no)
                full_text = None
                if pages:
                    parts = [f"— Page {pg} —\n{txt}".strip() for pg, txt in pages if txt]
                    full_text = "\n\n".join(parts) or None
                pdf_name = f"{reg_no}.pdf" if pages else None
                ok = _insert_from_grounding(con, g, full_text, pdf_name)
                if ok:
                    loaded += 1
                    existing.add(reg_no)
                elif ok is False:
                    skipped += 1
                else:
                    failed += 1
                if n % 500 == 0:
                    con.commit()
                    print(f"[gcs-raw] {n}/{len(lines)} processed ({loaded} loaded)", flush=True)
                    if progress:
                        progress(n, len(lines), loaded)
            except Exception as e:
                failed += 1
                try:
                    con.rollback()   # clear the aborted-transaction state, or every
                except Exception:    # line after this one would fail too
                    pass
                print(f"[gcs-raw] failed line {n} ({locals().get('reg_no', '?')}): {e}",
                      flush=True)
        con.commit()
    finally:
        con.close()
    print(f"[gcs-raw] done: {loaded} loaded, {skipped} already present, {failed} failed",
          flush=True)
    return loaded


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python ingest_json.py <data_dir | deed_folder>")
        sys.exit(1)
    p = Path(sys.argv[1])
    init_db()
    if (p / "grounding.json").exists():
        load_deed(p)
    else:
        ingest_dir(p, init=False)


def merge_existing_party_fields(con):
    """In-place migration: convert already-ingested per-item party fields
    (Seller 1 / Buyer 2 ... rows) into merged comma-separated fields, WITHOUT
    re-ingesting. Corrections are preserved (each item's current value joins
    the merged string) and edit history is repointed to the merged field.
    Idempotent: deeds already merged (or with no party fields) are untouched.

    This runs on every app startup (every deploy AND every restart), so the
    query MUST filter down to only the rows that still need migrating —
    pushed into SQL, not Python. The old version selected every row with
    src_block IS NOT NULL (which matches almost every field from ingestion,
    migrated or not) and fetched them ALL into memory before checking in a
    Python loop whether each one actually needed anything done. With ~10k+
    documents that's easily hundreds of thousands of rows loaded into
    memory on every single boot — including OOM-triggered restarts, which
    would then immediately repeat the same expensive load and could keep
    tripping the memory limit again right after "fixing" it. Once documents
    are migrated, this query now returns an empty (or near-empty) result
    set on every later startup instead of the whole table.
    Returns number of documents migrated."""
    import json as _json
    rows = con.execute(
        "SELECT id, document_id, section, label, ocr_value, current_value, "
        "odia_value, position, page_num, src_block FROM fields "
        "WHERE src_block IS NOT NULL "
        "AND src_block->>'id' IN ('seller_details','buyer_details','property_details') "
        "AND COALESCE((src_block->>'item_index')::int, 0) > 0 "
        "AND NOT (src_block ? 'group') "
        "ORDER BY document_id, position").fetchall()
    if not rows:
        return 0

    by_doc = {}
    for r in rows:
        sb = r["src_block"]
        if isinstance(sb, str):
            sb = _json.loads(sb)
        if not isinstance(sb, dict) or sb.get("group"):
            continue                      # already merged (belt-and-braces; SQL above already excludes these)
        fid = sb.get("id")
        idx = sb.get("item_index") or 0
        if fid not in SECTION_NAMES or not idx:
            continue                      # scalar field — untouched (also already excluded above)
        key = (fid, (sb.get("attr") or "").strip())
        by_doc.setdefault(r["document_id"], {}).setdefault(key, []).append(
            {**dict(r), "_sb": sb, "_idx": idx})

    migrated = 0
    for doc_id, groups in by_doc.items():
        for (fid, attr), items in groups.items():
            items.sort(key=lambda x: x["_idx"])
            n = max(i["_idx"] for i in items)
            english = ", ".join((i["current_value"] or "").strip() for i in items)
            ocr = ", ".join((i["ocr_value"] or "").strip() for i in items)
            odia = ", ".join((i["odia_value"] or "").strip() for i in items)
            base = SECTION_NAMES[fid]
            section = f"{SECTION_PLURALS[base]} ({n})" if n != 1 else base
            merged_block = {"group": True, "id": fid, "attr": attr,
                            "items": [i["_sb"] for i in items]}
            new_id = con.execute(
                "INSERT INTO fields (document_id, section, label, ocr_value, "
                "current_value, odia_value, multiline, position, field_kind, "
                "layout_tag, src_block, page_num) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,'text',%s,%s,%s) RETURNING id",
                (doc_id, section, _pretty_attr(attr), ocr, english, odia,
                 len(english) > 60, items[0]["position"], fid,
                 _json.dumps(merged_block), items[0]["page_num"])).fetchone()["id"]
            old_ids = [i["id"] for i in items]
            con.execute("UPDATE edit_log SET field_id=%s WHERE field_id = ANY(%s)",
                        (new_id, old_ids))
            con.execute("DELETE FROM fields WHERE id = ANY(%s)", (old_ids,))
        migrated += 1
        if migrated % 100 == 0:
            con.commit()
            print(f"[merge] {migrated} documents migrated...", flush=True)
    con.commit()
    return migrated
