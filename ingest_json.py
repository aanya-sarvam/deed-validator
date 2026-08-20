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
import os
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

# Scalar Deed-details fields that must ALWAYS appear in the portal, even when
# grounding omitted them (model didn't find them / merge only kept found=true
# rows). Mirrors the always-shown party template (Name / Address / …). Order
# is the display order under Deed details. Labels match the usual grounding
# `field` strings so export/src_block stay consistent.
CANON_DEED_DETAIL_FIELDS = (
    ("deed_type", "Deed type"),
    ("district", "District"),
    ("office", "Registration office"),
    ("registration_date", "Registration date"),
    ("presentation_date", "Presentation date"),
    ("consideration_amount", "Consideration Amount"),
    ("old_reg_no", "Old registration no"),
)

# Tabular / IGR-API aliases → canonical field id. The Gemini input for a deed
# is this shape (camelCase); english_value on each grounding field should
# echo it, but when the model omits a field entirely we still fill the portal
# box from these so reviewers never see a blank where the source metadata
# already had a value.
_TABULAR_SCALAR_ALIASES = {
    "deed_type": (
        "deed_type", "deedType", "DeedType", "documentType", "docType",
        "natureOfDocument", "nature", "documentName",
    ),
    "district": (
        "district", "districtName", "District", "district_name",
    ),
    "office": (
        "office", "sroName", "sro", "registrationOffice", "srOffice",
        "officeName", "RegistrationOffice", "subRegistrarOffice",
    ),
    "registration_date": (
        "registration_date", "registrationDate", "regDate",
        "dateOfRegistration", "RegistrationDate",
    ),
    "presentation_date": (
        "presentation_date", "presentationDate", "dateOfPresentation",
        "PresentationDate",
    ),
    "consideration_amount": (
        "consideration_amount", "considerationAmount", "consideration",
        "marketValue", "transactionValue", "ConsiderationAmount", "amount",
    ),
    "old_reg_no": (
        "old_reg_no", "oldRegNo", "oldRegNO", "oldRegistrationNo",
        "previousRegNo", "OldRegNo",
    ),
}


def _pretty_attr(attr):
    return ATTR_LABELS.get(attr, (attr or "value").replace("_", " ").title())


def _merge_enabled():
    import os
    # ON by default; set MERGE_PARTY_FIELDS=0 to keep per-item fields
    return os.environ.get("MERGE_PARTY_FIELDS", "1").lower() not in ("0", "false", "no")


# Deed categories confirmed so far to belong to Book 1 (the register of
# documents that transfer/create rights in immovable property — sale,
# gift, mortgage, exchange, partition, lease under the Indian Registration
# Act). Matched against EITHER book_label or deed_type since both have been
# observed to carry this category name in the real source data (e.g.
# book_label="SALE", deed_type="SALE IMMOVABLE"). Add more keywords here as
# they're confirmed (e.g. "gift", "mortgage") — matching is substring,
# case-insensitive, so "sale" also matches "SALE IMMOVABLE", "Sale Deed",
# etc. without needing every exact variant listed.
BOOK1_CATEGORY_KEYWORDS = {"sale"}


def _is_general_poa(book_label, deed_type=None):
    """True for a *General* Power of Attorney deed (not Special POA). Matches
    a value that mentions 'general' together with either 'power of attorney'
    (spelled out) or the abbreviation 'poa' — real data uses both forms (e.g.
    deed_type='GENERAL POA WITHOUT PROPERTY'). Explicitly excludes anything
    that also says 'special', so 'Special POA' never matches even if it
    somehow also mentioned 'general' elsewhere."""
    for value in (book_label, deed_type):
        if not value:
            continue
        norm = str(value).lower()
        if "special" in norm:
            continue
        mentions_poa = "power of attorney" in norm or re.search(r"\bpoa\b", norm)
        if mentions_poa and "general" in norm:
            return True
    return False


def _has_property(g):
    """Whether a grounding record has any Property data (a property_details
    or property_boundary field)."""
    for f in g.get("fields", []) or []:
        if str(f.get("id") or "").startswith("property"):
            return True
    return False


def _needs_consideration(g):
    """A Consideration Amount box (defaulted to 0) is shown for:
      - sale / immovable deeds (Book 1), always; and
      - General POA deeds that have NO property section.
    Special POA and property-bearing General POA are excluded."""
    book_label, deed_type = g.get("book_label"), g.get("deed_type")
    if _is_book1(book_label, deed_type):
        return True
    return _is_general_poa(book_label, deed_type) and not _has_property(g)


def _is_book1(book_label, deed_type=None):
    """Match Book 1 deeds by checking book_label and deed_type for any
    confirmed Book 1 category keyword (see BOOK1_CATEGORY_KEYWORDS).
    Originally this looked for literal 'Book 1' / 'Book I' text, based on
    an assumption about the source data format — checking against the
    actual data showed book_label instead holds a category name like
    'SALE', not a book number, so matching had to change to category
    keywords instead."""
    for value in (book_label, deed_type):
        if not value:
            continue
        norm = str(value).lower()
        if any(kw in norm for kw in BOOK1_CATEGORY_KEYWORDS):
            return True
    return False


def _ensure_consideration_amount(rows, g):
    """Book 1 deeds should always have a Consideration Amount field for
    reviewers to check — if the source data didn't extract one (missing
    from OCR/grounding), add it with a default of '0' rather than leaving
    it absent entirely, so it's always visible and editable.

    Inserted right after the 'Presentation date' field specifically —
    matching where a genuinely-extracted Consideration Amount field
    naturally sits in this data (Presentation date -> Consideration Amount
    -> Old Reg No). Falls back to the end of the Deed details block only
    if no Presentation date field exists for this deed. Simply appending
    to the very end of `rows` was wrong for two reasons: Deed details rows
    come first followed by merged party-group rows (Sellers/Buyers/
    Property), so appending at the absolute end put the field AFTER those
    party sections — splitting 'Deed details' into two separate blocks in
    the viewer, since the frontend starts a new section every time the
    section name changes while walking fields in position order; and even
    appending at the end of just the Deed details block still didn't match
    where this field naturally belongs relative to its neighbors.

    Also upgrades an empty auto-padded consideration row (from
    _ensure_deed_detail_fields) to the Book-1 default of '0'."""
    if not _needs_consideration(g):
        return rows
    for r in rows:
        if "consideration" not in (r.get("label") or "").lower():
            continue
        # Already have a real or defaulted value — leave it.
        if (r.get("english") or "").strip() or (r.get("odia") or "").strip():
            return rows
        sb = r.get("src_block") if isinstance(r.get("src_block"), dict) else {}
        if sb.get("auto_defaulted") or sb.get("auto_padded"):
            r["english"] = "0"
            r["odia"] = "0"
            sb = {**sb, "auto_defaulted": True, "id": "consideration_amount",
                  "field": "Consideration Amount"}
            r["src_block"] = sb
        return rows
    new_row = {
        "section": "Deed details",
        "label": "Consideration Amount",
        "english": "0",
        "odia": "0",
        "src_block": {"id": "consideration_amount", "field": "Consideration Amount",
                      "auto_defaulted": True},
        "page": None,
    }
    insert_at = None
    for i, r in enumerate(rows):
        if "presentation" in (r.get("label") or "").lower():
            insert_at = i + 1
    if insert_at is None:
        insert_at = 0
        for i, r in enumerate(rows):
            if r.get("section") == "Deed details":
                insert_at = i + 1
    return rows[:insert_at] + [new_row] + rows[insert_at:]


def _tabular_sources(g):
    """Yield dicts that may hold the original IGR/tabular metadata for a deed.

    Grounding JSONL lines vary: sometimes the camelCase input is at the top
    level, sometimes under meta/tabular/input/api, sometimes only deed_type
    survived on the header. We probe all of them."""
    if not isinstance(g, dict):
        return
    yield g
    for key in ("meta", "tabular", "input", "api", "api_meta", "igr",
                "raw_meta", "source_meta", "src_meta"):
        sub = g.get(key)
        if isinstance(sub, dict):
            yield sub
        elif isinstance(sub, str) and sub.strip().startswith("{"):
            try:
                parsed = json.loads(sub)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                yield parsed


def _scalar_from_tabular(g, field_id):
    """Pull the english/source value for a canon Deed-details id from tabular
    metadata on the grounding record (if present)."""
    aliases = _TABULAR_SCALAR_ALIASES.get(field_id) or (field_id,)
    for src in _tabular_sources(g):
        for a in aliases:
            if a not in src:
                continue
            val = src.get(a)
            if val is None:
                continue
            s = str(val).strip()
            if s and s.lower() not in ("", "-", "na", "n/a", "nil", "none",
                                       "null", "nan", "others"):
                return s
    return ""


def _grounding_scalar_map(g):
    """id → best field dict from g['fields'] for canon deed-detail scalars."""
    out = {}
    for f in (g.get("fields") or []) if isinstance(g, dict) else []:
        if not isinstance(f, dict):
            continue
        if f.get("attr"):
            continue  # party/property sub-attrs
        fid = (f.get("id") or "").strip()
        if fid not in dict(CANON_DEED_DETAIL_FIELDS):
            continue
        # Prefer found=true, then non-empty odia, then non-empty english.
        prev = out.get(fid)
        if prev is None:
            out[fid] = f
            continue
        cand = (bool(f.get("found")),
                1 if (f.get("odia_text") or "").strip() else 0,
                1 if (f.get("english_value") or "").strip() else 0,
                float(f.get("confidence") or 0))
        old = (bool(prev.get("found")),
               1 if (prev.get("odia_text") or "").strip() else 0,
               1 if (prev.get("english_value") or "").strip() else 0,
               float(prev.get("confidence") or 0))
        if cand > old:
            out[fid] = f
    return out


def _deed_detail_field_id(r):
    """Resolve the canonical scalar id for a Deed-details row, if any."""
    sb = r.get("src_block") if isinstance(r.get("src_block"), dict) else {}
    if sb.get("group") or sb.get("per_item"):
        return None
    fid = (sb.get("id") or r.get("layout_tag") or "").strip()
    if fid in dict(CANON_DEED_DETAIL_FIELDS):
        return fid
    # Fallback: match by label (handles older rows / short "Office" label).
    lab = (r.get("label") or "").strip().lower()
    for cid, clabel in CANON_DEED_DETAIL_FIELDS:
        if lab == clabel.lower() or lab == cid.replace("_", " "):
            return cid
        if cid == "office" and lab in ("office", "registration office", "sr office"):
            return cid
        if cid == "old_reg_no" and "old reg" in lab:
            return cid
        if cid == "consideration_amount" and "consideration" in lab:
            return cid
    return None


def _ensure_deed_detail_fields(rows, g=None):
    """Guarantee every canonical Deed-details scalar is present AND filled.

    Gemini is asked to return every tabular target (found true/false). Some
    completed-1k merges still drop unfound rows, so a deed can land with only
    District + Office. We:
      1. rebuild the Deed-details block in CANON order;
      2. for any missing/empty box, prefer the grounding field's
         english_value / odia_text when the model did return it;
      3. otherwise fill english from the original tabular/IGR metadata on
         the grounding record (deedType, presentationDate, …) so the portal
         never shows a blank where the input JSON already had a value.
    """
    g = g or {}
    from_grounding = _grounding_scalar_map(g)

    deed_rows, other_rows = [], []
    for r in rows:
        if r.get("section") == "Deed details":
            deed_rows.append(r)
        else:
            other_rows.append(r)

    by_id = {}
    extras = []
    for r in deed_rows:
        fid = _deed_detail_field_id(r)
        if fid and fid not in by_id:
            if fid == "office" and (r.get("label") or "").strip().lower() == "office":
                r = {**r, "label": "Registration office"}
                sb = r.get("src_block") if isinstance(r.get("src_block"), dict) else {}
                if sb:
                    r["src_block"] = {**sb, "field": "Registration office",
                                      "id": sb.get("id") or "office"}
            by_id[fid] = r
        elif not fid:
            extras.append(r)

    rebuilt = []
    for cid, clabel in CANON_DEED_DETAIL_FIELDS:
        gf = from_grounding.get(cid) or {}
        tab_en = _scalar_from_tabular(g, cid)
        if cid in by_id:
            r = dict(by_id[cid])
            # Fill empty english/odia from grounding field or tabular meta —
            # never wipe a value the reviewer / model already put there.
            if not (r.get("english") or "").strip():
                r["english"] = (gf.get("english_value") or tab_en or "")
            if not (r.get("odia") or "").strip():
                r["odia"] = (gf.get("odia_text") or "")
            # Keep odia in sync with latin dates/amounts when model left odia
            # empty but english carries the same digits (common for dates).
            if not (r.get("odia") or "").strip() and (r.get("english") or "").strip():
                # Only copy for date/amount/reg-no style fields — not names.
                if cid in ("registration_date", "presentation_date",
                           "consideration_amount", "old_reg_no"):
                    r["odia"] = r["english"]
            sb = r.get("src_block") if isinstance(r.get("src_block"), dict) else {}
            if sb is not None and gf:
                # Prefer richer src_block from grounding when we only had a pad.
                if sb.get("auto_padded") or sb.get("auto_defaulted"):
                    merged_sb = {**gf, "id": cid, "field": gf.get("field") or clabel}
                    r["src_block"] = merged_sb
            rebuilt.append(r)
        else:
            eng = (gf.get("english_value") or tab_en or "")
            odia = (gf.get("odia_text") or "")
            if not odia and eng and cid in ("registration_date", "presentation_date",
                                            "consideration_amount", "old_reg_no"):
                odia = eng
            sb = dict(gf) if gf else {
                "id": cid, "field": clabel, "attr": "", "item_index": 0,
                "found": False, "auto_padded": True,
            }
            sb.setdefault("id", cid)
            sb.setdefault("field", clabel)
            if not gf:
                sb["auto_padded"] = True
            if tab_en and not gf:
                sb["english_value"] = tab_en
                sb["from_tabular"] = True
            rebuilt.append({
                "section": "Deed details",
                "label": clabel,
                "english": eng,
                "odia": odia,
                "src_block": sb,
                "page": gf.get("page"),
            })
    return rebuilt + extras + other_rows


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


def _insert_from_grounding(con, g, full_text, pdf_name, source="classification"):
    """Shared insert used by both local and GCS paths. `source` records which
    GCS bucket/layout this deed's scans live in ('classification' | 'vertex')
    so the viewer can route page fetches to the right bucket+credentials."""
    reg_no = str(g.get("reg_no") or "").strip()
    if not reg_no:
        return None
    if con.execute("SELECT 1 FROM documents WHERE deed_number=%s",
                   (reg_no,)).fetchone():
        return False
    src_meta = {k: v for k, v in g.items() if k != "fields"}
    doc_id = con.execute(
        "INSERT INTO documents (deed_number, deed_type, year, pdf_file, status, "
        "digitized_text, digitized_status, src_meta, source) "
        "VALUES (%s,%s,%s,%s,'pending',%s,%s,%s,%s) RETURNING id",
        (reg_no, g.get("deed_type"), _year_from_fields(g.get("fields", [])),
         pdf_name, full_text,
         "ready" if full_text else "not_started",
         json.dumps(src_meta), source)).fetchone()["id"]
    rows = []
    field_rows = _ensure_consideration_amount(
        _ensure_deed_detail_fields(_build_field_rows(g.get("fields", [])), g), g)
    for i, r in enumerate(field_rows):
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


def ingest_gcs_vertex(init=True, progress=None):
    """Ingest the Gemini-mismatch batch (the 502) from the vision-vertex-batch
    bucket — a SECOND GCS source, read with its own service-account key.

    Reads a single grounding JSONL you upload to the bucket (one deed per
    line, same field shape as grounding_good_partial.jsonl:
    {reg_no, deed_type, fields:[{id, attr, item_index, field, english_value,
    odia_text, page}, ...]}). Path is GCS_VERTEX_GROUNDING (default
    'outputs/grounding_good_partial.jsonl'); optional full text from
    GCS_VERTEX_OCR (an ocr_dataset.jsonl). Rows are inserted with
    source='vertex' so the viewer serves their page images from the vertex
    bucket (inputs/grounding/images/<reg_no>/*.png). Safe to re-run:
    existing deed_numbers are skipped.
    """
    import gcs_store
    if not gcs_store.vertex_enabled():
        print("[gcs-vertex] GCS_VERTEX_BUCKET not set — skipping", flush=True)
        return 0
    if init:
        init_db()
    gpath = os.environ.get("GCS_VERTEX_GROUNDING",
                           "outputs/grounding_good_partial.jsonl")
    print(f"[gcs-vertex] reading grounding from gs://{os.environ.get('GCS_VERTEX_BUCKET')}/{gpath}",
          flush=True)
    graw = gcs_store.read_text_abs(gpath, source="vertex")
    if not graw:
        print(f"[gcs-vertex] grounding file not found at {gpath}", flush=True)
        return 0

    # optional OCR full text
    pages_by_deed = {}
    opath = os.environ.get("GCS_VERTEX_OCR", "").strip()
    if opath:
        ocr_raw = gcs_store.read_text_abs(opath, source="vertex")
        if ocr_raw:
            for line in ocr_raw.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    o = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rn = str(o.get("reg_no") or "")
                if rn:
                    pages_by_deed.setdefault(rn, []).append(
                        (o.get("page", 0), o.get("text", "")))
            for v in pages_by_deed.values():
                v.sort(key=lambda x: x[0])

    import psycopg

    def _fresh():
        # keepalives reduce idle drops; autocommit-style per-row commits below
        # make a mid-run disconnect lose at most the current row (which re-runs
        # skip anyway), instead of a whole uncommitted batch.
        return connect()

    con = _fresh()
    loaded = skipped = failed = 0
    try:
        existing = {r["deed_number"] for r in
                    con.execute("SELECT deed_number FROM documents").fetchall()}
        lines = [l for l in graw.splitlines() if l.strip()]
        print(f"[gcs-vertex] {len(lines)} deeds in grounding file "
              f"({len(existing)} already in DB)", flush=True)
        for n, line in enumerate(lines, 1):
            try:
                g = json.loads(line)
            except json.JSONDecodeError:
                failed += 1
                continue
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

            # Insert with reconnect-and-retry: on a dropped/lost connection,
            # rebuild it and retry this one deed once before giving up on it.
            for attempt in (1, 2):
                try:
                    ok = _insert_from_grounding(con, g, full_text, f"{reg_no}.pdf",
                                                source="vertex")
                    con.commit()   # commit each deed so progress always persists
                    if ok:
                        loaded += 1
                        existing.add(reg_no)
                    elif ok is False:
                        skipped += 1
                    else:
                        failed += 1
                    break
                except (psycopg.OperationalError, psycopg.InterfaceError) as e:
                    print(f"[gcs-vertex] connection dropped at line {n} "
                          f"({reg_no}), reconnecting (attempt {attempt}): {e}",
                          flush=True)
                    try:
                        con.close()
                    except Exception:
                        pass
                    if attempt == 2:
                        failed += 1
                    else:
                        con = _fresh()
                except Exception as e:
                    try:
                        con.rollback()
                    except Exception:
                        con = _fresh()
                    failed += 1
                    print(f"[gcs-vertex] failed line {n} ({reg_no}): {e}", flush=True)
                    break

            if n % 100 == 0:
                print(f"[gcs-vertex] {n}/{len(lines)} ({loaded} loaded, "
                      f"{skipped} skipped, {failed} failed)", flush=True)
                if progress:
                    progress(n, len(lines), loaded)
    finally:
        try:
            con.close()
        except Exception:
            pass
    print(f"[gcs-vertex] done: {loaded} loaded, {skipped} already present, "
          f"{failed} failed", flush=True)
    return loaded


def ingest_gcs_completed_1k(init=True, progress=None):
    """Ingest the 999 completed-1k deeds (prompt_v2 rerun over the fully
    populated 1985-1994 batch-2 cohort) from the SAME vertex bucket/SA as
    ingest_gcs_vertex, but tagged source='completed_1k' and priority_rank=30
    so they sort above the mismatch batch. Reads a per-deed grounding JSONL
    (the merge_local.py output, already in {reg_no, deed_type, fields:[...]}
    shape) at GCS_COMPLETED1K_GROUNDING. Safe to re-run: existing
    deed_numbers are skipped."""
    import gcs_store
    if not gcs_store.vertex_enabled():
        print("[completed1k] GCS_VERTEX_BUCKET not set — skipping", flush=True)
        return 0
    if init:
        init_db()
    gpath = os.environ.get("GCS_COMPLETED1K_GROUNDING",
                           "outputs/merged/grounding_results_vx_v2_1k.jsonl")
    print(f"[completed1k] reading grounding from "
          f"gs://{os.environ.get('GCS_VERTEX_BUCKET')}/{gpath}", flush=True)
    graw = gcs_store.read_text_abs(gpath, source="vertex")
    if not graw:
        print(f"[completed1k] grounding file not found at {gpath}", flush=True)
        return 0

    import psycopg
    con = connect()
    loaded = skipped = failed = 0
    try:
        existing = {r["deed_number"] for r in
                    con.execute("SELECT deed_number FROM documents").fetchall()}
        lines = [l for l in graw.splitlines() if l.strip()]
        print(f"[completed1k] {len(lines)} deeds in grounding file "
              f"({len(existing)} already in DB)", flush=True)
        for n, line in enumerate(lines, 1):
            try:
                g = json.loads(line)
            except json.JSONDecodeError:
                failed += 1
                continue
            reg_no = str(g.get("reg_no") or "").strip()
            if not reg_no:
                failed += 1
                continue
            if reg_no in existing:
                skipped += 1
                continue
            for attempt in (1, 2):
                try:
                    ok = _insert_from_grounding(con, g, None, f"{reg_no}.pdf",
                                                source="completed_1k")
                    if ok:
                        # rank 30 = above mismatch(20); set right after insert
                        con.execute("UPDATE documents SET priority_rank = 30 "
                                    "WHERE deed_number = %s", (reg_no,))
                        loaded += 1
                        existing.add(reg_no)
                    elif ok is False:
                        skipped += 1
                    else:
                        failed += 1
                    con.commit()
                    break
                except (psycopg.OperationalError, psycopg.InterfaceError) as e:
                    print(f"[completed1k] conn dropped at line {n} ({reg_no}), "
                          f"reconnect (attempt {attempt}): {e}", flush=True)
                    try:
                        con.close()
                    except Exception:
                        pass
                    if attempt == 2:
                        failed += 1
                    else:
                        con = connect()
                except Exception as e:
                    try:
                        con.rollback()
                    except Exception:
                        con = connect()
                    failed += 1
                    print(f"[completed1k] failed {reg_no}: {e}", flush=True)
                    break
            if progress and n % 25 == 0:
                progress(n, len(lines), loaded)
    finally:
        try:
            con.close()
        except Exception:
            pass
    print(f"[completed1k] done: {loaded} loaded, {skipped} present, "
          f"{failed} failed", flush=True)
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
        "AND NOT (src_block ? 'per_item') "
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


def backfill_book1_consideration(con):
    """One-time backfill for documents ingested BEFORE the default-
    Consideration-Amount rule existed: Book 1 deeds (identified by
    BOOK1_CATEGORY_KEYWORDS matching book_label or deed_type — see
    _is_book1) missing a Consideration Amount field get one added,
    defaulted to '0'.

    The keyword list lives in ONE place (BOOK1_CATEGORY_KEYWORDS) and this
    SQL filter is built from it directly, so this can never drift out of
    sync with the Python-side _is_book1 check used at ingest time.

    Runs on every startup like the other migrations here, so the query
    MUST stay filtered in SQL rather than pulling documents into Python to
    check one at a time — once every Book 1 deed has been backfilled, this
    returns an empty result set instantly on every later boot instead of
    scanning the whole table (see merge_existing_party_fields for why that
    distinction matters — an unfiltered full-table pull on every startup
    was a real cause of repeat OOM restarts previously).
    Returns number of documents backfilled."""
    # Eligibility mirrors _needs_consideration():
    #   sale (Book 1)  OR  (General POA AND the deed has no Property section).
    sale_conditions = []
    sale_params = []
    for kw in BOOK1_CATEGORY_KEYWORDS:
        sale_conditions.append(
            "(lower(COALESCE(d.src_meta->>'book_label','')) LIKE %s "
            "OR lower(COALESCE(d.deed_type,'')) LIKE %s)")
        sale_params.extend([f"%{kw}%", f"%{kw}%"])
    where_sale = "(" + " OR ".join(sale_conditions) + ")"

    # General POA: mentions 'general' together with either 'power of attorney'
    # (spelled out) or the abbreviation 'poa' (word-boundary, via ~*), and does
    # NOT mention 'special' — mirrors _is_general_poa exactly.
    def _gpoa_cond(col):
        return (
            f"(lower(COALESCE({col},'')) NOT LIKE '%%special%%' "
            f"AND lower(COALESCE({col},'')) LIKE '%%general%%' "
            f"AND (lower(COALESCE({col},'')) LIKE '%%power of attorney%%' "
            f"     OR COALESCE({col},'') ~* '\\ypoa\\y'))")
    book_col = "d.src_meta->>'book_label'"
    where_gpoa = (
        f"(({_gpoa_cond(book_col)}) "
        f" OR ({_gpoa_cond('d.deed_type')})) "
        " AND NOT EXISTS ("
        "   SELECT 1 FROM fields fp WHERE fp.document_id = d.id "
        "   AND (fp.src_block->>'id' IN ('property_details','property_boundary') "
        "        OR fp.section ILIKE 'propert%%'))")

    where_eligible = f"({where_sale} OR {where_gpoa})"
    params = ["%presentation%"] + sale_params + ["%consideration%"]
    rows = con.execute(
        "SELECT d.id, "
        "  COALESCE("
        "    (SELECT f.position + 1 FROM fields f WHERE f.document_id = d.id "
        "     AND lower(f.label) LIKE %s ORDER BY f.position DESC LIMIT 1), "
        "    (SELECT COALESCE(MAX(f2.position), -1) + 1 FROM fields f2 "
        "     WHERE f2.document_id = d.id AND f2.section = 'Deed details')"
        "  ) AS insert_pos "
        "FROM documents d "
        f"WHERE {where_eligible} "
        "AND NOT EXISTS ("
        "  SELECT 1 FROM fields f WHERE f.document_id = d.id "
        "  AND lower(f.label) LIKE %s"
        ")", params).fetchall()
    if not rows:
        return 0
    src_block = json.dumps({"id": "consideration_amount",
                             "field": "Consideration Amount", "auto_defaulted": True})
    for i, r in enumerate(rows, 1):
        # Make room right after the Presentation date field (or end of Deed
        # details as a fallback), instead of tacking the new field onto the
        # very end of the document (which would land it after any Sellers/
        # Buyers/Property sections and split "Deed details" into two
        # separate blocks in the viewer).
        con.execute(
            "UPDATE fields SET position = position + 1 "
            "WHERE document_id = %s AND position >= %s",
            (r["id"], r["insert_pos"]))
        con.execute(
            "INSERT INTO fields (document_id, section, label, ocr_value, "
            "current_value, odia_value, multiline, position, field_kind, "
            "layout_tag, src_block, page_num) "
            "VALUES (%s,'Deed details','Consideration Amount','0','0','0', "
            "false,%s,'text','consideration_amount',%s,NULL)",
            (r["id"], r["insert_pos"], src_block))
        if i % 200 == 0:
            con.commit()
            print(f"[book1-backfill] {i}/{len(rows)} documents backfilled...", flush=True)
    con.commit()
    return len(rows)


def migrate_boundaries_into_property(con):
    """Move legacy section-level plot boundaries (fields with
    src_block.id='property_boundary', one set per deed) into the Property
    group as per-property boundary attributes (boundary_north/south/east/west
    on item 1), so boundaries render inside each Property card under a
    'Boundaries' heading instead of as a separate block.

    SQL-filtered so it's a no-op once done (returns instantly on later boots).
    Non-directional/blank leftovers are dropped. Returns fields migrated."""
    rows = con.execute(
        "SELECT id, document_id, section, label, ocr_value, current_value, "
        "odia_value, position, src_block FROM fields "
        "WHERE src_block->>'id' = 'property_boundary'").fetchall()
    if not rows:
        return 0
    migrated = 0
    for r in rows:
        sb = r["src_block"]
        if isinstance(sb, str):
            sb = json.loads(sb)
        d = (sb.get("boundary") or "").lower()
        if d not in ("north", "south", "east", "west"):
            con.execute("UPDATE edit_log SET field_id=NULL WHERE field_id=%s", (r["id"],))
            con.execute("DELETE FROM fields WHERE id=%s", (r["id"],))
            continue
        attr = "boundary_" + d
        # already have a group boundary attr for this doc? then just drop legacy.
        ex = con.execute(
            "SELECT id FROM fields WHERE document_id=%s "
            "AND src_block->>'id'='property_details' AND src_block->>'attr'=%s "
            "AND (src_block ? 'group') LIMIT 1", (r["document_id"], attr)).fetchone()
        if ex:
            con.execute("UPDATE edit_log SET field_id=%s WHERE field_id=%s", (ex["id"], r["id"]))
            con.execute("DELETE FROM fields WHERE id=%s", (r["id"],))
            continue
        prop = con.execute(
            "SELECT section FROM fields WHERE document_id=%s "
            "AND src_block->>'id'='property_details' AND (src_block ? 'group') LIMIT 1",
            (r["document_id"],)).fetchone()
        section = prop["section"] if prop else "Property"
        template = {"id": "property_details", "attr": attr, "item_index": 1,
                    "field": d.title(), "english_value": r["current_value"] or "",
                    "odia_text": r["odia_value"] or "", "found": False}
        merged = {"group": True, "id": "property_details", "attr": attr,
                  "items": [template]}
        new_id = con.execute(
            "INSERT INTO fields (document_id, section, label, ocr_value, current_value, "
            "odia_value, multiline, position, field_kind, src_block) "
            "VALUES (%s,%s,%s,%s,%s,%s,false,%s,'text',%s) RETURNING id",
            (r["document_id"], section, d.title(), r["ocr_value"] or "",
             r["current_value"] or "", r["odia_value"] or "", r["position"],
             json.dumps(merged))).fetchone()["id"]
        con.execute("UPDATE edit_log SET field_id=%s WHERE field_id=%s", (new_id, r["id"]))
        con.execute("DELETE FROM fields WHERE id=%s", (r["id"],))
        migrated += 1
        if migrated % 200 == 0:
            con.commit()
            print(f"[boundary-migrate] {migrated} boundaries moved...", flush=True)
    con.commit()
    return migrated


def reposition_consideration_amount(con):
    """One-time repositioning fix for documents already backfilled by an
    EARLIER, buggy version of backfill_book1_consideration /
    _ensure_consideration_amount, which either appended the auto-added
    Consideration Amount field after ALL other fields (including any
    Sellers/Buyers/Property sections) or just at the end of the Deed
    details block — instead of right after 'Presentation date', matching
    where this field naturally sits (Presentation date -> Consideration
    Amount -> Old Reg No). That placement is fixed for anything processed
    from now on, but backfill_book1_consideration is idempotent (skips
    documents that already have ANY Consideration Amount field), so it
    won't repair already-backfilled documents on its own — this migration
    does that specifically, moving them to the exact same target spot the
    other two functions now use.

    Only ever touches fields we auto-added ourselves
    (layout_tag='consideration_amount' AND src_block auto_defaulted=true)
    — a genuinely OCR-extracted Consideration Amount field is never moved.
    Returns number of fields repositioned."""
    rows = con.execute(
        "SELECT f.id AS field_id, f.document_id, f.position AS cur_pos, "
        "  COALESCE("
        "    (SELECT f2.position + 1 FROM fields f2 WHERE f2.document_id = f.document_id "
        "     AND lower(f2.label) LIKE %s AND f2.id != f.id "
        "     ORDER BY f2.position DESC LIMIT 1), "
        "    (SELECT COALESCE(MAX(f3.position), -1) + 1 FROM fields f3 "
        "     WHERE f3.document_id = f.document_id AND f3.section = 'Deed details' "
        "     AND f3.id != f.id)"
        "  ) AS correct_pos "
        "FROM fields f "
        "WHERE f.layout_tag = 'consideration_amount' "
        "AND (f.src_block->>'auto_defaulted') = 'true'",
        ["%presentation%"]).fetchall()
    to_fix = [r for r in rows if r["cur_pos"] != r["correct_pos"]]
    if not to_fix:
        return 0
    for i, r in enumerate(to_fix, 1):
        con.execute(
            "UPDATE fields SET position = position + 1 "
            "WHERE document_id = %s AND position >= %s AND id != %s",
            (r["document_id"], r["correct_pos"], r["field_id"]))
        con.execute("UPDATE fields SET position = %s WHERE id = %s",
                    (r["correct_pos"], r["field_id"]))
        if i % 200 == 0:
            con.commit()
            print(f"[consideration-reposition] {i}/{len(to_fix)} fixed...", flush=True)
    con.commit()
    return len(to_fix)


def backfill_deed_detail_fields(con):
    """Pad already-ingested documents that are missing canonical Deed-details
    scalars (Deed type, dates, consideration, old reg no, …). Same rule as
    _ensure_deed_detail_fields at ingest — needed because completed-1k (and
    some earlier batches) only stored found=true fields.

    SQL-scoped to documents that are actually missing at least one canon id,
    so later boots are a near-no-op. Also renames short 'Office' labels to
    'Registration office'. Returns number of documents touched."""
    canon_ids = [cid for cid, _ in CANON_DEED_DETAIL_FIELDS]
    missing_docs = con.execute(
        """
        SELECT d.id
        FROM documents d
        WHERE EXISTS (SELECT 1 FROM fields f WHERE f.document_id = d.id)
          AND EXISTS (
            SELECT 1 FROM unnest(%s::text[]) AS need(id)
            WHERE NOT EXISTS (
              SELECT 1 FROM fields f
              WHERE f.document_id = d.id
                AND f.section = 'Deed details'
                AND NOT COALESCE((f.src_block ? 'group'), false)
                AND (
                  f.layout_tag = need.id
                  OR f.src_block->>'id' = need.id
                  OR (need.id = 'office' AND lower(f.label) IN
                        ('office','registration office','sr office'))
                  OR (need.id = 'old_reg_no' AND lower(f.label) LIKE '%%old reg%%')
                  OR (need.id = 'consideration_amount'
                      AND lower(f.label) LIKE '%%consideration%%')
                  OR lower(f.label) = replace(need.id, '_', ' ')
                )
            )
          )
        """,
        (canon_ids,)).fetchall()

    renamed = con.execute(
        """
        UPDATE fields
        SET label = 'Registration office',
            layout_tag = COALESCE(NULLIF(layout_tag, ''), 'office'),
            src_block = COALESCE(src_block, '{}'::jsonb)
                         || jsonb_build_object('id', 'office',
                                               'field', 'Registration office')
        WHERE section = 'Deed details'
          AND lower(label) = 'office'
        """).rowcount

    if not missing_docs and not renamed:
        return 0

    touched = 0
    for i, doc in enumerate(missing_docs, 1):
        doc_id = doc["id"]
        existing = con.execute(
            "SELECT id, label, layout_tag, src_block, position FROM fields "
            "WHERE document_id = %s AND section = 'Deed details' "
            "ORDER BY position", (doc_id,)).fetchall()
        have = set()
        for r in existing:
            sb = r["src_block"]
            if isinstance(sb, str):
                sb = json.loads(sb)
            fake = {"label": r["label"], "layout_tag": r["layout_tag"],
                    "src_block": sb if isinstance(sb, dict) else {}}
            fid = _deed_detail_field_id(fake)
            if fid:
                have.add(fid)

        end_pos = con.execute(
            "SELECT COALESCE(MAX(position), -1) + 1 AS p FROM fields "
            "WHERE document_id = %s AND section = 'Deed details'",
            (doc_id,)).fetchone()["p"]
        to_add = [(cid, clabel) for cid, clabel in CANON_DEED_DETAIL_FIELDS
                  if cid not in have]
        if not to_add:
            continue
        con.execute(
            "UPDATE fields SET position = position + %s "
            "WHERE document_id = %s AND position >= %s",
            (len(to_add), doc_id, end_pos))
        meta = con.execute(
            "SELECT deed_type, src_meta FROM documents WHERE id=%s",
            (doc_id,)).fetchone()
        g = {}
        if meta:
            sm = meta["src_meta"]
            if isinstance(sm, dict):
                g = dict(sm)
            elif sm:
                g = json.loads(sm)
            g.setdefault("deed_type", meta["deed_type"])
        for j, (cid, clabel) in enumerate(to_add):
            # Prefer tabular/IGR english on src_meta so padded boxes aren't blank
            # when the input JSON already had the value (deedType, etc.).
            english = _scalar_from_tabular(g, cid)
            odia = ""
            sb = {"id": cid, "field": clabel, "attr": "", "item_index": 0,
                  "found": False, "auto_padded": True}
            if english:
                sb["english_value"] = english
                sb["from_tabular"] = True
            if cid == "consideration_amount" and not english and _needs_consideration(g):
                english = odia = "0"
                sb["auto_defaulted"] = True
            elif english and cid in ("registration_date", "presentation_date",
                                     "consideration_amount", "old_reg_no"):
                # Latin dates/amounts: seed Odia box with the same digits so
                # the visible field isn't empty when Gemini omitted the row.
                odia = english
            con.execute(
                "INSERT INTO fields (document_id, section, label, ocr_value, "
                "current_value, odia_value, multiline, position, field_kind, "
                "layout_tag, src_block, page_num) "
                "VALUES (%s,'Deed details',%s,%s,%s,%s,false,%s,'text',%s,%s,NULL)",
                (doc_id, clabel, english, english, odia, end_pos + j, cid,
                 json.dumps(sb)))
        touched += 1
        if i % 200 == 0:
            con.commit()
            print(f"[deed-detail-backfill] {i}/{len(missing_docs)} docs...", flush=True)
    con.commit()
    if renamed:
        print(f"[deed-detail-backfill] renamed {renamed} 'Office' → "
              f"'Registration office'", flush=True)
    return touched


def sync_deed_details_from_completed1k(con=None, progress=None):
    """Re-read the completed-1k grounding JSONL and fill any Deed-details
    fields that are missing or still empty in the DB.

    Use when Gemini/tabular had the values but an earlier ingest only kept
    found=true rows (District/Office only). For each JSONL line we:
      - take english/odia from the grounding fields array when present;
      - else fall back to camelCase tabular keys on the same record
        (deedType, presentationDate, considerationAmount, oldRegNO, …);
      - INSERT missing field rows or UPDATE empty current_value/odia_value
        (never overwrite a non-empty expert correction).

    Returns {docs_touched, fields_inserted, fields_updated}."""
    import gcs_store
    own_con = con is None
    if own_con:
        con = connect()
    stats = {"docs_touched": 0, "fields_inserted": 0, "fields_updated": 0}
    if not gcs_store.vertex_enabled():
        print("[deed-detail-sync] GCS_VERTEX_BUCKET not set — skipping", flush=True)
        if own_con:
            con.close()
        return stats
    gpath = os.environ.get("GCS_COMPLETED1K_GROUNDING",
                           "outputs/merged/grounding_results_vx_v2_1k.jsonl")
    print(f"[deed-detail-sync] reading {gpath}", flush=True)
    graw = gcs_store.read_text_abs(gpath, source="vertex")
    if not graw:
        print(f"[deed-detail-sync] grounding file not found at {gpath}", flush=True)
        if own_con:
            con.close()
        return stats

    by_reg = {r["deed_number"]: r["id"] for r in con.execute(
        "SELECT id, deed_number FROM documents WHERE source='completed_1k'"
    ).fetchall()}
    if not by_reg:
        print("[deed-detail-sync] no completed_1k documents in DB", flush=True)
        if own_con:
            con.close()
        return stats

    lines = [l for l in graw.splitlines() if l.strip()]
    for n, line in enumerate(lines, 1):
        try:
            g = json.loads(line)
        except json.JSONDecodeError:
            continue
        reg_no = str(g.get("reg_no") or g.get("key") or
                     g.get("registration_no") or g.get("registrationNo") or "").strip()
        doc_id = by_reg.get(reg_no)
        if not doc_id:
            continue

        # Build the desired deed-detail rows from this grounding record.
        desired = _ensure_deed_detail_fields([], g)
        desired = _ensure_consideration_amount(desired, g)
        desired = [r for r in desired if r.get("section") == "Deed details"]
        if not desired:
            continue

        existing = con.execute(
            "SELECT id, label, layout_tag, src_block, current_value, odia_value, "
            "ocr_value, position FROM fields "
            "WHERE document_id=%s AND section='Deed details' ORDER BY position",
            (doc_id,)).fetchall()
        have = {}
        for r in existing:
            sb = r["src_block"]
            if isinstance(sb, str):
                sb = json.loads(sb)
            fake = {"label": r["label"], "layout_tag": r["layout_tag"],
                    "src_block": sb if isinstance(sb, dict) else {}}
            fid = _deed_detail_field_id(fake)
            if fid:
                have[fid] = r

        doc_changed = False
        end_pos = con.execute(
            "SELECT COALESCE(MAX(position), -1) + 1 AS p FROM fields "
            "WHERE document_id=%s AND section='Deed details'",
            (doc_id,)).fetchone()["p"]

        for r in desired:
            cid = _deed_detail_field_id(r)
            if not cid:
                continue
            eng = (r.get("english") or "").strip()
            odia = (r.get("odia") or "").strip()
            if cid in have:
                cur = have[cid]
                sets, params = [], []
                # Only fill empties — never clobber expert edits.
                if eng and not (cur["current_value"] or "").strip():
                    sets.append("current_value=%s")
                    params.append(eng)
                    if not (cur["ocr_value"] or "").strip():
                        sets.append("ocr_value=%s")
                        params.append(eng)
                if odia and not (cur["odia_value"] or "").strip():
                    sets.append("odia_value=%s")
                    params.append(odia)
                if (cur["label"] or "").strip().lower() == "office":
                    sets.append("label=%s")
                    params.append("Registration office")
                if sets:
                    params.append(cur["id"])
                    con.execute(
                        f"UPDATE fields SET {', '.join(sets)} WHERE id=%s", params)
                    stats["fields_updated"] += 1
                    doc_changed = True
            else:
                if not eng and not odia:
                    # Still insert the empty template so the box appears;
                    # consideration default handled by _ensure_consideration_amount.
                    pass
                sb = r.get("src_block") if isinstance(r.get("src_block"), dict) else {
                    "id": cid, "field": r["label"], "auto_padded": True}
                con.execute(
                    "UPDATE fields SET position = position + 1 "
                    "WHERE document_id=%s AND position >= %s",
                    (doc_id, end_pos))
                con.execute(
                    "INSERT INTO fields (document_id, section, label, ocr_value, "
                    "current_value, odia_value, multiline, position, field_kind, "
                    "layout_tag, src_block, page_num) "
                    "VALUES (%s,'Deed details',%s,%s,%s,%s,false,%s,'text',%s,%s,NULL)",
                    (doc_id, r["label"], eng, eng, odia, end_pos, cid,
                     json.dumps(sb)))
                end_pos += 1
                stats["fields_inserted"] += 1
                doc_changed = True

        if doc_changed:
            stats["docs_touched"] += 1
        if progress and n % 50 == 0:
            progress(n, len(lines), stats["docs_touched"])
        if n % 100 == 0:
            con.commit()
            print(f"[deed-detail-sync] {n}/{len(lines)} "
                  f"({stats['docs_touched']} docs, "
                  f"+{stats['fields_inserted']} / ~{stats['fields_updated']})",
                  flush=True)
    con.commit()
    print(f"[deed-detail-sync] done: {stats}", flush=True)
    if own_con:
        con.close()
    return stats
