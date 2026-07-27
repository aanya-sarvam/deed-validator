#!/usr/bin/env python3
"""
compare_metadata.py — find mismatches between official Sarvam/IGR deed
metadata and the OCR/grounding metadata we hold in GCS (or local data/).

Typical Book-1 run:

    export SARVAM_BASE_URL='https://<shared-api-host>'
    # optional GCS_*: same vars the app already uses
    python compare_metadata.py --book 1 \\
        --out data/mismatches/book1_mismatches.json \\
        --update-db

Then open the frontend Records tab and filter to "API mismatches".

What it does
------------
1. Collect registration numbers from GCS / local grounding / the DB
   (or from GetDeedRegNoDetail over a date range).
2. Keep only the requested book(s) — Book 1 / 3 / 4.
3. For each reg_no, call GetDeedInfoByRegNo and load the matching
   grounding.json (GCS or local).
4. Normalize + compare key fields; write a JSON report of mismatches only.
5. Optionally stamp documents.has_api_mismatch / api_mismatch_detail so
   the frontend can filter and highlight them.

Probe a single registration (prints raw API JSON + mapped fields):

    python compare_metadata.py --probe 910010000401
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable

from sarvam_client import (
    SarvamClient,
    _as_book_int,
    _pick,
    book_no_from_api_deed,
    is_empty_deed_payload,
)

# Book 1 = documents that transfer/create rights in immovable property
# (Registration Act). Matched on book_label / deed_type when the API does
# not return an explicit book number — same idea as ingest_json.BOOK1_*.
BOOK_CATEGORY_KEYWORDS = {
    1: {"sale", "gift", "mortgage", "lease", "exchange", "partition",
        "conveyance", "sale immovable", "agreement to sell"},
    3: {"will", "authority to adopt", "adoption"},
    4: {"power of attorney", "poa", "general power", "special power",
        "miscellaneous", "agreement", "bond", "affidavit"},
}

# Scalar fields compared between official API metadata and GCS grounding.
# Each entry: (canonical_key, api_aliases, grounding_field_id)
SCALAR_FIELDS = [
    ("deed_type",
     ("deedType", "deed_type", "documentType", "docType", "natureOfDocument",
      "nature", "DeedType", "documentName"),
     "deed_type"),
    ("district",
     ("district", "districtName", "District", "district_name"),
     "district"),
    ("office",
     ("office", "sroName", "sro", "registrationOffice", "srOffice",
      "officeName", "RegistrationOffice", "subRegistrarOffice"),
     "office"),
    ("registration_date",
     ("registrationDate", "registration_date", "regDate", "dateOfRegistration",
      "RegistrationDate"),
     "registration_date"),
    ("presentation_date",
     ("presentationDate", "presentation_date", "dateOfPresentation",
      "PresentationDate"),
     "presentation_date"),
    ("consideration_amount",
     ("considerationAmount", "consideration_amount", "consideration",
      "marketValue", "transactionValue", "ConsiderationAmount", "amount"),
     "consideration_amount"),
    ("old_reg_no",
     ("oldRegNo", "old_reg_no", "oldRegistrationNo", "previousRegNo",
      "OldRegNo"),
     "old_reg_no"),
]

PARTY_API_KEYS = {
    "seller": ("executant", "executants", "seller", "sellers", "firstParty",
               "party1", "executantDetails", "sellerDetails", "Vendors",
               "vendor", "transferor"),
    "buyer": ("claimant", "claimants", "buyer", "buyers", "secondParty",
              "party2", "claimantDetails", "buyerDetails", "Vendees",
              "vendee", "transferee"),
}

PROPERTY_API_KEYS = (
    "property", "properties", "propertyDetails", "PropertyDetails",
    "schedule", "schedules", "landDetails",
)


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

_SPACE_RE = re.compile(r"\s+")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_DATE_RE = re.compile(
    r"(\d{1,2})[\-/\s.]([A-Za-z]{3}|\d{1,2})[\-/\s.](\d{2,4})")
_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def norm_text(value: Any) -> str:
    if value is None:
        return ""
    s = str(value).strip().lower()
    s = s.replace("&", " and ")
    s = _SPACE_RE.sub(" ", s)
    return s


def norm_amount(value: Any) -> str:
    s = norm_text(value)
    s = s.replace(",", "").replace("rs.", "").replace("rs", "").replace("/-", "")
    s = s.replace("inr", "").strip()
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    if not m:
        return s
    num = m.group(0)
    if "." in num:
        num = num.rstrip("0").rstrip(".")
    return num


def norm_date(value: Any) -> str:
    s = norm_text(value)
    if not s:
        return ""
    m = _DATE_RE.search(s)
    if not m:
        return s
    d, mon, y = m.group(1), m.group(2), m.group(3)
    if mon.isalpha():
        month = _MONTHS.get(mon[:3].lower(), 0)
    else:
        month = int(mon)
    year = int(y)
    if year < 100:
        year = 2000 + year if year <= 30 else 1900 + year
    if not month:
        return s
    return f"{year:04d}-{month:02d}-{int(d):02d}"


def norm_name(value: Any) -> str:
    s = norm_text(value)
    s = _NON_ALNUM_RE.sub(" ", s)
    return _SPACE_RE.sub(" ", s).strip()


def values_equal(field: str, a: Any, b: Any) -> bool:
    if field in ("consideration_amount",):
        return norm_amount(a) == norm_amount(b)
    if field.endswith("_date") or "date" in field:
        na, nb = norm_date(a), norm_date(b)
        if na and nb:
            return na == nb
    if field in ("seller_names", "buyer_names", "property_plots"):
        return set(a or []) == set(b or [])
    return norm_text(a) == norm_text(b)


# ---------------------------------------------------------------------------
# GCS / local grounding loaders
# ---------------------------------------------------------------------------

def load_grounding_index(local_dir: str | None = None) -> dict[str, dict]:
    """reg_no -> grounding dict. Prefers GCS when configured, else local."""
    out: dict[str, dict] = {}

    # 1) local deed folders
    roots = []
    if local_dir:
        roots.append(Path(local_dir))
    roots.append(Path("data"))
    for root in roots:
        if not root.exists():
            continue
        for gpath in root.rglob("grounding.json"):
            try:
                g = json.loads(gpath.read_text(encoding="utf-8"))
            except Exception:
                continue
            reg = str(g.get("reg_no") or gpath.parent.name).strip()
            if reg:
                out[reg] = g

    # 2) GCS sample_1000-style folders (GCS_PREFIX)
    try:
        import gcs_store
        if gcs_store.enabled():
            for reg in gcs_store.list_deed_ids():
                if reg in out:
                    continue
                raw = gcs_store.read_text(f"{reg}/grounding.json")
                if not raw:
                    continue
                try:
                    g = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                g.setdefault("reg_no", reg)
                out[reg] = g

            # 3) raw orissa_deeds jsonl (one line per deed)
            for prefix in gcs_store.raw_prefixes():
                graw = gcs_store.read_text_abs(
                    f"{prefix}/grounding/grounding_good_partial.jsonl")
                if not graw:
                    continue
                for line in graw.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        g = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    reg = str(g.get("reg_no") or "").strip()
                    if reg and reg not in out:
                        out[reg] = g
    except Exception as e:
        print(f"[compare] GCS load skipped/partial: {e}", flush=True)

    return out


def grounding_book_no(g: dict) -> int | None:
    """Infer book number from grounding header fields."""
    for key in ("book_no", "bookNo", "book"):
        if g.get(key) not in (None, ""):
            return _as_book_int(g.get(key))
    label = " ".join(str(g.get(k) or "") for k in ("book_label", "deed_type"))
    low = label.lower()
    for book, keywords in BOOK_CATEGORY_KEYWORDS.items():
        if any(kw in low for kw in keywords):
            return book
    return None


def grounding_scalars(g: dict) -> dict[str, str]:
    fields = g.get("fields") or []
    by_id = {}
    for f in fields:
        fid = f.get("id")
        if not fid:
            continue
        val = f.get("english_value")
        if val in (None, "") and f.get("odia_text"):
            val = f.get("odia_text")
        # first non-empty wins for scalars (item_index 0)
        if fid not in by_id or (not by_id[fid] and val):
            by_id[fid] = "" if val is None else str(val)
    # top-level header fallbacks
    if not by_id.get("deed_type") and g.get("deed_type"):
        by_id["deed_type"] = str(g["deed_type"])
    return by_id


def grounding_party_names(g: dict, party_id: str) -> list[str]:
    names = []
    for f in g.get("fields") or []:
        if f.get("id") == party_id and (f.get("attr") or "") == "name":
            n = norm_name(f.get("english_value") or f.get("odia_text") or "")
            if n:
                names.append(n)
    return names


def grounding_property_keys(g: dict) -> list[str]:
    """Comparable property fingerprints: village|khata|plot."""
    by_idx: dict[int, dict] = {}
    for f in g.get("fields") or []:
        if f.get("id") != "property_details":
            continue
        idx = int(f.get("item_index") or 0)
        by_idx.setdefault(idx, {})
        attr = (f.get("attr") or "").strip()
        by_idx[idx][attr] = f.get("english_value") or f.get("odia_text") or ""
    keys = []
    for idx in sorted(by_idx):
        p = by_idx[idx]
        key = "|".join(norm_text(p.get(a, "")) for a in ("village", "khata", "plot"))
        if key.strip("|"):
            keys.append(key)
    return keys


# ---------------------------------------------------------------------------
# API deed → normalised comparable dict
# ---------------------------------------------------------------------------

def _iter_dicts(value: Any) -> Iterable[dict]:
    if value is None:
        return
    if isinstance(value, dict):
        # sometimes wrapped as {"list":[...]} 
        for k in ("list", "List", "items", "Items", "data", "Data"):
            if isinstance(value.get(k), list):
                yield from _iter_dicts(value[k])
                return
        yield value
        return
    if isinstance(value, list):
        for item in value:
            yield from _iter_dicts(item)


def _name_from_party_row(row: dict) -> str:
    return norm_name(_pick(
        row, "name", "partyName", "personName", "fullName", "executantName",
        "claimantName", "firstName", "Name", default=""))


def api_party_names(deed: dict, side: str) -> list[str]:
    aliases = PARTY_API_KEYS[side]
    found = []
    # direct keys on the deed
    for alias in aliases:
        raw = _pick(deed, alias)
        if raw is None:
            continue
        for row in _iter_dicts(raw):
            n = _name_from_party_row(row)
            if n:
                found.append(n)
        if found:
            return found
    # nested under common parents
    for parent in ("parties", "partyDetails", "PartyDetails", "deedParties"):
        block = _pick(deed, parent)
        if not block:
            continue
        for row in _iter_dicts(block):
            role = norm_text(_pick(row, "role", "partyType", "type", "partyRole",
                                   default=""))
            if side == "seller" and any(x in role for x in (
                    "execut", "seller", "vendor", "first", "transferor")):
                n = _name_from_party_row(row)
                if n:
                    found.append(n)
            if side == "buyer" and any(x in role for x in (
                    "claim", "buyer", "vendee", "second", "transferee")):
                n = _name_from_party_row(row)
                if n:
                    found.append(n)
    return found


def api_property_keys(deed: dict) -> list[str]:
    keys = []
    for alias in PROPERTY_API_KEYS:
        raw = _pick(deed, alias)
        if raw is None:
            continue
        for row in _iter_dicts(raw):
            village = _pick(row, "village", "mouza", "revenueVillage",
                            "Village", default="")
            khata = _pick(row, "khata", "khataNo", "khata_no", "Khata",
                          default="")
            plot = _pick(row, "plot", "plotNo", "plot_no", "Plot", "sabikPlot",
                         default="")
            key = "|".join(norm_text(x) for x in (village, khata, plot))
            if key.strip("|"):
                keys.append(key)
        if keys:
            break
    return keys


def api_scalars(deed: dict) -> dict[str, str]:
    out = {}
    for canon, aliases, _gid in SCALAR_FIELDS:
        val = _pick(deed, *aliases)
        out[canon] = "" if val is None else str(val)
    return out


def flatten_api_for_display(deed: dict) -> dict:
    """Mapped view used in the report / UI."""
    return {
        "registration_no": str(_pick(
            deed, "registrationNo", "registration_no", "regNo", "reg_no",
            default=deed.get("registration_no", "")) or ""),
        "book_no": book_no_from_api_deed(deed),
        "scalars": api_scalars(deed),
        "seller_names": api_party_names(deed, "seller"),
        "buyer_names": api_party_names(deed, "buyer"),
        "property_plots": api_property_keys(deed),
    }


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def compare_one(reg_no: str, api_deed: dict, grounding: dict) -> dict:
    if is_empty_deed_payload(api_deed):
        return {
            "registration_no": reg_no,
            "book_no": grounding_book_no(grounding),
            "has_mismatch": False,
            "mismatch_count": 0,
            "mismatches": [],
            "api_empty": True,
            "error": "GetDeedInfoByRegNo returned empty payload "
                     "(data/information null) — not comparable",
            "api": flatten_api_for_display(api_deed) if isinstance(api_deed, dict) else {},
            "gcs_book_label": grounding.get("book_label"),
            "gcs_deed_type": grounding.get("deed_type"),
        }

    api_view = flatten_api_for_display(api_deed)
    g_scalars = grounding_scalars(grounding)
    mismatches = []

    for canon, _aliases, gid in SCALAR_FIELDS:
        api_val = api_view["scalars"].get(canon, "")
        g_val = g_scalars.get(gid, "")
        # skip if API didn't return the field at all
        if api_val in ("", None) and g_val in ("", None):
            continue
        if api_val in ("", None):
            # API silent on this field — not a mismatch, just missing source
            continue
        if not values_equal(canon, api_val, g_val):
            mismatches.append({
                "field": canon,
                "api_value": api_val,
                "gcs_value": g_val,
            })

    for side, party_id, field_name in (
        ("seller", "seller_details", "seller_names"),
        ("buyer", "buyer_details", "buyer_names"),
    ):
        api_names = api_view[field_name]
        g_names = grounding_party_names(grounding, party_id)
        if not api_names:
            continue
        if not values_equal(field_name, api_names, g_names):
            mismatches.append({
                "field": field_name,
                "api_value": api_names,
                "gcs_value": g_names,
            })

    api_props = api_view["property_plots"]
    g_props = grounding_property_keys(grounding)
    if api_props and not values_equal("property_plots", api_props, g_props):
        mismatches.append({
            "field": "property_plots",
            "api_value": api_props,
            "gcs_value": g_props,
        })

    api_book = api_view.get("book_no")
    g_book = grounding_book_no(grounding)
    if api_book is not None and g_book is not None and api_book != g_book:
        mismatches.append({
            "field": "book_no",
            "api_value": api_book,
            "gcs_value": g_book,
        })

    return {
        "registration_no": reg_no,
        "book_no": api_book if api_book is not None else g_book,
        "has_mismatch": bool(mismatches),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "api": api_view,
        "gcs_book_label": grounding.get("book_label"),
        "gcs_deed_type": grounding.get("deed_type"),
    }


def filter_regs_for_book(
    grounding_index: dict[str, dict],
    book: int,
    api_book_map: dict[str, int] | None = None,
) -> list[str]:
    """Pick registration numbers that belong to `book`."""
    regs = []
    for reg, g in grounding_index.items():
        b = None
        if api_book_map and reg in api_book_map:
            b = api_book_map[reg]
        if b is None:
            b = grounding_book_no(g)
        if b == book:
            regs.append(reg)
    return sorted(regs)


# ---------------------------------------------------------------------------
# DB update (optional — powers the frontend filter)
# ---------------------------------------------------------------------------

def ensure_mismatch_columns(con):
    con.execute(
        "ALTER TABLE documents ADD COLUMN IF NOT EXISTS has_api_mismatch "
        "BOOLEAN NOT NULL DEFAULT FALSE")
    con.execute(
        "ALTER TABLE documents ADD COLUMN IF NOT EXISTS api_mismatch_detail JSONB")
    con.execute(
        "ALTER TABLE documents ADD COLUMN IF NOT EXISTS api_book_no INT")
    con.commit()


def update_db(results: list[dict], clear_book: int | None = None):
    from db import connect, init_db
    init_db()
    con = connect()
    try:
        ensure_mismatch_columns(con)
        if clear_book is not None:
            # reset flags for this book before applying the new report
            con.execute(
                "UPDATE documents SET has_api_mismatch = FALSE, "
                "api_mismatch_detail = NULL "
                "WHERE api_book_no = %s OR book_no = %s "
                "OR (api_mismatch_detail->>'book_no')::int = %s",
                (clear_book, clear_book, clear_book))
        updated = 0
        missing = 0
        for row in results:
            reg = row["registration_no"]
            detail = {
                "book_no": row.get("book_no"),
                "mismatch_count": row.get("mismatch_count", 0),
                "mismatches": row.get("mismatches") or [],
                "api": row.get("api"),
            }
            cur = con.execute(
                "UPDATE documents SET has_api_mismatch = %s, "
                "api_mismatch_detail = %s::jsonb, api_book_no = %s "
                "WHERE deed_number = %s",
                (bool(row.get("has_mismatch")), json.dumps(detail),
                 row.get("book_no"), reg))
            if cur.rowcount:
                updated += 1
            else:
                missing += 1
        con.commit()
        print(f"[db] updated {updated} documents "
              f"({missing} reg_nos not yet ingested)", flush=True)
    finally:
        con.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--book", type=int, choices=(1, 3, 4), default=1,
                   help="Which registration book to compare (default: 1)")
    p.add_argument("--books", type=str, default="",
                   help="Comma-separated books, e.g. 1,3,4 (overrides --book)")
    p.add_argument("--base-url", default=os.environ.get("SARVAM_BASE_URL", ""),
                   help="Sarvam API BaseURL (or set SARVAM_BASE_URL)")
    p.add_argument("--from-date", default="",
                   help="Optional: also pull reg nos via GetDeedRegNoDetail")
    p.add_argument("--to-date", default="",
                   help="Optional: end date for GetDeedRegNoDetail")
    p.add_argument("--local-dir", default="",
                   help="Local folder of deed dirs (each with grounding.json)")
    p.add_argument("--reg-nos", default="",
                   help="Comma-separated registration numbers to check")
    p.add_argument("--limit", type=int, default=0,
                   help="Max deeds to compare (0 = all)")
    p.add_argument("--sleep", type=float, default=0.15,
                   help="Pause between API calls (seconds)")
    p.add_argument("--out", default="data/mismatches/book_mismatches.json",
                   help="Output JSON path (mismatches only by default)")
    p.add_argument("--save-all", action="store_true",
                   help="Also write comparisons with no mismatches")
    p.add_argument("--save-raw", default="",
                   help="Directory to dump raw GetDeedInfoByRegNo JSON per reg")
    p.add_argument("--update-db", action="store_true",
                   help="Stamp has_api_mismatch on matching documents rows")
    p.add_argument("--probe", default="",
                   help="Fetch + print one registration's API payload and exit")
    p.add_argument("--dry-run", action="store_true",
                   help="Resolve the reg list and exit without calling deed API")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    books = []
    if args.books.strip():
        books = [int(x) for x in args.books.split(",") if x.strip()]
    else:
        books = [args.book]

    base_url = args.base_url or os.environ.get("SARVAM_BASE_URL", "")
    if not base_url and not args.dry_run:
        print("ERROR: pass --base-url or set SARVAM_BASE_URL "
              "(the BaseURL from the Postman collection).", file=sys.stderr)
        return 2

    client = None
    need_api = not args.dry_run  # dry-run only needs local/GCS grounding
    if base_url and need_api:
        client = SarvamClient(base_url=base_url)
        print(f"[auth] logging in at {base_url} …", flush=True)
        client.authenticate()
        print("[auth] ok", flush=True)

    if args.probe:
        if not client:
            print("ERROR: --probe needs SARVAM_BASE_URL", file=sys.stderr)
            return 2
        parsed = client.get_deed_info(args.probe)
        print("--- raw HTTP JSON ---")
        print(json.dumps(client.last_raw, indent=2, default=str))
        print("\n--- parsed ---")
        print(json.dumps(parsed, indent=2, default=str))
        if is_empty_deed_payload(parsed):
            print("\nWARNING: API returned an empty envelope — no metadata fields "
                  "to compare. Try the curl below and ask the IGR contact whether "
                  "GetDeedInfoByRegNo is enabled for this login / reg no.",
                  flush=True)
            print(
                "\ncurl -k -X POST "
                f"'{base_url}/api/Deed/GetDeedInfoByRegNo' "
                "-H 'Content-Type: application/json' "
                "-H \"Authorization: Bearer $TOKEN\" "
                f"-d '{{\"registrationNo\": \"{args.probe}\"}}'",
                flush=True)
            return 3
        print("\n--- mapped ---")
        print(json.dumps(flatten_api_for_display(parsed), indent=2, default=str))
        return 0

    print("[gcs] loading grounding index…", flush=True)
    grounding_index = load_grounding_index(args.local_dir or None)
    print(f"[gcs] {len(grounding_index)} deeds with grounding metadata",
          flush=True)
    if len(grounding_index) <= 1:
        print("[gcs] NOTE: only the local sample deed is loaded. To compare the "
              "real Book 1 corpus, set GCS_BUCKET (+ GCS_CREDENTIALS_JSON) and "
              "GCS_PREFIX / GCS_RAW_PREFIX like the deployed app.", flush=True)

    api_book_map: dict[str, int] = {}
    if args.from_date and args.to_date and client:
        print(f"[api] GetDeedRegNoDetail {args.from_date} → {args.to_date}",
              flush=True)
        rows = client.get_reg_nos_by_date(args.from_date, args.to_date)
        for row in rows:
            if row.get("book_no") is not None:
                api_book_map[row["registration_no"]] = row["book_no"]
        print(f"[api] {len(rows)} reg nos ({len(api_book_map)} with book_no)",
              flush=True)

    if args.reg_nos.strip():
        target_regs = [r.strip() for r in args.reg_nos.split(",") if r.strip()]
    else:
        target_regs = []
        for b in books:
            regs = filter_regs_for_book(grounding_index, b, api_book_map)
            print(f"[book {b}] {len(regs)} candidate registration numbers",
                  flush=True)
            target_regs.extend(regs)
        # de-dupe preserving order
        seen = set()
        target_regs = [r for r in target_regs
                       if not (r in seen or seen.add(r))]

    if args.limit and args.limit > 0:
        target_regs = target_regs[: args.limit]

    print(f"[compare] {len(target_regs)} deeds to check", flush=True)
    if args.dry_run:
        for r in target_regs[:50]:
            g = grounding_index.get(r, {})
            print(f"  {r}  book={grounding_book_no(g)}  "
                  f"{g.get('book_label')}/{g.get('deed_type')}")
        if len(target_regs) > 50:
            print(f"  … +{len(target_regs) - 50} more")
        return 0

    if not client:
        print("ERROR: SARVAM_BASE_URL required", file=sys.stderr)
        return 2

    raw_dir = Path(args.save_raw) if args.save_raw else None
    if raw_dir:
        raw_dir.mkdir(parents=True, exist_ok=True)

    results = []
    mismatches_only = []
    errors = []

    for i, reg in enumerate(target_regs, 1):
        g = grounding_index.get(reg)
        if not g:
            errors.append({"registration_no": reg, "error": "no grounding in GCS/local"})
            continue
        try:
            api_deed = client.get_deed_info(reg)
            if raw_dir:
                (raw_dir / f"{reg}.json").write_text(
                    json.dumps(client.last_raw if client.last_raw is not None
                               else api_deed, indent=2, default=str),
                    encoding="utf-8")
            if is_empty_deed_payload(api_deed):
                errors.append({
                    "registration_no": reg,
                    "error": "GetDeedInfoByRegNo returned empty payload",
                    "raw": client.last_raw,
                })
                print(f"  [{i}/{len(target_regs)}] EMPTY API {reg}", flush=True)
                continue
            row = compare_one(reg, api_deed, g)
            results.append(row)
            if row["has_mismatch"]:
                mismatches_only.append(row)
                print(f"  [{i}/{len(target_regs)}] MISMATCH {reg} "
                      f"({row['mismatch_count']} fields)", flush=True)
            elif i % 25 == 0 or i == len(target_regs):
                print(f"  [{i}/{len(target_regs)}] ok {reg}", flush=True)
        except Exception as e:
            errors.append({"registration_no": reg, "error": str(e)})
            print(f"  [{i}/{len(target_regs)}] ERROR {reg}: {e}", flush=True)
        if args.sleep:
            time.sleep(args.sleep)

    out_path = Path(args.out)
    # If the default path is used with a single book, name the file by book.
    if args.out == "data/mismatches/book_mismatches.json" and len(books) == 1:
        out_path = Path(f"data/mismatches/book{books[0]}_mismatches.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "books": books,
        "compared": len(results),
        "mismatch_count": len(mismatches_only),
        "error_count": len(errors),
        "mismatches": mismatches_only,
        "errors": errors,
    }
    if args.save_all:
        report["all"] = results
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\n[done] compared={len(results)} mismatches={len(mismatches_only)} "
          f"errors={len(errors)}", flush=True)
    print(f"[done] wrote {out_path}", flush=True)

    if args.update_db:
        # Persist every compared deed (clear + mismatch flags) so the UI
        # filter stays accurate after re-runs.
        update_db(results, clear_book=books[0] if len(books) == 1 else None)

    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
