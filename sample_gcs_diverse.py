#!/usr/bin/env python3
"""
sample_gcs_diverse.py — pull a diverse set of deeds that already exist in
our GCS OCR/grounding corpus (Books 1 / 3 / 4).

Unlike sample_igr_diverse.py (which samples live IGR API registration
numbers that may not overlap GCS at all), this sampler starts from the
deeds we hold — so scans / OCR / grounding are guaranteed to exist.

Sources (first available wins unless --source is set)
----------------------------------------------------
1. GCS  — GCS_BUCKET (+ GCS_CREDENTIALS_JSON) with GCS_PREFIX / GCS_RAW_PREFIX
2. Render/API — --render-url of a deployed deed-validator that already
   ingested the GCS corpus (uses admin search + document endpoints)
3. Local — data/ folders with grounding.json (or --local-dir)

Example
-------
    # Against the deployed app that already ingested GCS:
    python sample_gcs_diverse.py --source render \\
        --render-url https://deed-validator.onrender.com \\
        --render-user admin --render-password '...' \\
        --target 2500 --out data/mismatches/gcs_diverse_sample.json

    # Directly from the bucket (same env vars as the app):
    export GCS_BUCKET=classification-vision
    export GCS_CREDENTIALS_JSON='...'
    export GCS_RAW_PREFIX=ocr_outputs/orissa_deeds
    python sample_gcs_diverse.py --source gcs --target 2500
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from compare_metadata import (
    BOOK_CATEGORY_KEYWORDS,
    grounding_book_no,
    grounding_scalars,
    load_grounding_index,
)
from sarvam_client import normalize_reg_no

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None


FIELD_LABEL_TO_KEY = {
    "deed type": "deed_type",
    "district": "district",
    "registration office": "office",
    "registration date": "registration_date",
    "presentation date": "presentation_date",
    "consideration amount": "consideration_amount",
    "old registration no": "old_reg_no",
    "old reg no": "old_reg_no",
}


def infer_book(book_label: str | None, deed_type: str | None) -> int | None:
    """Map GCS book_label / deed_type → Book 1 / 3 / 4."""
    # Prefer the explicit category labels used in orissa_deeds grounding.
    label = (book_label or "").strip().upper()
    if label == "SALE":
        return 1
    if label == "WILL":
        return 3
    if label in ("MISC", "MISCELLANEOUS", "POA"):
        return 4

    blob = " ".join(str(x or "") for x in (book_label, deed_type)).lower()
    if not blob.strip():
        return None
    # More specific books first so "sale agreement" stays Book 1, etc.
    for book in (1, 3, 4):
        if any(kw in blob for kw in BOOK_CATEGORY_KEYWORDS[book]):
            return book
    return None


def row_from_grounding(reg: str, g: dict) -> dict | None:
    book = grounding_book_no(g) or infer_book(g.get("book_label"), g.get("deed_type"))
    if book not in (1, 3, 4):
        return None
    scalars = grounding_scalars(g)
    district = (scalars.get("district") or "").strip() or "UNKNOWN"
    office = (scalars.get("office") or "").strip() or "UNKNOWN"
    return {
        "registration_no": reg,
        "book_no": book,
        "book_label": g.get("book_label"),
        "district": district,
        "office": office,
        "deed_type": scalars.get("deed_type") or g.get("deed_type"),
        "registration_date": scalars.get("registration_date"),
        "presentation_date": scalars.get("presentation_date"),
        "consideration_amount": scalars.get("consideration_amount"),
        "old_reg_no": scalars.get("old_reg_no"),
        "year": None,
        "source": "gcs_grounding",
        "has_scan": True,  # GCS corpus — page images / PDF available
    }


def load_from_gcs_or_local(local_dir: str | None) -> list[dict]:
    print("[source] loading grounding index (GCS if configured, else local)…",
          flush=True)
    index = load_grounding_index(local_dir)
    print(f"[source] {len(index)} deeds with grounding", flush=True)
    rows = []
    skipped = 0
    for reg, g in index.items():
        row = row_from_grounding(reg, g)
        if row is None:
            skipped += 1
            continue
        rows.append(row)
    print(f"[source] {len(rows)} in Books 1/3/4 (skipped {skipped})", flush=True)
    return rows


def render_login(base_url: str, username: str, password: str) -> str:
    if requests is None:
        raise RuntimeError("requests package required for --source render")
    r = requests.post(
        f"{base_url.rstrip('/')}/api/login",
        json={"username": username, "password": password},
        timeout=60,
    )
    r.raise_for_status()
    token = r.json().get("token")
    if not token:
        raise RuntimeError(f"login failed: {r.text[:200]}")
    return token


def render_list_documents(base_url: str, token: str) -> list[dict]:
    """Page through /api/search as admin to list the ingested GCS corpus."""
    if requests is None:
        raise RuntimeError("requests package required for --source render")
    base = base_url.rstrip("/")
    headers = {"Authorization": f"Bearer {token}"}
    # FastAPI Depends(current_user) also accepts ?token=
    per_page = 100
    page = 1
    out: list[dict] = []
    total = None
    while True:
        r = requests.get(
            f"{base}/api/search",
            params={"token": token, "page": page, "per_page": per_page,
                    "sort_by": "deed_number", "sort_order": "asc"},
            headers=headers,
            timeout=120,
        )
        r.raise_for_status()
        payload = r.json()
        if total is None:
            total = payload.get("total", 0)
            print(f"[render] corpus size={total}", flush=True)
        batch = payload.get("results") or []
        if not batch:
            break
        out.extend(batch)
        if page % 20 == 0 or len(out) >= total:
            print(f"[render] listed {len(out)}/{total}", flush=True)
        if len(out) >= total:
            break
        page += 1
    return out


def _fields_to_scalars(fields: list[dict]) -> dict[str, str]:
    out: dict[str, str] = {}
    for f in fields or []:
        label = (f.get("label") or "").strip().lower()
        key = FIELD_LABEL_TO_KEY.get(label)
        if not key or key in out:
            continue
        val = f.get("current_value")
        if val in (None, ""):
            val = f.get("ocr_value")
        if val not in (None, ""):
            out[key] = str(val)
    return out


def render_fetch_detail(base_url: str, token: str, doc_id: int) -> dict | None:
    if requests is None:
        return None
    r = requests.get(
        f"{base_url.rstrip('/')}/api/documents/{doc_id}",
        params={"token": token},
        timeout=60,
    )
    if r.status_code != 200:
        return None
    payload = r.json()
    doc = payload.get("document") or {}
    scalars = _fields_to_scalars(payload.get("fields") or [])
    src_meta = doc.get("src_meta") or {}
    if isinstance(src_meta, str):
        try:
            src_meta = json.loads(src_meta)
        except json.JSONDecodeError:
            src_meta = {}
    book_label = src_meta.get("book_label") if isinstance(src_meta, dict) else None
    deed_type = doc.get("deed_type") or scalars.get("deed_type")
    book = infer_book(book_label, deed_type)
    if book not in (1, 3, 4):
        return None
    reg = normalize_reg_no(doc.get("deed_number") or doc.get("reg_no") or "")
    if not reg:
        return None
    return {
        "registration_no": reg,
        "document_id": doc_id,
        "book_no": book,
        "book_label": book_label,
        "district": (scalars.get("district") or doc.get("district") or "UNKNOWN").strip()
                    or "UNKNOWN",
        "office": (scalars.get("office") or doc.get("sr_office") or "UNKNOWN").strip()
                  or "UNKNOWN",
        "deed_type": deed_type,
        "registration_date": scalars.get("registration_date"),
        "presentation_date": scalars.get("presentation_date"),
        "consideration_amount": scalars.get("consideration_amount"),
        "old_reg_no": scalars.get("old_reg_no"),
        "year": doc.get("year"),
        "status": doc.get("status"),
        "has_pdf": bool(doc.get("pdf_file")),
        "source": "render_gcs_ingest",
        "has_scan": True,
    }


def load_from_render(args, target: int, book_targets: dict[int, int],
                     per_district_cap: int, per_office_cap: int,
                     rng: random.Random) -> list[dict]:
    """List the ingested GCS corpus, then fetch details only until diverse
    quotas are filled (avoids pulling all ~11k document payloads)."""
    print(f"[render] logging in at {args.render_url}", flush=True)
    token = args.render_token or render_login(
        args.render_url, args.render_user, args.render_password)
    listed = render_list_documents(args.render_url, token)

    by_book: dict[int, list[dict]] = defaultdict(list)
    skipped = 0
    for item in listed:
        book = infer_book(None, item.get("deed_type"))
        if book in (1, 3, 4):
            by_book[book].append(item)
        else:
            skipped += 1
    for b in by_book:
        rng.shuffle(by_book[b])
    print(f"[render] book pools={{1: {len(by_book[1])}, 3: {len(by_book[3])}, "
          f"4: {len(by_book[4])}}} skipped_other={skipped}", flush=True)

    selected: list[dict] = []
    counts = Counter()
    by_district = Counter()
    by_office = Counter()
    skipped_cap = 0
    fetched = 0
    workers = max(1, args.workers)
    batch_size = max(workers * 2, 32)

    idxs = {1: 0, 3: 0, 4: 0}

    def take_batch(book: int, n: int) -> list[dict]:
        start = idxs[book]
        end = min(start + n, len(by_book[book]))
        idxs[book] = end
        return by_book[book][start:end]

    while len(selected) < target:
        need_books = [b for b in (1, 3, 4)
                      if counts[b] < book_targets.get(b, 0) and idxs[b] < len(by_book[b])]
        if not need_books:
            break
        need_books.sort(key=lambda b: (book_targets.get(b, 0) - counts[b]), reverse=True)
        book = need_books[0]
        want = min(batch_size, book_targets.get(book, 0) - counts[book] + batch_size // 2)
        batch = take_batch(book, max(want, workers))
        if not batch:
            continue

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(render_fetch_detail, args.render_url, token, item["id"])
                    for item in batch]
            for fut in as_completed(futs):
                fetched += 1
                try:
                    row = fut.result()
                except Exception:
                    continue
                if not row:
                    continue
                b = row["book_no"]
                if counts[b] >= book_targets.get(b, 0):
                    skipped_cap += 1
                    continue
                district = row["district"]
                office = row["office"]
                if by_district[district] >= per_district_cap:
                    skipped_cap += 1
                    continue
                if by_office[office] >= per_office_cap:
                    skipped_cap += 1
                    continue
                selected.append(row)
                counts[b] += 1
                by_district[district] += 1
                by_office[office] += 1

        print(f"  [render selected {len(selected)}/{target}] "
              f"books={dict(counts)} districts={len(by_district)} "
              f"offices={len(by_office)} fetched={fetched} "
              f"cap_skips={skipped_cap}", flush=True)

    return selected


def select_diverse(
    pool: list[dict],
    target: int,
    book_targets: dict[int, int],
    per_district_cap: int,
    per_office_cap: int,
    rng: random.Random,
) -> list[dict]:
    by_book = defaultdict(list)
    for row in pool:
        by_book[row["book_no"]].append(row)
    for b in by_book:
        rng.shuffle(by_book[b])

    # Round-robin across books so geographic caps don't starve later books.
    selected: list[dict] = []
    counts = Counter()
    by_district = Counter()
    by_office = Counter()
    skipped_cap = 0
    idxs = {1: 0, 3: 0, 4: 0}

    def next_for(book: int) -> dict | None:
        while idxs[book] < len(by_book[book]):
            row = by_book[book][idxs[book]]
            idxs[book] += 1
            return row
        return None

    # Keep going until quotas filled or pools exhausted.
    progressed = True
    while len(selected) < target and progressed:
        progressed = False
        for book in (1, 3, 4):
            if counts[book] >= book_targets.get(book, 0):
                continue
            if len(selected) >= target:
                break
            # Try several candidates for this book before moving on.
            attempts = 0
            while attempts < 40 and counts[book] < book_targets.get(book, 0) \
                    and len(selected) < target:
                attempts += 1
                row = next_for(book)
                if row is None:
                    break
                district = row["district"]
                office = row["office"]
                if by_district[district] >= per_district_cap:
                    skipped_cap += 1
                    continue
                if by_office[office] >= per_office_cap:
                    skipped_cap += 1
                    continue
                selected.append(row)
                counts[book] += 1
                by_district[district] += 1
                by_office[office] += 1
                progressed = True
                if len(selected) % 100 == 0:
                    print(f"  [selected {len(selected)}/{target}] "
                          f"books={dict(counts)} districts={len(by_district)} "
                          f"offices={len(by_office)} cap_skips={skipped_cap}",
                          flush=True)

    print(f"[select] kept {len(selected)} cap_skips={skipped_cap} "
          f"books={dict(counts)} districts={len(by_district)} "
          f"offices={len(by_office)}", flush=True)
    return selected


def write_outputs(rows: list[dict], out_json: Path):
    out_json.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "count": len(rows),
        "source": "gcs_corpus",
        "by_book": dict(Counter(r["book_no"] for r in rows)),
        "by_book_label": dict(Counter((r.get("book_label") or "?") for r in rows)),
        "by_district": dict(Counter(r["district"] for r in rows).most_common()),
        "by_office_top30": dict(Counter(r["office"] for r in rows).most_common(30)),
        "district_count": len({r["district"] for r in rows}),
        "office_count": len({r["office"] for r in rows}),
        "with_scan": sum(1 for r in rows if r.get("has_scan") or r.get("has_pdf")),
        "year_span": {
            "min": min((r["year"] for r in rows if r.get("year")), default=None),
            "max": max((r["year"] for r in rows if r.get("year")), default=None),
        },
        "date_span": {
            "min_registration_date": min(
                (r.get("registration_date") or "9999" for r in rows), default=None),
            "max_registration_date": max(
                (r.get("registration_date") or "" for r in rows), default=None),
        },
    }
    # Drop bulky raw blobs if any
    slim = [{k: v for k, v in r.items() if k != "raw"} for r in rows]
    payload = {"summary": summary, "deeds": slim}
    out_json.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    csv_path = out_json.with_suffix(".csv")
    fields = [
        "registration_no", "book_no", "book_label", "district", "office",
        "deed_type", "registration_date", "presentation_date",
        "consideration_amount", "old_reg_no", "year", "has_scan", "has_pdf",
        "source",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in slim:
            w.writerow(r)

    regs_path = out_json.with_name(out_json.stem + "_reg_nos.txt")
    regs_path.write_text(
        "\n".join(r["registration_no"] for r in slim) + "\n", encoding="utf-8")

    print(f"\nwrote {out_json}", flush=True)
    print(f"wrote {csv_path}", flush=True)
    print(f"wrote {regs_path}", flush=True)
    print(json.dumps(summary, indent=2, default=str), flush=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", choices=("auto", "gcs", "render", "local"),
                   default="auto")
    p.add_argument("--local-dir", default="",
                   help="Local folder of deed dirs (each with grounding.json)")
    p.add_argument("--render-url", default=os.environ.get(
        "RENDER_URL", "https://deed-validator.onrender.com"))
    p.add_argument("--render-user", default=os.environ.get("RENDER_USER", "admin"))
    p.add_argument("--render-password",
                   default=os.environ.get("RENDER_PASSWORD", "sarvam123"))
    p.add_argument("--render-token", default=os.environ.get("RENDER_TOKEN", ""),
                   help="Skip login if you already have an admin session token")
    p.add_argument("--workers", type=int, default=16,
                   help="Parallel document fetches for --source render")
    p.add_argument("--target", type=int, default=2500)
    p.add_argument("--book1", type=int, default=1200)
    p.add_argument("--book3", type=int, default=650)
    p.add_argument("--book4", type=int, default=650)
    p.add_argument("--per-district-cap", type=int, default=200)
    p.add_argument("--per-office-cap", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="data/mismatches/gcs_diverse_sample.json")
    return p.parse_args(argv)


def resolve_source(args) -> str:
    if args.source != "auto":
        return args.source
    if os.environ.get("GCS_BUCKET"):
        return "gcs"
    # Prefer render when the deployed corpus is larger than the local sample.
    return "render"


def main(argv=None):
    args = parse_args(argv)
    source = resolve_source(args)
    print(f"[sample_gcs_diverse] source={source} target={args.target}", flush=True)

    book_targets = {1: args.book1, 3: args.book3, 4: args.book4}
    total_quota = sum(book_targets.values())
    if total_quota != args.target:
        scale = args.target / max(total_quota, 1)
        book_targets = {b: max(1, int(round(n * scale))) for b, n in book_targets.items()}
        drift = args.target - sum(book_targets.values())
        book_targets[1] = max(1, book_targets[1] + drift)

    t0 = time.time()
    rng = random.Random(args.seed)

    if source == "render":
        # Selection happens while fetching so we don't pull every document.
        print(f"[select] quotas={book_targets} district_cap={args.per_district_cap} "
              f"office_cap={args.per_office_cap}", flush=True)
        rows = load_from_render(
            args, args.target, book_targets,
            args.per_district_cap, args.per_office_cap, rng)
        pool_summary = {
            "mode": "render_streaming_select",
            "selected": len(rows),
            "by_book": dict(Counter(r["book_no"] for r in rows)),
            "elapsed_sec": round(time.time() - t0, 1),
        }
    elif source in ("gcs", "local"):
        pool = load_from_gcs_or_local(args.local_dir or None)
        pool_summary = {
            "count": len(pool),
            "by_book": dict(Counter(r["book_no"] for r in pool)),
            "by_book_label": dict(Counter((r.get("book_label") or "?") for r in pool)),
            "district_count": len({r["district"] for r in pool}),
            "office_count": len({r["office"] for r in pool}),
            "elapsed_sec": round(time.time() - t0, 1),
        }
        if not pool:
            print("ERROR: empty pool — check GCS credentials / local dir",
                  file=sys.stderr)
            return 1
        print(f"[pool] {json.dumps(pool_summary)}", flush=True)
        print(f"[select] quotas={book_targets} district_cap={args.per_district_cap} "
              f"office_cap={args.per_office_cap}", flush=True)
        rows = select_diverse(
            pool, args.target, book_targets,
            args.per_district_cap, args.per_office_cap, rng)
    else:
        print(f"ERROR: unknown source {source}", file=sys.stderr)
        return 2

    pool_path = Path(args.out).with_name(Path(args.out).stem + "_pool_summary.json")
    pool_path.parent.mkdir(parents=True, exist_ok=True)
    pool_path.write_text(json.dumps(pool_summary, indent=2), encoding="utf-8")
    print(f"[pool] {json.dumps(pool_summary)}", flush=True)

    if not rows:
        print("ERROR: no rows selected", file=sys.stderr)
        return 1

    write_outputs(rows, Path(args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
