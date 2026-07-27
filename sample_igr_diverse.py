#!/usr/bin/env python3
"""
sample_igr_diverse.py — pull a diverse set of live IGR deeds for Books 1 / 3 / 4.

Strategy
--------
1. Call GetDeedRegNoDetail on many calendar days spread across years.
2. Keep registration nos tagged with bookNo in {1, 3, 4}.
3. Fetch GetDeedInfoByRegNo and retain only rows with a real `data` payload.
4. Enforce diversity caps by district / office so one city doesn't dominate.
5. Write JSON + CSV under data/mismatches/.

Example
-------
    export SARVAM_BASE_URL=https://erp.igrodisha.gov.in/igrone
    export SARVAM_VERIFY_SSL=0
    python sample_igr_diverse.py --target 2500 --out data/mismatches/igr_diverse_sample.json
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
from datetime import date, timedelta
from pathlib import Path

from sarvam_client import SarvamClient, is_empty_deed_payload, normalize_reg_no


def daterange_sample(start: date, end: date, step_days: int) -> list[date]:
    out = []
    d = start
    while d <= end:
        out.append(d)
        d += timedelta(days=step_days)
    return out


def build_date_grid(args) -> list[date]:
    """Many days across years — denser in years that historically return rows."""
    dates: set[date] = set()
    # Every N days across the main window
    dates.update(daterange_sample(
        date(args.year_from, 1, 1),
        date(args.year_to, 12, 31),
        args.date_step_days,
    ))
    # Also mid-month anchors (IGR often has activity mid-month in our probes)
    for y in range(args.year_from, args.year_to + 1):
        for m in range(1, 13):
            dates.add(date(y, m, 15))
            dates.add(date(y, m, 1))
            if m in (1, 3, 5, 7, 8, 10, 12):
                dates.add(date(y, m, 28))
            elif m != 2:
                dates.add(date(y, m, 27))
            else:
                dates.add(date(y, m, 20))
    return sorted(dates)


def collect_reg_index(client: SarvamClient, dates: list[date], sleep: float
                      ) -> dict[str, dict]:
    """reg_no -> {book_no, from_date} from GetDeedRegNoDetail."""
    index: dict[str, dict] = {}
    empty_days = 0
    for i, d in enumerate(dates, 1):
        fd = d.strftime("%d-%b-%Y")
        try:
            rows = client.get_reg_nos_by_date(fd, fd)
        except Exception as e:
            print(f"  [date {i}/{len(dates)}] {fd} ERR {e}", flush=True)
            continue
        if not rows:
            empty_days += 1
        for row in rows:
            book = row.get("book_no")
            reg = normalize_reg_no(row.get("registration_no"))
            if not reg or book not in (1, 3, 4):
                continue
            if reg not in index:
                index[reg] = {"book_no": book, "listed_on": fd}
        if i % 50 == 0 or i == len(dates):
            by_book = Counter(v["book_no"] for v in index.values())
            print(f"  [date {i}/{len(dates)}] {fd}: pool={len(index)} "
                  f"by_book={dict(by_book)} empty_days={empty_days}", flush=True)
        if sleep:
            time.sleep(sleep)
    return index


def fetch_diverse(
    client: SarvamClient,
    index: dict[str, dict],
    target: int,
    per_district_cap: int,
    per_office_cap: int,
    book_targets: dict[int, int],
    sleep: float,
    rng: random.Random,
    checkpoint_path: Path | None = None,
) -> list[dict]:
    """Fetch deed info until target / book quotas / diversity caps are hit."""
    # Prefer under-filled books first so quotas fill instead of one book eating caps
    by_listed_book = defaultdict(list)
    for reg, meta in index.items():
        by_listed_book[meta["book_no"]].append(reg)
    regs: list[str] = []
    for b in (1, 3, 4):
        chunk = by_listed_book.get(b, [])
        rng.shuffle(chunk)
        regs.extend(chunk)

    selected: list[dict] = []
    by_book = Counter()
    by_district = Counter()
    by_office = Counter()
    skipped_empty = 0
    skipped_cap = 0
    fetched = 0

    def room_left() -> bool:
        for b, need in book_targets.items():
            if by_book[b] < need:
                return True
        return len(selected) < target

    for i, reg in enumerate(regs, 1):
        if len(selected) >= target or not room_left():
            break

        meta = index[reg]
        book = meta["book_no"]
        if by_book[book] >= book_targets.get(book, target):
            continue

        try:
            info = client.get_deed_info(reg)
            fetched += 1
        except Exception as e:
            skipped_empty += 1
            if fetched % 100 == 0:
                print(f"  [fetch] errors; last={e}", flush=True)
            continue

        if is_empty_deed_payload(info):
            skipped_empty += 1
            continue

        district = str(info.get("district") or info.get("districtName") or "UNKNOWN").strip()
        office = str(info.get("office") or info.get("sroName") or "UNKNOWN").strip()
        api_book = info.get("deedBookNo") or info.get("bookNo") or book
        try:
            api_book = int(str(api_book).strip() or book)
        except ValueError:
            api_book = book
        use_book = api_book if api_book in (1, 3, 4) else book

        if by_book[use_book] >= book_targets.get(use_book, target):
            continue
        if by_district[district] >= per_district_cap:
            skipped_cap += 1
            continue
        if by_office[office] >= per_office_cap:
            skipped_cap += 1
            continue

        row = {
            "registration_no": reg,
            "book_no": use_book,
            "listed_on": meta.get("listed_on"),
            "district": district,
            "office": office,
            "deed_type": info.get("deedType") or info.get("deed_type"),
            "registration_date": info.get("registrationDate") or info.get("registration_date"),
            "presentation_date": info.get("presentationDate") or info.get("presentation_date"),
            "consideration_amount": info.get("considerationAmount") or info.get("consideration_amount"),
            "old_reg_no": info.get("oldRegNO") or info.get("oldRegNo"),
            "deed_volume_no": info.get("deedVolumeNo"),
            "seller_details": info.get("sellerDetails"),
            "buyer_details": info.get("buyerDetails"),
            "property_details": info.get("propertyDetails"),
            "raw": {k: v for k, v in info.items() if not str(k).startswith("_")},
        }
        selected.append(row)
        by_book[use_book] += 1
        by_district[district] += 1
        by_office[office] += 1

        if len(selected) % 50 == 0 or len(selected) == target:
            print(f"  [selected {len(selected)}/{target}] books={dict(by_book)} "
                  f"districts={len(by_district)} offices={len(by_office)} "
                  f"empty_skips={skipped_empty} cap_skips={skipped_cap} "
                  f"fetched={fetched}", flush=True)
            if checkpoint_path:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                checkpoint_path.write_text(
                    json.dumps({"count": len(selected), "deeds": selected}, default=str),
                    encoding="utf-8")
        if sleep:
            time.sleep(sleep)

    return selected


def write_outputs(rows: list[dict], out_json: Path):
    out_json.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "count": len(rows),
        "by_book": dict(Counter(r["book_no"] for r in rows)),
        "by_district": dict(Counter(r["district"] for r in rows).most_common()),
        "by_office_top30": dict(Counter(r["office"] for r in rows).most_common(30)),
        "district_count": len({r["district"] for r in rows}),
        "office_count": len({r["office"] for r in rows}),
        "date_span": {
            "min_listed_on": min((r.get("listed_on") or "9999" for r in rows), default=None),
            "max_listed_on": max((r.get("listed_on") or "" for r in rows), default=None),
            "min_registration_date": min(
                (r.get("registration_date") or "9999" for r in rows), default=None),
            "max_registration_date": max(
                (r.get("registration_date") or "" for r in rows), default=None),
        },
    }
    payload = {"summary": summary, "deeds": rows}
    out_json.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    csv_path = out_json.with_suffix(".csv")
    fields = [
        "registration_no", "book_no", "district", "office", "deed_type",
        "registration_date", "presentation_date", "consideration_amount",
        "old_reg_no", "deed_volume_no", "listed_on",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # compact reg list for follow-up compare runs
    regs_path = out_json.with_name(out_json.stem + "_reg_nos.txt")
    regs_path.write_text(
        "\n".join(r["registration_no"] for r in rows) + "\n", encoding="utf-8")

    print(f"\nwrote {out_json}", flush=True)
    print(f"wrote {csv_path}", flush=True)
    print(f"wrote {regs_path}", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target", type=int, default=2500,
                   help="How many deeds with full metadata to keep (default 2500)")
    p.add_argument("--year-from", type=int, default=2000)
    p.add_argument("--year-to", type=int, default=2006)
    p.add_argument("--date-step-days", type=int, default=3,
                   help="Spacing between calendar days queried (default 3)")
    p.add_argument("--per-district-cap", type=int, default=200,
                   help="Max deeds kept per district (forces geo diversity)")
    p.add_argument("--per-office-cap", type=int, default=100,
                   help="Max deeds kept per registration office")
    p.add_argument("--book1", type=int, default=1200, help="Target count for Book 1")
    p.add_argument("--book3", type=int, default=650, help="Target count for Book 3")
    p.add_argument("--book4", type=int, default=650, help="Target count for Book 4")
    p.add_argument("--sleep-date", type=float, default=0.03)
    p.add_argument("--sleep-info", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="data/mismatches/igr_diverse_sample.json")
    p.add_argument("--base-url", default=os.environ.get("SARVAM_BASE_URL", ""))
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    base = args.base_url or os.environ.get("SARVAM_BASE_URL", "")
    if not base:
        print("ERROR: set SARVAM_BASE_URL", file=sys.stderr)
        return 2

    # Normalize book targets to sum ≈ target
    book_targets = {1: args.book1, 3: args.book3, 4: args.book4}
    total_quota = sum(book_targets.values())
    if total_quota != args.target:
        # scale quotas proportionally to --target
        scale = args.target / max(total_quota, 1)
        book_targets = {b: max(1, int(round(n * scale))) for b, n in book_targets.items()}
        # fix rounding drift
        drift = args.target - sum(book_targets.values())
        book_targets[1] = max(1, book_targets[1] + drift)

    client = SarvamClient(base_url=base)
    print(f"[auth] {base}", flush=True)
    client.authenticate()
    print("[auth] ok", flush=True)

    dates = build_date_grid(args)
    print(f"[dates] querying {len(dates)} calendar days "
          f"({args.year_from}–{args.year_to}, step={args.date_step_days})", flush=True)

    index = collect_reg_index(client, dates, args.sleep_date)
    print(f"[pool] {len(index)} unique Book 1/3/4 registration numbers", flush=True)
    pool_path = Path(args.out).with_name(Path(args.out).stem + "_pool.json")
    pool_path.parent.mkdir(parents=True, exist_ok=True)
    pool_path.write_text(json.dumps(index), encoding="utf-8")
    print(f"[pool] wrote {pool_path}", flush=True)
    if len(index) < args.target:
        print(f"[warn] pool smaller than target {args.target}; will take all usable",
              flush=True)

    print(f"[fetch] target={args.target} quotas={book_targets} "
          f"district_cap={args.per_district_cap} office_cap={args.per_office_cap}",
          flush=True)
    ckpt = Path(args.out).with_name(Path(args.out).stem + "_checkpoint.json")
    rows = fetch_diverse(
        client, index, args.target,
        args.per_district_cap, args.per_office_cap,
        book_targets, args.sleep_info, random.Random(args.seed),
        checkpoint_path=ckpt,
    )
    write_outputs(rows, Path(args.out))
    return 0 if rows else 1


if __name__ == "__main__":
    sys.exit(main())
