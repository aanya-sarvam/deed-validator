"""Ingest deed metadata (Excel) + scanned PDFs into the validation database.

Usage:  python ingest.py "data/data-85-94/DEED 85-94.xlsx" data/data-85-94
Re-runnable: skips deeds already present.
"""
import re
import sys
import shutil
import sqlite3
from pathlib import Path

import pandas as pd

from db import connect, init_db

PARTY_SPLIT = re.compile(r",\s*(?=\d+-)")
PARTY_PARSE = re.compile(
    r"^(?P<num>\d+)-\s*(?P<name>.*?)\s*\(\s*RELATION\s*:\s*\)\s*(?P<relation>.*?)\s*"
    r"\(\s*RELATION NAME\s*:\s*\)\s*(?P<relname>.*?)\s*\(\s*ADDRESS\s*:\s*\)\s*(?P<address>.*)$",
    re.S,
)


def parse_parties(raw):
    """'1-NAME (RELATION:) FATHER (RELATION NAME:) X (ADDRESS:) Y ,2-...' -> list of dicts."""
    if not isinstance(raw, str) or not raw.strip():
        return []
    parties = []
    for chunk in PARTY_SPLIT.split(raw.strip()):
        m = PARTY_PARSE.match(chunk.strip())
        if m:
            parties.append({k: " ".join(m.group(k).split()) for k in
                            ("num", "name", "relation", "relname", "address")})
        else:  # keep unparseable chunks so nothing is silently dropped
            parties.append({"num": str(len(parties) + 1), "name": chunk.strip(),
                            "relation": "", "relname": "", "address": ""})
    return parties


_used_pdfs = set()
_pdf_cache = {}

def _all_pdfs(data_dir):
    """Scan the data folder once and reuse the list (rglob per deed is very
    slow on Windows Docker bind mounts)."""
    if data_dir not in _pdf_cache:
        _pdf_cache[data_dir] = list(Path(data_dir).rglob("*.pdf"))
    return _pdf_cache[data_dir]


def find_pdf(data_dir, book, regno, year):
    """Match R0xx_{regno}_{year}_{book}*.pdf; filenames use execution year which can
    drift +-1 from the registration year in the Excel, so try exact then +-1."""
    for y in (year, year - 1, year + 1):
        pat = re.compile(rf"^R\d+_{regno}_{y}_{book}\b", re.I)
        for p in _all_pdfs(data_dir):
            if pat.match(p.name) and str(p) not in _used_pdfs:
                _used_pdfs.add(str(p))
                return p
    return None


def year_of(row):
    if pd.notna(row.get("DEED_EXECUTED_YEAR")):
        return int(row["DEED_EXECUTED_YEAR"])
    for col in ("REGISTRATION_DATE", "EXECUTION_DATE"):
        v = str(row.get(col, ""))
        m = re.search(r"(19|20)\d\d", v)
        if m:
            return int(m.group())
    return None


def add_field(fields, section, label, value, multiline=False):
    v = "" if pd.isna(value) else str(value).strip()
    fields.append((section, label, v, multiline))


def ingest(xlsx_path, data_dir, scans_dir="static/scans"):
    Path(scans_dir).mkdir(parents=True, exist_ok=True)
    init_db()
    con = connect()
    xl = pd.ExcelFile(xlsx_path)
    loaded = skipped = 0

    for sheet in xl.sheet_names:
        df = pd.read_excel(xl, sheet)
        for _, row in df.iterrows():
            book = int(row["DEED_BOOK_NO"])
            regno = int(row["DEED_OLD_REG_NO"])
            year = year_of(row)
            deed_no = f"{regno}/{year}/B{book}"

            if con.execute("SELECT 1 FROM documents WHERE deed_number=%s", (deed_no,)).fetchone():
                skipped += 1
                continue

            pdf = find_pdf(data_dir, book, regno, year)
            pdf_name = None
            if pdf:
                pdf_name = f"{regno}_{year}_{book}.pdf"
                shutil.copy(pdf, Path(scans_dir) / pdf_name)

            exec_year = None
            if pd.notna(row.get("DEED_EXECUTED_YEAR")):
                exec_year = int(row["DEED_EXECUTED_YEAR"])
            vol = None if pd.isna(row.get("DEED_VOLUME_NO")) else str(row["DEED_VOLUME_NO"])
            doc_id = con.execute(
                "INSERT INTO documents (deed_number, deed_type, year, book_no, reg_no, "
                "sr_office, district, volume_no, executed_year, pdf_file, status) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,'pending') RETURNING id",
                (deed_no, str(row["DEED_TYPE"]).strip(), year, book, regno,
                 str(row["DEED_REGISTRATION_OFFICE"]).strip(),
                 str(row["DEED_DISTRICT"]).strip(), vol, exec_year, pdf_name)).fetchone()["id"]

            fields = []
            add_field(fields, "Deed", "Registration no", regno)
            add_field(fields, "Deed", "Deed type", row["DEED_TYPE"])
            add_field(fields, "Deed", "Volume no", row.get("DEED_VOLUME_NO"))
            add_field(fields, "Deed", "Execution date", row.get("EXECUTION_DATE"))
            add_field(fields, "Deed", "Presentation date", row.get("PRESENTATION_DATE"))
            add_field(fields, "Deed", "Registration date", row.get("REGISTRATION_DATE"))
            if "CONSIDERATION_AMOUNT" in row.index:
                add_field(fields, "Deed", "Consideration (Rs)", row.get("CONSIDERATION_AMOUNT"))

            for side, col in (("First party", "FIRST_PARTY_DETAILS"),
                              ("Second party", "SECOND_PARTY_DETAILS")):
                for p in parse_parties(row.get(col)):
                    sec = f"{side} {p['num']}"
                    add_field(fields, sec, "Name", p["name"])
                    add_field(fields, sec, "Relation", p["relation"])
                    add_field(fields, sec, "Relation name", p["relname"])
                    add_field(fields, sec, "Address", p["address"], multiline=True)

            if "PROPERTY_DETAILS" in row.index and pd.notna(row.get("PROPERTY_DETAILS")):
                add_field(fields, "Property", "Property details", row["PROPERTY_DETAILS"], multiline=True)

            with con.cursor() as cur:
                cur.executemany(
                    "INSERT INTO fields (document_id, section, label, ocr_value, current_value, multiline, position) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s)",
                    [(doc_id, s, l, v, v, bool(m), i) for i, (s, l, v, m) in enumerate(fields)])
            loaded += 1
            print(f"loaded {deed_no}  pdf={'yes' if pdf_name else 'MISSING'}", flush=True)

    con.commit()
    con.close()
    print(f"\ndone: {loaded} loaded, {skipped} already present")


if __name__ == "__main__":
    xlsx = sys.argv[1] if len(sys.argv) > 1 else "data/data-85-94/DEED 85-94.xlsx"
    ddir = sys.argv[2] if len(sys.argv) > 2 else "data/data-85-94"
    ingest(xlsx, ddir)
