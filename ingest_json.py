"""
ingest_json.py — load deeds from PDF + Akshar-JSON pairs.

NEW INPUT FORMAT (replaces the old Excel + PDF ingest)
------------------------------------------------------
Instead of an Excel sheet of metadata, each deed now arrives as:

    a PDF          e.g.  R050_1334_1986_1=OK.pdf
    a JSON folder  holding that deed's Akshar page output:
                        page_001.json, page_002.json, ...

The PDF filename encodes the deed identity:  R0xx_{regno}_{year}_{book}
  -> deed_number = "{regno}/{year}/B{book}"   e.g. 1334/1986/B1

Akshar's page JSON has, per page, a list of blocks:
    layout_tag   header | paragraph | footer | table | image-caption | ...
    text         plain text, or <table>...</table> HTML for table blocks
    reading_order

The portal's Metadata tab mirrors these blocks (one field per block, headed
by its layout tag, tables as editable grids); the Full text tab is the same
blocks flowed together as paragraphs.

LAYOUT ON DISK
--------------
Put everything under data/. Two accepted arrangements:

  (A) PDFs in one place, each deed's JSON in a sibling folder whose name
      contains the same deed id, e.g.:
          data/BOOK-1/R050_1334_1986_1=OK.pdf
          data/BOOK-1/R050_1334_1986_1/page_001.json ...
  (B) A single deed's JSON folder passed directly (dev / one-off).

USAGE
    python ingest_json.py data            # scan data/ for all PDF+JSON deeds
    python ingest_json.py <json_folder> <deed_number>   # load one folder
"""

import json
import re
import shutil
import sys
from pathlib import Path

from db import init_db, connect

DEED_RE = re.compile(r"R\d+[_-](\d+)[_-](\d+)[_-](\d+)", re.I)  # regno, year, book


def parse_deed_id(name):
    """From 'R050_1334_1986_1=OK' -> (regno, year, book, deed_number)."""
    m = DEED_RE.search(name)
    if not m:
        return None
    regno, year, book = int(m.group(1)), int(m.group(2)), int(m.group(3))
    return regno, year, book, f"{regno}/{year}/B{book}"


def _tables_to_text(html):
    rows = []
    for row in re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.S | re.I):
        cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, re.S | re.I)
        vals = []
        for c in cells:
            c = re.sub(r"<br\s*/?>", " ", c, flags=re.I)
            c = re.sub(r"<[^>]+>", "", c).strip()
            vals.append(c)
        line = " | ".join(v for v in vals if v).strip()
        if line:
            rows.append(line)
    return "\n".join(rows)


def _block_to_fulltext(tag, text):
    if tag == "table":
        return _tables_to_text(text)
    t = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
    t = re.sub(r"<[^>]+>", "", t)
    return t.strip()


def _pretty_tag(tag):
    return (tag or "block").replace("-", " ").replace("_", " ").title()


def _load_pages(json_dir):
    """Return blocks across all page_*.json in a folder (page, order sorted)."""
    p = Path(json_dir)
    pages = sorted(p.glob("page_*.json")) or sorted(p.glob("metadata/page_*.json"))
    blocks = []
    for pg in pages:
        d = json.load(open(pg, encoding="utf-8"))
        pnum = d.get("page_num", 0)
        for b in sorted(d.get("blocks", []), key=lambda x: x.get("reading_order", 0)):
            # keep (page, order, tag, text, full-original-block) so export can
            # emit the exact same JSON shape with only the text corrected
            blocks.append((pnum, b.get("reading_order", 0),
                           b.get("layout_tag", "block"), b.get("text", ""), b))
    return blocks


def _insert_deed(con, deed_number, blocks, book=None, regno=None, year=None,
                 pdf_name=None):
    if con.execute("SELECT 1 FROM documents WHERE deed_number=%s", (deed_number,)).fetchone():
        return False

    full_text = "\n\n".join(
        t for t in (_block_to_fulltext(tag, txt) for _, _, tag, txt, _b in blocks) if t)

    doc_id = con.execute(
        "INSERT INTO documents (deed_number, year, book_no, reg_no, pdf_file, "
        "status, digitized_text, digitized_status) "
        "VALUES (%s,%s,%s,%s,%s,'pending',%s,%s) RETURNING id",
        (deed_number, year, book, regno, pdf_name,
         full_text, "ready" if full_text else "not_started")).fetchone()["id"]

    rows = []
    for i, (pnum, order, tag, txt, block) in enumerate(blocks):
        is_table = (tag == "table")
        section = f"Page {pnum}" if pnum else "Document"
        label = _pretty_tag(tag)
        if is_table:
            value, kind, multiline = txt.strip(), "table", True
        else:
            value, kind, multiline = _block_to_fulltext(tag, txt), "text", True
        rows.append((doc_id, section, label, value, value, multiline, i, kind, tag,
                     json.dumps(block), pnum))
    if rows:
        with con.cursor() as cur:
            cur.executemany(
                "INSERT INTO fields (document_id, section, label, ocr_value, current_value, "
                "multiline, position, field_kind, layout_tag, src_block, page_num) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                rows)
    return True


def _find_json_dir_for(deed_id_name, data_dir):
    """Find the JSON folder for a PDF by matching the deed id (regno_year_book)
    in the folder's own name (or its parent). Strict — no borrowing another
    deed's JSON."""
    parsed = parse_deed_id(deed_id_name)
    if not parsed:
        return None
    data = Path(data_dir)
    for cand in data.rglob("page_001.json"):
        folder = cand.parent
        pinfo = parse_deed_id(folder.name) or parse_deed_id(folder.parent.name)
        if pinfo and pinfo[:3] == parsed[:3]:
            return folder
    return None


def ingest_dir(data_dir, scans_dir="static/scans", init=True):
    """Scan data_dir for deed PDFs and pair each with its JSON folder.
    init=False when the caller has already run init_db() (avoids a nested
    schema migration that would collide on table locks)."""
    Path(scans_dir).mkdir(parents=True, exist_ok=True)
    if init:
        init_db()
    con = connect()
    data = Path(data_dir)
    pdfs = sorted(data.rglob("R*.pdf"))
    loaded = skipped = nojson = 0
    for pdf in pdfs:
        parsed = parse_deed_id(pdf.stem)
        if not parsed:
            continue
        regno, year, book, deed_no = parsed
        json_dir = _find_json_dir_for(pdf.stem, data_dir)
        if not json_dir:
            nojson += 1
            print(f"no JSON found for {deed_no} ({pdf.name})", flush=True)
            continue
        blocks = _load_pages(json_dir)
        pdf_name = f"{regno}_{year}_{book}.pdf"
        shutil.copy(pdf, Path(scans_dir) / pdf_name)
        ok = _insert_deed(con, deed_no, blocks, book, regno, year, pdf_name)
        if ok:
            loaded += 1
            print(f"loaded {deed_no}: {len(blocks)} blocks, pdf={pdf_name}", flush=True)
        else:
            skipped += 1
    con.commit()
    con.close()
    print(f"\ndone: {loaded} loaded, {skipped} already present, {nojson} missing JSON")


def load_akshar(json_dir, deed_number=None, scans_dir="static/scans"):
    """Load a single deed from one JSON folder (dev / one-off)."""
    init_db()
    con = connect()
    blocks = _load_pages(json_dir)
    deed_number = deed_number or Path(json_dir).name
    parsed = parse_deed_id(deed_number) or parse_deed_id(Path(json_dir).name)
    kw = {}
    if parsed:
        kw = dict(regno=parsed[0], year=parsed[1], book=parsed[2])
        deed_number = deed_number if "/" in str(deed_number) else parsed[3]
    ok = _insert_deed(con, deed_number, blocks, **kw)
    con.commit()
    con.close()
    print(f"{'loaded' if ok else 'already present'} {deed_number}: {len(blocks)} blocks")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python ingest_json.py <data_dir>  |  <json_folder> <deed_number>")
        sys.exit(1)
    arg = sys.argv[1]
    if len(sys.argv) >= 3:
        load_akshar(arg, sys.argv[2])
    else:
        ingest_dir(arg)
