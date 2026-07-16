"""
export_json.py — export corrected deeds as JSON, in the SAME per-page block
shape that Akshar produced on input. Only the block `text` is replaced with
the expert's corrected value; block_id / coordinates / confidence / layout_tag
/ reading_order are preserved exactly. Output is a ZIP:

    <deed>/metadata/page_001.json
    <deed>/metadata/page_002.json
    ...

one folder per deed, mirroring the input layout (no PDFs).
"""

import io
import json
import zipfile
from collections import defaultdict

from db import connect


def _corrected_blocks_by_deed():
    """Return {deed_number: {page_num: [block, ...]}} with corrected text."""
    with connect() as con:
        rows = con.execute(
            "SELECT d.deed_number, f.current_value, f.field_kind, f.src_block, "
            "       f.page_num, f.position "
            "FROM fields f JOIN documents d ON d.id = f.document_id "
            "WHERE f.src_block IS NOT NULL "
            "ORDER BY d.deed_number, f.position"
        ).fetchall()

    deeds = defaultdict(lambda: defaultdict(list))
    for r in rows:
        block = r["src_block"]  # original Akshar block (JSONB -> dict)
        if isinstance(block, str):
            block = json.loads(block)
        # replace only the text with the corrected value
        block = dict(block)
        block["text"] = r["current_value"] or ""
        page = r["page_num"] or block.get("page_num") or 0
        deeds[r["deed_number"]][page].append(block)
    return deeds


def build_export_zip() -> io.BytesIO:
    """Build a ZIP of corrected per-page JSON, mirroring the Akshar input."""
    deeds = _corrected_blocks_by_deed()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for deed_number, pages in deeds.items():
            # deed folder name from the deed number, e.g. 1334/1986/B1 -> 1334_1986_B1
            safe = deed_number.replace("/", "_")
            for page_num, blocks in sorted(pages.items()):
                # sort blocks by their original reading order
                blocks_sorted = sorted(blocks, key=lambda b: b.get("reading_order", 0))
                page_obj = {
                    "page_num": page_num,
                    "blocks": blocks_sorted,
                }
                fname = f"{safe}/metadata/page_{page_num:03d}.json"
                zf.writestr(fname, json.dumps(page_obj, ensure_ascii=False, indent=2))
    buf.seek(0)
    return buf


def build_single_deed_json(deed_number) -> io.BytesIO:
    """Corrected JSON for one deed, as a ZIP of its per-page files."""
    deeds = _corrected_blocks_by_deed()
    pages = deeds.get(deed_number, {})
    buf = io.BytesIO()
    safe = deed_number.replace("/", "_")
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for page_num, blocks in sorted(pages.items()):
            blocks_sorted = sorted(blocks, key=lambda b: b.get("reading_order", 0))
            page_obj = {"page_num": page_num, "blocks": blocks_sorted}
            zf.writestr(f"{safe}/metadata/page_{page_num:03d}.json",
                        json.dumps(page_obj, ensure_ascii=False, indent=2))
    buf.seek(0)
    return buf
