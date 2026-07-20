"""
export_json.py — export corrected deeds as grounding.json, in the SAME shape
as the input. Each field keeps its original id/attr/item_index/page/confidence
etc.; only english_value and odia_text are replaced with the experts'
corrected values. The corrected full text (from the Full text tab) is included
as fulltext.txt. Output ZIP mirrors the input layout (no PDFs):

    <reg_no>/grounding.json
    <reg_no>/fulltext.txt
"""

import io
import json
import zipfile

from db import connect


def _split_csv(s):
    return [p.strip() for p in (s or "").split(",")]


def _expand_group(block, english, odia):
    """A merged group field back into per-item grounding fields. The corrected
    comma-separated string is authoritative: item count follows the splits."""
    items = block.get("items", [])
    evals = _split_csv(english)
    ovals = _split_csv(odia)
    n = max(len(evals), len(ovals), 1)
    out = []
    for i in range(n):
        base = items[i] if i < len(items) else dict(items[-1] if items else {},
                                                    found=False, notes="added by expert",
                                                    confidence=None, page=None)
        b = dict(base)
        b["item_index"] = i + 1
        b["english_value"] = evals[i] if i < len(evals) else ""
        b["odia_text"] = ovals[i] if i < len(ovals) else ""
        out.append(b)
    return out


def _corrected_grounding(con, doc_id):
    doc = con.execute(
        "SELECT deed_number, src_meta, digitized_text FROM documents WHERE id=%s",
        (doc_id,)).fetchone()
    if not doc:
        return None, None, None
    meta = doc["src_meta"] or {}
    if isinstance(meta, str):
        meta = json.loads(meta)
    rows = con.execute(
        "SELECT current_value, odia_value, src_block FROM fields "
        "WHERE document_id=%s AND src_block IS NOT NULL ORDER BY position",
        (doc_id,)).fetchall()
    fields = []
    for r in rows:
        block = r["src_block"]
        if isinstance(block, str):
            block = json.loads(block)
        if isinstance(block, dict) and block.get("group"):
            fields.extend(_expand_group(block, r["current_value"], r["odia_value"]))
        else:
            block = dict(block)
            block["english_value"] = r["current_value"] or ""
            block["odia_text"] = r["odia_value"] or ""
            fields.append(block)
    grounding = {**meta, "fields": fields}
    return doc["deed_number"], grounding, doc["digitized_text"]


def build_export_zip() -> io.BytesIO:
    """Corrected grounding.json (+ corrected full text) for every deed."""
    buf = io.BytesIO()
    with connect() as con:
        ids = [r["id"] for r in
               con.execute("SELECT id FROM documents ORDER BY deed_number").fetchall()]
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for doc_id in ids:
                reg_no, grounding, fulltext = _corrected_grounding(con, doc_id)
                if not reg_no:
                    continue
                zf.writestr(f"{reg_no}/grounding.json",
                            json.dumps(grounding, ensure_ascii=False, indent=2))
                if fulltext:
                    zf.writestr(f"{reg_no}/fulltext.txt", fulltext)
    buf.seek(0)
    return buf


def build_single_deed_json(deed_number) -> io.BytesIO:
    buf = io.BytesIO()
    with connect() as con:
        d = con.execute("SELECT id FROM documents WHERE deed_number=%s",
                        (deed_number,)).fetchone()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            if d:
                reg_no, grounding, fulltext = _corrected_grounding(con, d["id"])
                zf.writestr(f"{reg_no}/grounding.json",
                            json.dumps(grounding, ensure_ascii=False, indent=2))
                if fulltext:
                    zf.writestr(f"{reg_no}/fulltext.txt", fulltext)
    buf.seek(0)
    return buf
