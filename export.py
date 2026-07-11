"""Builds the corrected-dataset workbook in the SAME layout as the input Excel:
one sheet per book, original column names and order, party details reassembled
into the '1-NAME ( RELATION : ) ... ,2-...' string format — but containing the
experts' corrected values. An extra 'Audit' sheet lists every changed field.
"""
import io
import re
from collections import defaultdict

import pandas as pd

from db import connect

SHEET_NAMES = {1: "BOOK-1", 3: "BOOK_3", 4: "BOOK_4"}

BASE_COLS = ["SERIAL", "DEED_DISTRICT", "DEED_REGISTRATION_OFFICE", "DEED_BOOK_NO",
             "DEED_OLD_REG_NO"]
TAIL_COLS = ["EXECUTION_DATE", "PRESENTATION_DATE", "REGISTRATION_DATE"]

FIELD_TO_COL = {
    "Registration no": "DEED_OLD_REG_NO",
    "Deed type": "DEED_TYPE",
    "Volume no": "DEED_VOLUME_NO",
    "Execution date": "EXECUTION_DATE",
    "Presentation date": "PRESENTATION_DATE",
    "Registration date": "REGISTRATION_DATE",
    "Consideration (Rs)": "CONSIDERATION_AMOUNT",
    "Property details": "PROPERTY_DETAILS",
}


def party_string(parties):
    """Reassemble structured party fields into the input string format."""
    chunks = []
    for n in sorted(parties, key=int):
        p = parties[n]
        chunks.append(
            f"{n}-{p.get('Name','')}   ( RELATION : )  {p.get('Relation','')}  "
            f"(  RELATION NAME : )  {p.get('Relation name','')}  "
            f"(  ADDRESS : )  {p.get('Address','')}")
    return " ,".join(chunks)


def columns_for_book(book, has_consideration, has_property, has_exec_year):
    cols = list(BASE_COLS)
    if has_exec_year:
        cols.append("DEED_EXECUTED_YEAR")
    cols += TAIL_COLS
    if has_consideration:
        cols.append("CONSIDERATION_AMOUNT")
    cols += ["DEED_TYPE", "DEED_VOLUME_NO", "FIRST_PARTY_DETAILS", "SECOND_PARTY_DETAILS"]
    if has_property:
        cols.append("PROPERTY_DETAILS")
    return cols


def build_export_workbook() -> io.BytesIO:
    with connect() as con:
        docs = con.execute(
            "SELECT d.*, u.full_name validated_name FROM documents d "
            "LEFT JOIN users u ON u.id = d.validated_by "
            "ORDER BY d.book_no, d.year, d.reg_no").fetchall()
        all_fields = con.execute(
            "SELECT document_id, section, label, ocr_value, current_value "
            "FROM fields ORDER BY document_id, position").fetchall()

    fields_by_doc = defaultdict(list)
    for f in all_fields:
        fields_by_doc[f["document_id"]].append(f)

    per_book_rows = defaultdict(list)
    audit_rows = []

    for d in docs:
        row = {
            "DEED_DISTRICT": d["district"],
            "DEED_REGISTRATION_OFFICE": d["sr_office"],
            "DEED_BOOK_NO": d["book_no"],
        }
        if d["executed_year"]:
            row["DEED_EXECUTED_YEAR"] = d["executed_year"]

        first, second = defaultdict(dict), defaultdict(dict)
        for f in fields_by_doc[d["id"]]:
            sec, label, val = f["section"], f["label"], f["current_value"]
            m = re.match(r"(First|Second) party (\d+)", sec)
            if m:
                (first if m.group(1) == "First" else second)[m.group(2)][label] = val
            elif label in FIELD_TO_COL:
                row[FIELD_TO_COL[label]] = val
            if f["ocr_value"] != f["current_value"]:
                audit_rows.append({
                    "DEED": d["deed_number"], "SECTION": sec, "FIELD": label,
                    "OCR_VALUE": f["ocr_value"], "CORRECTED_VALUE": val,
                    "STATUS": d["status"], "VALIDATED_BY": d["validated_name"]})

        row["FIRST_PARTY_DETAILS"] = party_string(first)
        row["SECOND_PARTY_DETAILS"] = party_string(second)
        row["_STATUS"] = d["status"]
        per_book_rows[d["book_no"]].append(row)

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        for book in sorted(per_book_rows):
            rows = per_book_rows[book]
            cols = columns_for_book(
                book,
                has_consideration=any("CONSIDERATION_AMOUNT" in r for r in rows),
                has_property=any("PROPERTY_DETAILS" in r for r in rows),
                has_exec_year=any("DEED_EXECUTED_YEAR" in r for r in rows))
            for i, r in enumerate(rows, 1):
                r["SERIAL"] = i
            df = pd.DataFrame(rows)
            for c in cols:
                if c not in df.columns:
                    df[c] = ""
            # VALIDATION_STATUS is appended after the original columns so the
            # client can filter to validated rows without breaking their template
            df = df[cols + ["_STATUS"]].rename(columns={"_STATUS": "VALIDATION_STATUS"})
            df.to_excel(xw, sheet_name=SHEET_NAMES.get(book, f"BOOK_{book}"), index=False)
        pd.DataFrame(audit_rows or [{}]).to_excel(xw, sheet_name="Audit", index=False)
    buf.seek(0)
    return buf


if __name__ == "__main__":
    out = build_export_workbook()
    with open("corrected_deeds.xlsx", "wb") as fh:
        fh.write(out.read())
    print("wrote corrected_deeds.xlsx")
