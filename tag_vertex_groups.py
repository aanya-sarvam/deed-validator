"""One-time script: tag the vertex batch (source='vertex') deeds as
'mismatch' or 'control' for DASHBOARD REPORTING ONLY.

This does NOT touch is_priority, assignment, status, or anything the
validation queue reads — vertex_group is a purely read-only reporting column.

Usage (PowerShell):
    python tag_vertex_groups.py path\to\random_500_reg_nos.txt

What it does:
  1. Reads the file of the 500 random-control reg_nos (one per line).
  2. Among documents where source='vertex':
       - reg_no (deed_number) IN the 500 list  -> vertex_group = 'control'
       - every other vertex deed                -> vertex_group = 'mismatch'
  3. Prints a summary and a couple of sanity checks (counts should be
     ~500 / ~502, and every 500-list reg_no should actually have been found
     in the vertex batch — any misses are printed so you can double check).

Safe to re-run: it always recomputes both groups from scratch based on the
current file, it never deletes documents, and it never touches any other
column.
"""
import sys
from pathlib import Path

from db import connect, init_db


def main(path: str):
    p = Path(path)
    if not p.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    control_regnos = {
        line.strip() for line in p.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    print(f"Loaded {len(control_regnos)} control reg_nos from {path}")

    init_db()  # ensures vertex_group column exists (idempotent migration)
    con = connect()
    try:
        vertex_regnos = {
            r["deed_number"] for r in con.execute(
                "SELECT deed_number FROM documents WHERE source='vertex'"
            ).fetchall()
        }
        print(f"{len(vertex_regnos)} deeds currently in the vertex batch (source='vertex')")

        missing = control_regnos - vertex_regnos
        if missing:
            print(f"WARNING: {len(missing)} reg_nos from the control file were "
                  f"NOT found among vertex deeds (not ingested yet, or typo?). "
                  f"First few: {sorted(missing)[:10]}")

        with con.cursor() as cur:
            cur.execute(
                "UPDATE documents SET vertex_group = 'control' "
                "WHERE source='vertex' AND deed_number = ANY(%s)",
                (list(control_regnos),))
            n_control = cur.rowcount
            cur.execute(
                "UPDATE documents SET vertex_group = 'mismatch' "
                "WHERE source='vertex' AND NOT (deed_number = ANY(%s))",
                (list(control_regnos),))
            n_mismatch = cur.rowcount
        con.commit()

        print(f"Tagged {n_control} deeds as 'control'")
        print(f"Tagged {n_mismatch} deeds as 'mismatch'")

        # sanity check: re-read counts from the DB
        counts = con.execute(
            "SELECT vertex_group, COUNT(*) c FROM documents "
            "WHERE source='vertex' GROUP BY vertex_group"
        ).fetchall()
        print("Final vertex_group counts:", {r["vertex_group"]: r["c"] for r in counts})
    finally:
        con.close()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python tag_vertex_groups.py <path_to_500_control_reg_nos.txt>")
        sys.exit(1)
    main(sys.argv[1])
