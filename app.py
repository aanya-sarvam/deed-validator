"""Deed validation backend (PostgreSQL).

Run:  uvicorn app:app --port 8000
Env:  DATABASE_URL=postgresql://deeds:deeds@localhost:5432/deeds
Seed logins (rotate in prod): expert1..expert3 / admin, password sarvam123
"""
import secrets
import json
from pathlib import Path

from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from db import connect, init_db, hash_pw, check_pw, SESSION_HOURS

LOCK_TIMEOUT_MIN = 30
app = FastAPI(title="Deed Validation")
# NOTE: init_db() is intentionally NOT called here at import time. On some
# platforms the database isn't accepting connections yet when the app image
# boots; calling it here would crash the import and the web port would never
# open. It runs in the background startup thread (with retries) instead.


def _repair_scans(con):
    """If the DB references scans that are missing on disk (e.g. after a
    redeploy on an ephemeral filesystem) and the source data folder is
    present, re-copy them from source."""
    import shutil
    docs = con.execute(
        "SELECT reg_no, year, book_no, pdf_file FROM documents "
        "WHERE pdf_file IS NOT NULL").fetchall()
    scans = Path("static/scans")
    scans.mkdir(parents=True, exist_ok=True)
    missing = [d for d in docs if not (scans / d["pdf_file"]).exists()]
    if not missing or not Path("data").exists():
        return
    from ingest_json import find_pdf
    repaired = 0
    for d in missing:
        # deed_number IS the reg_no in the grounding format; pdf named <reg_no>.pdf
        reg = d["pdf_file"].rsplit(".", 1)[0]
        src = find_pdf("data", reg)
        if src:
            shutil.copy(src, scans / d["pdf_file"])
            repaired += 1
    print(f"[startup] repaired {repaired}/{len(missing)} missing scan files")


_ingest_status = {"state": "not_started", "detail": "", "documents": 0}


def _auto_ingest():
    """Runs in a background thread after startup. Waits for the database to be
    reachable, initialises the schema, then loads sample data if the DB is
    empty. Logs loudly on failure. The web port is already open by now, so a
    slow database never blocks the platform's health check."""
    import glob
    import time
    import traceback

    # 1) wait for DB + init schema, with retries (free-tier DBs boot slowly)
    for attempt in range(1, 31):
        try:
            init_db()
            break
        except Exception as e:
            _ingest_status.update(state="waiting_for_db", detail=str(e))
            print(f"[startup] DB not ready (attempt {attempt}/30): {e}", flush=True)
            time.sleep(3)
    else:
        _ingest_status.update(state="error", detail="database never became reachable")
        print("[startup] ERROR: database never became reachable", flush=True)
        return

    # 2) load data if the DB is empty.
    #    We run a single worker (see Dockerfile), so no cross-worker race exists.
    #    We use a NON-BLOCKING advisory lock: if for any reason another process
    #    holds it, we simply skip rather than hang forever (a blocking lock left
    #    behind by a killed process was causing the app to freeze on boot).
    try:
        with connect() as con:
            got_lock = con.execute("SELECT pg_try_advisory_lock(424242) AS ok").fetchone()["ok"]
            if not got_lock:
                # someone else is already ingesting; just report current count
                n = con.execute("SELECT COUNT(*) c FROM documents").fetchone()["c"]
                _ingest_status.update(state="done", documents=n, detail="ingest handled elsewhere")
                print("[startup] ingest lock held elsewhere; skipping", flush=True)
                return
            try:
                n = con.execute("SELECT COUNT(*) c FROM documents").fetchone()["c"]
                if n:
                    # backfill: populate year from the registration/presentation
                    # date fields (English or Odia value, 2- or 4-digit year) for
                    # deeds still missing one. Idempotent.
                    from ingest_json import _year_from_text
                    rows = con.execute(
                        "SELECT f.document_id, f.layout_tag, f.current_value, f.odia_value "
                        "FROM fields f JOIN documents d ON d.id = f.document_id "
                        "WHERE d.year IS NULL "
                        "  AND f.layout_tag IN ('registration_date','presentation_date') "
                        "ORDER BY f.document_id, (f.layout_tag='registration_date') DESC"
                    ).fetchall()
                    years = {}
                    for r in rows:
                        if r["document_id"] in years:
                            continue
                        y = _year_from_text(r["current_value"]) or _year_from_text(r["odia_value"])
                        if y:
                            years[r["document_id"]] = y
                    for doc_id, y in years.items():
                        con.execute("UPDATE documents SET year=%s WHERE id=%s", (y, doc_id))
                    con.commit()
                    if years:
                        print(f"[startup] backfilled year on {len(years)} documents", flush=True)
                    # in-place migration: merge per-item party fields into
                    # comma-separated fields (idempotent, preserves corrections)
                    from ingest_json import merge_existing_party_fields, _merge_enabled
                    if _merge_enabled():
                        m = merge_existing_party_fields(con)
                        if m:
                            print(f"[startup] merged party fields on {m} documents", flush=True)
                    from ingest_json import backfill_book1_consideration
                    b1 = backfill_book1_consideration(con)
                    if b1:
                        print(f"[startup] backfilled Consideration Amount on {b1} Book 1 documents", flush=True)
                    from ingest_json import reposition_consideration_amount
                    rp = reposition_consideration_amount(con)
                    if rp:
                        print(f"[startup] repositioned {rp} mis-placed Consideration Amount field(s)", flush=True)
                    _ingest_status.update(state="done", documents=n, detail="already loaded")
                    _repair_scans(con)
                    print(f"[startup] already loaded — {n} documents", flush=True)
                    return

                import gcs_store
                if gcs_store.enabled():
                    from ingest_json import ingest_gcs
                    _ingest_status.update(state="running", detail="reading from GCS bucket")
                    print("[startup] auto-ingesting from GCS bucket", flush=True)
                    ingest_gcs(init=False)
                    n = con.execute("SELECT COUNT(*) c FROM documents").fetchone()["c"]
                    _ingest_status.update(state="done", documents=n, detail="GCS")
                    print(f"[startup] GCS ingest complete — {n} documents", flush=True)
                    return

                groundings = glob.glob("data/**/grounding.json", recursive=True)
                if groundings:
                    from ingest_json import ingest_dir
                    _ingest_status.update(state="running", detail=f"{len(groundings)} deeds")
                    print(f"[startup] auto-ingesting {len(groundings)} grounding-format deed(s) from data/", flush=True)
                    ingest_dir("data", init=False)
                    n = con.execute("SELECT COUNT(*) c FROM documents").fetchone()["c"]
                    _ingest_status.update(state="done", documents=n, detail="grounding+ocr")
                    print(f"[startup] ingest complete — {n} documents", flush=True)
                    return

                _ingest_status.update(state="error", detail="no deed folders (grounding.json) under data/")
                print("[startup] ERROR: no grounding.json deed folders found under data/", flush=True)
            finally:
                con.execute("SELECT pg_advisory_unlock(424242)")
    except Exception as e:
        _ingest_status.update(state="error", detail=str(e))
        print("[startup] INGEST FAILED:", flush=True)
        traceback.print_exc()


_auto_ingest_started = False


@app.on_event("startup")
def _startup():
    global _auto_ingest_started
    if _auto_ingest_started:
        return
    _auto_ingest_started = True
    import threading
    threading.Thread(target=_auto_ingest, daemon=True).start()


@app.get("/api/health")
def health():
    # Must NOT touch the database — this is the platform's port/liveness probe.
    # If it depended on the DB, a slow or not-yet-ready DB would make the
    # platform think the app never started and kill it.
    return {"ok": True}


@app.get("/api/ingest-status")
def ingest_status():
    """Diagnostic: shows whether first-boot data loading ran and its result."""
    import glob
    import os
    try:
        with connect() as con:
            docs = con.execute("SELECT COUNT(*) c FROM documents").fetchone()["c"]
    except Exception as e:
        docs = f"db error: {e}"
    return {**_ingest_status, "documents_now": docs,
            "deeds_found": len(glob.glob("data/**/grounding.json", recursive=True)),
            "data_dir_exists": os.path.isdir("data"),
            "cwd": os.getcwd()}


@app.get("/api/ready")
def ready():
    """Reports whether first-boot data loading has finished."""
    try:
        with connect() as con:
            n = con.execute("SELECT COUNT(*) c FROM documents").fetchone()["c"]
        return {"loading": n == 0, "documents": n, "state": _ingest_status["state"]}
    except Exception as e:
        return {"loading": True, "documents": 0,
                "state": _ingest_status["state"], "detail": str(e)}


# ---------- auth ----------

class LoginIn(BaseModel):
    username: str
    password: str


def current_user(token: str = Query(...)):
    with connect() as con:
        row = con.execute(
            "SELECT u.* FROM sessions s JOIN users u ON u.id = s.user_id "
            "WHERE s.token = %s AND s.expires_at > now()", (token,)).fetchone()
    if not row:
        raise HTTPException(401, "Session expired — please sign in again")
    return row


def require_admin(user=Depends(current_user)):
    if user["role"] != "admin":
        raise HTTPException(403, "Admin access required")
    return user


def require_monitor(user=Depends(current_user)):
    if user["role"] not in ("monitor", "admin"):
        raise HTTPException(403, "Monitor access required")
    return user


def _run_incremental_ingest():
    """Checks every configured source for deeds not yet in the DB and loads
    them: GCS sample_1000-style pre-made-PDF batches, the raw orissa_deeds
    export straight from GCS (no local files, no bucket writes — PDFs build
    lazily on first view), and local data/ folders. Each source skips
    deed_numbers already present, so this is safe to call anytime — startup
    or Admin -> Reingest — without resetting or duplicating anything, unlike
    _auto_ingest() above which only loads data on a first, empty-DB boot."""
    import glob
    try:
        _ingest_status.update(state="running", detail="reingest started")

        def _progress(detail, documents=None):
            upd = {"state": "running", "detail": detail}
            if documents is not None:
                upd["documents"] = documents
            _ingest_status.update(**upd)

        def _gcs_progress(n, total, loaded):
            _progress(f"gcs sample: {n}/{total} ({loaded} loaded)")

        def _gcs_raw_progress(n, total, loaded, prefix=None):
            pfx = f" [{prefix}]" if prefix else ""
            _progress(f"gcs-raw{pfx}: {n}/{total} lines ({loaded} loaded)")

        import gcs_store
        if gcs_store.enabled():
            from ingest_json import ingest_gcs, ingest_gcs_raw
            _progress("checking GCS sample_1000-style batches")
            print("[reingest] checking GCS sample_1000-style batches...", flush=True)
            ingest_gcs(init=False, progress=_gcs_progress)
            _progress("checking raw orissa_deeds export on GCS")
            print("[reingest] checking raw orissa_deeds export on GCS...", flush=True)
            ingest_gcs_raw(init=False, progress=_gcs_raw_progress)
        groundings = glob.glob("data/**/grounding.json", recursive=True)
        if groundings:
            from ingest_json import ingest_dir
            _progress(f"checking {len(groundings)} local deed folder(s)")
            print(f"[reingest] checking {len(groundings)} local deed folder(s)...", flush=True)
            ingest_dir("data", init=False)
        with connect() as con:
            n = con.execute("SELECT COUNT(*) c FROM documents").fetchone()["c"]
        _ingest_status.update(state="done", documents=n, detail="reingest complete")
        print(f"[reingest] done — {n} documents total", flush=True)
    except Exception as e:
        _ingest_status.update(state="error", detail=str(e))
        print("[reingest] FAILED:", flush=True)
        import traceback
        traceback.print_exc()


@app.post("/api/admin/reingest")
def admin_reingest(user=Depends(require_admin)):
    """Manually run ingestion for any NEW deeds across all configured
    sources. Safe: existing deeds are skipped everywhere — this is how you
    add a new batch (e.g. the raw orissa_deeds export) without resetting
    the DB."""
    import threading
    threading.Thread(target=_run_incremental_ingest, daemon=True).start()
    return {"ok": True, "message": "Ingestion started — check /api/ingest-status"}


@app.get("/api/admin/gcs-raw-diagnostics")
def gcs_raw_diagnostics(user=Depends(require_admin)):
    """Cheap read-only check: for each configured raw prefix, report whether
    the grounding and OCR jsonl files exist in the bucket and their sizes.
    Uses blob.exists() only — no full-file download."""
    import os
    import gcs_store
    if not gcs_store.enabled():
        return {"gcs_enabled": False, "bucket": None, "prefixes": []}
    out = []
    for prefix in gcs_store.raw_prefixes():
        grounding_path = f"{prefix}/grounding/grounding_good_partial.jsonl"
        ocr_path = f"{prefix}/ocr/ocr_dataset.jsonl"
        out.append({
            "prefix": prefix,
            "grounding_good_partial": {
                "path": grounding_path,
                **gcs_store.blob_stat(grounding_path),
            },
            "ocr_dataset": {
                "path": ocr_path,
                **gcs_store.blob_stat(ocr_path),
            },
        })
    return {"gcs_enabled": True, "bucket": os.environ.get("GCS_BUCKET"), "prefixes": out}


class SignupIn(BaseModel):
    username: str
    password: str
    full_name: str = ""


@app.post("/api/signup")
def signup(body: SignupIn):
    """Open self-registration. Creates an expert account, no approval needed."""
    uname = body.username.strip()
    if len(uname) < 3:
        raise HTTPException(400, "Username must be at least 3 characters")
    if len(body.password) < 6:
        raise HTTPException(400, "Password must be at least 6 characters")
    with connect() as con:
        try:
            con.execute(
                "INSERT INTO users (username, password_hash, full_name, role) "
                "VALUES (%s,%s,%s,'expert')",
                (uname, hash_pw(body.password), body.full_name.strip() or uname))
            con.commit()
        except Exception:
            raise HTTPException(409, "That username is already taken")
    return {"ok": True}


@app.post("/api/login")
def login(body: LoginIn):
    with connect() as con:
        row = con.execute("SELECT * FROM users WHERE username = %s",
                          (body.username.strip(),)).fetchone()
        if not row or not check_pw(body.password, row["password_hash"]):
            raise HTTPException(401, "Wrong username or password")
        token = secrets.token_urlsafe(32)
        con.execute("DELETE FROM sessions WHERE expires_at < now()")
        con.execute(
            "INSERT INTO sessions (token, user_id, expires_at) "
            "VALUES (%s, %s, now() + make_interval(hours => %s))",
            (token, row["id"], SESSION_HOURS))
        con.commit()
    return {"token": token, "role": row["role"],
            "full_name": row["full_name"], "user_id": row["id"]}


# ---------- documents ----------


class PasswordChange(BaseModel):
    old_password: str
    new_password: str


@app.post("/api/me/password")
def change_password(body: PasswordChange, token: str = Query(...),
                    user=Depends(current_user)):
    if len(body.new_password) < 6:
        raise HTTPException(400, "New password must be at least 6 characters")
    with connect() as con:
        row = con.execute("SELECT password_hash FROM users WHERE id = %s",
                          (user["id"],)).fetchone()
        if not check_pw(body.old_password, row["password_hash"]):
            raise HTTPException(401, "Current password is incorrect")
        con.execute("UPDATE users SET password_hash = %s WHERE id = %s",
                    (hash_pw(body.new_password), user["id"]))
        # sign out this user's OTHER sessions; keep the one making this request
        con.execute("DELETE FROM sessions WHERE user_id = %s AND token != %s",
                    (user["id"], token))
        con.commit()
    return {"ok": True}

def doc_payload(con, doc_id):
    doc = con.execute("SELECT * FROM documents WHERE id = %s", (doc_id,)).fetchone()
    if not doc:
        raise HTTPException(404, "Document not found")
    fields = con.execute(
        "SELECT id, section, label, ocr_value, current_value, odia_value, multiline, "
        "field_kind, layout_tag, src_block "
        "FROM fields WHERE document_id = %s ORDER BY position", (doc_id,)).fetchall()
    remaining = con.execute(
        "SELECT COUNT(*) c FROM documents WHERE status IN ('pending','in_review')"
    ).fetchone()["c"]
    doc = dict(doc)
    for k in ("locked_at", "validated_at", "reviewed_at"):
        doc[k] = str(doc[k]) if doc.get(k) else None
    return {"document": doc, "fields": [dict(f) for f in fields], "remaining": remaining}


@app.get("/api/queue/next")
def queue_next(user=Depends(current_user)):
    with connect() as con:
        row = con.execute(
            "UPDATE documents SET status='in_review', locked_by=%s, locked_at=now() "
            "WHERE id = (SELECT id FROM documents "
            "  WHERE (status='pending' OR (status='in_review' "
            "        AND locked_at < now() - make_interval(mins => %s))) "
            "    AND (assigned_to IS NULL OR assigned_to = %s) "
            "  ORDER BY CASE WHEN assigned_to = %s THEN 0 ELSE 1 END, year, id "
            "  LIMIT 1 FOR UPDATE SKIP LOCKED) "
            "RETURNING id", (user["id"], LOCK_TIMEOUT_MIN, user["id"], user["id"])).fetchone()
        if not row:
            con.commit()
            return {"document": None, "remaining": 0}
        con.commit()
        out = doc_payload(con, row["id"])
    out["editable"] = True
    return out


@app.get("/api/documents/{doc_id}/view")
def view_doc(doc_id: int, user=Depends(current_user)):
    """Read-only fetch — does not claim or lock the document."""
    with connect() as con:
        out = doc_payload(con, doc_id)
    out["editable"] = False
    return out


@app.post("/api/documents/{doc_id}/claim")
def claim(doc_id: int, user=Depends(current_user)):
    with connect() as con:
        doc = con.execute(
            "SELECT d.*, u.full_name locked_name, a.full_name assigned_name "
            "FROM documents d "
            "LEFT JOIN users u ON u.id = d.locked_by "
            "LEFT JOIN users a ON a.id = d.assigned_to "
            "WHERE d.id = %s FOR UPDATE OF d", (doc_id,)).fetchone()
        if not doc:
            raise HTTPException(404, "Document not found")
        editable = False
        if doc["status"] in ("validated", "reviewed"):
            pass  # read-only
        elif doc["status"] == "flagged":
            # A flagged deed can be re-opened to correct it. Only the assigned
            # expert (or admin) may do so; opening takes the lock but leaves the
            # flag until an explicit Unflag/Approve/Skip action.
            if user["role"] == "admin" or doc["assigned_to"] == user["id"] or not doc["assigned_to"]:
                con.execute(
                    "UPDATE documents SET locked_by=%s, locked_at=now() WHERE id=%s",
                    (user["id"], doc_id))
                editable = True
            else:
                raise HTTPException(409, f"Assigned to {doc['assigned_name']}")
        elif doc["status"] == "in_monitor_review":
            if user["role"] in ("monitor", "admin"):
                con.execute(
                    "UPDATE documents SET locked_by=%s, locked_at=now() WHERE id=%s",
                    (user["id"], doc_id))
                editable = True
            else:
                raise HTTPException(409, "This deed is under monitor review")
        elif doc["status"] == "in_review" and doc["locked_by"] == user["id"]:
            editable = True
        elif doc["status"] == "in_review" and not con.execute(
                "SELECT 1 FROM documents WHERE id=%s "
                "AND locked_at < now() - make_interval(mins => %s)",
                (doc_id, LOCK_TIMEOUT_MIN)).fetchone():
            raise HTTPException(409, f"Being reviewed by {doc['locked_name']}")
        elif (doc["assigned_to"] and doc["assigned_to"] != user["id"]
              and user["role"] != "admin"):
            raise HTTPException(409, f"Assigned to {doc['assigned_name']}")
        else:  # pending, or expired lock
            # Take the lock but DO NOT change the status yet. Status only moves
            # to in_review when the user actually edits a field, and to
            # validated/flagged/reviewed on an explicit button. This means
            # simply opening then going back leaves the deed untouched.
            con.execute(
                "UPDATE documents SET locked_by=%s, locked_at=now() WHERE id=%s",
                (user["id"], doc_id))
            editable = True
        con.commit()
        out = doc_payload(con, doc_id)
    out["editable"] = editable
    return out


@app.get("/api/documents/{doc_id}")
def get_document(doc_id: int, user=Depends(current_user)):
    with connect() as con:
        return doc_payload(con, doc_id)


@app.get("/api/transliterate")
def transliterate(text: str, user=Depends(current_user)):
    """English (phonetic) -> Odia via Google Input Tools. Proxied server-side
    so the browser needs no extension and no cross-origin access."""
    import urllib.parse
    import urllib.request
    text = text.strip()
    if not text or len(text) > 80:
        return {"suggestions": []}
    url = "https://inputtools.google.com/request?" + urllib.parse.urlencode(
        {"itc": "or-t-i0-und", "num": 4, "text": text})
    try:
        r = json.loads(urllib.request.urlopen(url, timeout=5).read())
        if r[0] == "SUCCESS" and r[1]:
            return {"suggestions": r[1][0][1]}
    except Exception:
        pass
    return {"suggestions": []}


@app.get("/api/documents/{doc_id}/pdf")
def get_pdf(doc_id: int, user=Depends(current_user)):
    with connect() as con:
        doc = con.execute("SELECT deed_number, pdf_file FROM documents WHERE id = %s",
                          (doc_id,)).fetchone()
    if not doc or not doc["pdf_file"]:
        raise HTTPException(404, "No scan attached to this deed")
    path = Path("static/scans") / doc["pdf_file"]
    if not path.exists():
        # Only ever fetches a pre-made <reg_no>.pdf (a plain download, cheap).
        # Deeds with no pre-made PDF are served via /pages + /page/{n} below
        # instead — we no longer stitch a PDF from raw page images here at
        # all, which is what used to spike memory on this route.
        import gcs_store
        if gcs_store.enabled():
            try:
                fetched = gcs_store.fetch_pdf(doc["deed_number"])
                if fetched:
                    path = fetched
            except Exception as e:
                raise HTTPException(502, f"Could not fetch scan from GCS: {e}")
    if not path.exists():
        raise HTTPException(404, "Scan file missing")
    return FileResponse(path, media_type="application/pdf")


@app.get("/api/documents/{doc_id}/debug-meta")
def debug_meta(doc_id: int, user=Depends(current_user)):
    """Inspect a document's raw src_meta (the catch-all JSON blob captured
    from the source grounding data at ingest time) plus whether the Book 1
    detection logic matches it. Exists so this can be checked from any
    logged-in browser session — no Render Shell access needed. Call from
    the browser console while a document is open:
        api(`/documents/${DOC.document.id}/debug-meta`).then(console.log)
    """
    with connect() as con:
        doc = con.execute(
            "SELECT deed_number, deed_type, src_meta FROM documents WHERE id = %s",
            (doc_id,)).fetchone()
    if not doc:
        raise HTTPException(404, "Document not found")
    from ingest_json import _is_book1
    src_meta = doc["src_meta"] or {}
    book_label = src_meta.get("book_label") if isinstance(src_meta, dict) else None
    return {
        "deed_number": doc["deed_number"],
        "deed_type": doc["deed_type"],
        "src_meta": src_meta,
        "book_label_value": book_label,
        "is_book1_match": _is_book1(book_label, doc["deed_type"]),
    }


@app.get("/api/debug/book-labels")
def debug_book_labels(user=Depends(current_user)):
    """List every distinct book_label AND deed_type value actually present
    across the dataset, with counts — so the Book 1 classification mapping
    can be built from the REAL vocabulary instead of an assumption. Call
    from the browser console (no need to have any particular document
    open):
        api('/debug/book-labels').then(console.log)
    """
    with connect() as con:
        book_labels = con.execute(
            "SELECT src_meta->>'book_label' AS value, COUNT(*) AS n "
            "FROM documents GROUP BY 1 ORDER BY n DESC").fetchall()
        deed_types = con.execute(
            "SELECT deed_type AS value, COUNT(*) AS n "
            "FROM documents GROUP BY 1 ORDER BY n DESC").fetchall()
    return {
        "book_label_values": [dict(r) for r in book_labels],
        "deed_type_values": [dict(r) for r in deed_types],
    }


@app.get("/api/documents/{doc_id}/pages")
def get_pages(doc_id: int, user=Depends(current_user)):
    """Tell the frontend how to show this deed's scan: a single pre-made PDF
    (mode 'pdf', served as-is via /pdf above — cheap, just a file download),
    or a sequence of individually-served page images (mode 'images') for
    deeds that have no pre-made PDF. We deliberately never build a PDF from
    raw page images anymore — the browser just lays out the images in order,
    which is simpler and means the server never holds more than one page's
    bytes in memory at a time."""
    with connect() as con:
        doc = con.execute("SELECT deed_number, pdf_file FROM documents WHERE id = %s",
                          (doc_id,)).fetchone()
    if not doc:
        raise HTTPException(404, "Document not found")
    if doc["pdf_file"]:
        local = Path("static/scans") / doc["pdf_file"]
        if local.exists():
            return {"mode": "pdf"}
        import gcs_store
        try:
            if gcs_store.enabled() and gcs_store.premade_pdf_exists(doc["deed_number"]):
                return {"mode": "pdf"}
        except Exception as e:
            raise HTTPException(502, f"Could not check for pre-made PDF: {e}")
    import gcs_store
    if gcs_store.enabled():
        try:
            entry = gcs_store.pages_entry(doc["deed_number"])
        except Exception as e:
            raise HTTPException(502, f"Could not look up scan pages: {e}")
        if entry:
            # Actual page numbers, in order — NOT assumed to be 1..count.
            # The raw dataset's page numbers aren't guaranteed to start at 1
            # or be contiguous, so the frontend must use these exact values
            # rather than generating its own 1..count sequence (which was
            # producing 404s — and blank space, not a visible error — for
            # any deed whose numbering didn't happen to match).
            page_nums = [pg for pg, _prefix, _rel in entry["pages"]]
            return {"mode": "images", "pages": page_nums}
    raise HTTPException(404, "No scan attached to this deed")


@app.get("/api/documents/{doc_id}/page/{page_num}")
def get_page_image(doc_id: int, page_num: int, user=Depends(current_user)):
    """Serve a single raw scan page's image bytes directly — no PDF
    assembly, no decode/re-encode. Cached to local disk per page so a
    reopened deed doesn't re-hit GCS."""
    with connect() as con:
        doc = con.execute("SELECT deed_number FROM documents WHERE id = %s",
                          (doc_id,)).fetchone()
    if not doc:
        raise HTTPException(404, "Document not found")
    import gcs_store
    if not gcs_store.enabled():
        raise HTTPException(404, "No scan source configured")
    try:
        data, content_type = gcs_store.fetch_page_image(doc["deed_number"], page_num)
    except Exception as e:
        raise HTTPException(502, f"Could not fetch page image: {e}")
    if data is None:
        raise HTTPException(404, "Page not found")
    return Response(content=data, media_type=content_type)


# ---------- full text (populated from Akshar's JSON on ingest; edited here) ----------
# NOTE: digitization itself is done in Akshar, not here. The portal only
# stores the extracted full text and lets an expert correct it. When the
# Akshar JSON format is finalised, the ingest step will populate
# documents.digitized_text (and the structured metadata fields); this API
# just serves and saves that text.


@app.get("/api/documents/{doc_id}/digitized")
def get_digitized(doc_id: int, user=Depends(current_user)):
    with connect() as con:
        d = con.execute("SELECT digitized_text, digitized_status FROM documents WHERE id=%s",
                        (doc_id,)).fetchone()
    if not d:
        raise HTTPException(404, "Document not found")
    return {"status": d["digitized_status"], "text": d["digitized_text"] or ""}


class DigitizedText(BaseModel):
    text: str


@app.put("/api/documents/{doc_id}/digitized")
def save_digitized(doc_id: int, body: DigitizedText, user=Depends(current_user)):
    with connect() as con:
        if not con.execute("SELECT 1 FROM documents WHERE id=%s", (doc_id,)).fetchone():
            raise HTTPException(404, "Document not found")
        con.execute("UPDATE documents SET digitized_text=%s, digitized_status='corrected', "
                    "last_edited_by=%s, last_edited_at=now() WHERE id=%s",
                    (body.text, user["id"], doc_id))
        con.execute("INSERT INTO edit_log (document_id, action, user_id) "
                    "VALUES (%s,'fulltext_edit',%s)", (doc_id, user["id"]))
        con.commit()
    return {"ok": True}


# ---------- search ----------

PARTY_NAME_SQL = ("(SELECT string_agg(current_value, E'\\n' ORDER BY position) FROM fields f "
                  "WHERE f.document_id = d.id AND f.label = 'Name' AND f.section LIKE %s)")


@app.get("/api/search")
def search(q: str = "", field: str = "deed_number", status: str = "",
           assigned: str = "", sort_by: str = "", sort_order: str = "asc",
           page: int = 1, per_page: int = 10, user=Depends(current_user)):
    where, params = [], []
    q = q.strip()
    if q:
        if field == "deed_number":
            where.append("d.deed_number ILIKE %s"); params.append(f"%{q}%")
        elif field == "party_name":
            where.append("EXISTS (SELECT 1 FROM fields f WHERE f.document_id = d.id "
                         "AND f.label IN ('Name','Relation name') AND f.current_value ILIKE %s)")
            params.append(f"%{q}%")
        elif field == "year" and q.isdigit():
            where.append("d.year = %s"); params.append(int(q))
        elif field == "book_no" and q.isdigit():
            where.append("d.book_no = %s"); params.append(int(q))
    if status:
        where.append("d.status = %s"); params.append(status)
    # assigned-to filter (admin/monitor views): expert id, or 'none' = unassigned
    if assigned and user["role"] in ("admin", "monitor"):
        if assigned == "none":
            where.append("d.assigned_to IS NULL")
        elif assigned.isdigit():
            where.append("d.assigned_to = %s"); params.append(int(assigned))
    # Role-scoped visibility:
    #  - experts see only deeds assigned to them
    #  - monitors see the review pool + anything they've reviewed
    #  - admins see everything
    if user["role"] == "expert":
        where.append("d.assigned_to = %s"); params.append(user["id"])
        # Experts never see deeds that have gone to the monitor.
        where.append("d.status NOT IN ('in_monitor_review','reviewed')")
    elif user["role"] == "monitor":
        where.append("(d.status IN ('in_monitor_review','reviewed') OR d.review_by = %s)")
        params.append(user["id"])
    wsql = ("WHERE " + " AND ".join(where)) if where else ""
    order_col = {"year": "d.year", "deed_number": "d.deed_number",
                 "status": "d.status", "book_no": "d.book_no",
                 "last_edited": "d.last_edited_at"}.get(sort_by, "d.id")
    order = "DESC" if sort_order == "desc" else "ASC"
    # Experts get a fixed priority sort by default (what to work on first):
    # in_review (mid-way) -> pending -> flagged -> validated.
    expert_priority = (user["role"] == "expert" and sort_by in ("", "id", None))
    if expert_priority:
        order_sql = ("CASE d.status WHEN 'in_review' THEN 0 WHEN 'pending' THEN 1 "
                     "WHEN 'flagged' THEN 2 WHEN 'validated' THEN 3 ELSE 4 END, d.year, d.id")
    else:
        order_sql = f"{order_col} {order}, d.id"
    per_page = max(1, min(per_page, 100))
    offset = (max(page, 1) - 1) * per_page

    with connect() as con:
        total = con.execute(
            f"SELECT COUNT(*) c FROM documents d {wsql}", params).fetchone()["c"]
        rows = con.execute(
            f"SELECT d.id, d.deed_number, d.deed_type, d.year, d.book_no, d.sr_office, "
            f"d.status, d.flag_reason, (d.pdf_file IS NOT NULL) has_pdf, u.full_name locked_name, "
            f"a.full_name assigned_name, a.id assigned_id, "
            f"le.full_name last_edited_name, to_char(d.last_edited_at, 'DD Mon HH24:MI') last_edited_at, "
            f"{PARTY_NAME_SQL} first_party, {PARTY_NAME_SQL} second_party "
            f"FROM documents d LEFT JOIN users u ON u.id = d.locked_by "
            f"LEFT JOIN users a ON a.id = d.assigned_to "
            f"LEFT JOIN users le ON le.id = d.last_edited_by "
            f"{wsql} ORDER BY {order_sql} LIMIT %s OFFSET %s",
            ["First party%", "Second party%"] + params + [per_page, offset]).fetchall()
    return {"total": total, "page": page, "per_page": per_page,
            "results": [dict(r) for r in rows]}


# ---------- edits & lifecycle ----------

class FieldPatch(BaseModel):
    value: str
    odia: str | None = None


def require_lock(con, doc_id, user):
    doc = con.execute("SELECT * FROM documents WHERE id = %s FOR UPDATE",
                      (doc_id,)).fetchone()
    if not doc:
        raise HTTPException(404, "Document not found")
    # The user must hold the lock. Editable statuses are pending / in_review
    # (expert) and in_monitor_review (monitor/admin). Opening a deed takes the
    # lock but leaves status pending; the first edit promotes it to in_review.
    holds = doc["locked_by"] == user["id"]
    ok = holds and (
        doc["status"] in ("pending", "in_review", "flagged") or
        (doc["status"] == "in_monitor_review" and user["role"] in ("monitor", "admin")))
    if not ok:
        raise HTTPException(409, "Document is not checked out to you — open it first")
    return doc


@app.patch("/api/documents/{doc_id}/fields/{field_id}")
def patch_field(doc_id: int, field_id: int, body: FieldPatch, user=Depends(current_user)):
    with connect() as con:
        require_lock(con, doc_id, user)
        f = con.execute("SELECT * FROM fields WHERE id = %s AND document_id = %s",
                        (field_id, doc_id)).fetchone()
        if not f:
            raise HTTPException(404, "Field not found")
        changed = False
        if f["current_value"] != body.value:
            con.execute("UPDATE fields SET current_value = %s WHERE id = %s",
                        (body.value, field_id))
            con.execute(
                "INSERT INTO edit_log (document_id, field_id, old_value, new_value, action, user_id) "
                "VALUES (%s,%s,%s,%s,'edit',%s)",
                (doc_id, field_id, f["current_value"], body.value, user["id"]))
            changed = True
        if body.odia is not None and f["odia_value"] != body.odia:
            con.execute("UPDATE fields SET odia_value = %s WHERE id = %s",
                        (body.odia, field_id))
            con.execute(
                "INSERT INTO edit_log (document_id, field_id, old_value, new_value, action, user_id) "
                "VALUES (%s,%s,%s,%s,'edit_odia',%s)",
                (doc_id, field_id, f["odia_value"], body.odia, user["id"]))
            changed = True
        if changed:
            # first edit on a still-pending deed promotes it to in_review
            con.execute(
                "UPDATE documents SET status = CASE WHEN status='pending' THEN 'in_review' "
                "ELSE status END, locked_at = now(), "
                "last_edited_by = %s, last_edited_at = now() WHERE id = %s",
                (user["id"], doc_id))
        con.commit()
    return {"ok": True}


@app.get("/api/documents/{doc_id}/history")
def field_history(doc_id: int, user=Depends(current_user)):
    """Every field change on this deed: field, old, new, who, when.
    Powers the admin 'who changed what' view."""
    with connect() as con:
        rows = con.execute(
            "SELECT e.ts, e.action, e.old_value, e.new_value, "
            "u.full_name AS by_name, f.section, f.label "
            "FROM edit_log e JOIN users u ON u.id = e.user_id "
            "LEFT JOIN fields f ON f.id = e.field_id "
            "WHERE e.document_id = %s ORDER BY e.ts DESC", (doc_id,)).fetchall()
    out = []
    for r in rows:
        r = dict(r)
        r["ts"] = str(r["ts"])[:19]
        out.append(r)
    return out


@app.post("/api/documents/{doc_id}/approve")
def approve(doc_id: int, user=Depends(current_user)):
    with connect() as con:
        require_lock(con, doc_id, user)
        con.execute(
            "UPDATE documents SET status='validated', validated_by=%s, validated_at=now(), "
            "locked_by=NULL, locked_at=NULL WHERE id=%s", (user["id"], doc_id))
        con.execute("INSERT INTO edit_log (document_id, action, user_id) "
                    "VALUES (%s,'approve',%s)", (doc_id, user["id"]))
        # Sample ~1% of each day's validated records for monitor review.
        # Deterministic boundary test: when today's validated count crosses a
        # new multiple of 100, route THIS deed to the monitor review pool.
        today_count = con.execute(
            "SELECT COUNT(*) c FROM documents "
            "WHERE status IN ('validated','reviewed','in_monitor_review') "
            "AND validated_at::date = now()::date").fetchone()["c"]
        # 1st, 101st, 201st ... validated deed of the day gets sampled (>=1%)
        if today_count % 100 == 1:
            con.execute(
                "UPDATE documents SET status='in_monitor_review', "
                "sent_to_review_on=now()::date, sent_to_review_by=%s, sent_reason='sample' "
                "WHERE id=%s", (user["id"], doc_id))
            con.execute("INSERT INTO edit_log (document_id, action, user_id) "
                        "VALUES (%s,'sent_to_review',%s)", (doc_id, user["id"]))
        con.commit()
    return {"ok": True}


class FlagIn(BaseModel):
    reason: str = ""


@app.post("/api/documents/{doc_id}/flag")
def flag(doc_id: int, body: FlagIn, user=Depends(current_user)):
    with connect() as con:
        require_lock(con, doc_id, user)
        con.execute(
            "UPDATE documents SET status='flagged', flag_reason=%s, "
            "locked_by=NULL, locked_at=NULL WHERE id=%s", (body.reason, doc_id))
        con.execute("INSERT INTO edit_log (document_id, action, new_value, user_id) "
                    "VALUES (%s,'flag',%s,%s)", (doc_id, body.reason, user["id"]))
        con.commit()
    return {"ok": True}


@app.post("/api/documents/{doc_id}/unflag")
def unflag(doc_id: int, user=Depends(current_user)):
    """Clear a flag so the deed can be corrected. Returns it to in_review,
    locked to the current user so they can keep editing."""
    with connect() as con:
        require_lock(con, doc_id, user)
        con.execute(
            "UPDATE documents SET status='in_review', flag_reason=NULL, "
            "locked_by=%s, locked_at=now() WHERE id=%s", (user["id"], doc_id))
        con.execute("INSERT INTO edit_log (document_id, action, user_id) "
                    "VALUES (%s,'unflag',%s)", (doc_id, user["id"]))
        con.commit()
        out = doc_payload(con, doc_id)
    out["editable"] = True
    return out


@app.post("/api/documents/{doc_id}/skip")
def skip(doc_id: int, user=Depends(current_user)):
    with connect() as con:
        require_lock(con, doc_id, user)
        if user["role"] == "expert":
            # An expert skipping a deed (can't complete it) sends it to the
            # monitor review pool rather than back to the general queue.
            con.execute(
                "UPDATE documents SET status='in_monitor_review', locked_by=NULL, "
                "locked_at=NULL, sent_to_review_on=now()::date, sent_to_review_by=%s, "
                "sent_reason='skip' WHERE id=%s", (user["id"], doc_id))
            con.execute("INSERT INTO edit_log (document_id, action, user_id) "
                        "VALUES (%s,'skip_to_monitor',%s)", (doc_id, user["id"]))
        else:
            con.execute("UPDATE documents SET status='pending', locked_by=NULL, "
                        "locked_at=NULL WHERE id=%s", (doc_id,))
        con.commit()
    return {"ok": True}


@app.get("/api/monitor/queue")
def monitor_queue(user=Depends(require_monitor)):
    """Deeds waiting for monitor review (sampled 1% + expert skips)."""
    with connect() as con:
        rows = con.execute(
            "SELECT d.id, d.deed_number, d.deed_type, d.year, "
            "vu.full_name validated_by_name, su.full_name sent_by_name, "
            "d.sent_reason, d.sent_to_review_on, d.status "
            "FROM documents d "
            "LEFT JOIN users vu ON vu.id = d.validated_by "
            "LEFT JOIN users su ON su.id = d.sent_to_review_by "
            "WHERE d.status = 'in_monitor_review' ORDER BY d.sent_to_review_on, d.id").fetchall()
    return [dict(r, sent_to_review_on=str(r["sent_to_review_on"]) if r["sent_to_review_on"] else None)
            for r in rows]


@app.get("/api/monitor/next")
def monitor_next(user=Depends(require_monitor)):
    """Claim the next deed awaiting monitor review and return it editable."""
    with connect() as con:
        row = con.execute(
            "UPDATE documents SET locked_by=%s, locked_at=now() "
            "WHERE id = (SELECT id FROM documents WHERE status='in_monitor_review' "
            "  ORDER BY sent_to_review_on, id LIMIT 1 FOR UPDATE SKIP LOCKED) "
            "RETURNING id", (user["id"],)).fetchone()
        if not row:
            con.commit()
            return {"document": None, "remaining": 0}
        remaining = con.execute(
            "SELECT COUNT(*) c FROM documents WHERE status='in_monitor_review'").fetchone()["c"]
        con.commit()
        out = doc_payload(con, row["id"])
    out["editable"] = True
    out["remaining"] = remaining
    return out


@app.post("/api/documents/{doc_id}/unvalidate")
def unvalidate(doc_id: int, user=Depends(current_user)):
    """An expert can un-validate a deed THEY validated (and that's still
    assigned to them) to edit it again — e.g. accidental click or a late fix.
    Returns it to in_review, locked to them."""
    with connect() as con:
        doc = con.execute("SELECT status, validated_by, assigned_to FROM documents "
                          "WHERE id=%s FOR UPDATE", (doc_id,)).fetchone()
        if not doc:
            raise HTTPException(404, "Document not found")
        if doc["status"] != "validated":
            raise HTTPException(400, "Only a validated deed can be un-validated")
        # expert may only reopen their own validation, still assigned to them;
        # admins can always
        if user["role"] != "admin":
            if doc["validated_by"] != user["id"] or doc["assigned_to"] != user["id"]:
                raise HTTPException(403, "You can only re-open deeds you validated")
        con.execute(
            "UPDATE documents SET status='in_review', validated_by=NULL, validated_at=NULL, "
            "locked_by=%s, locked_at=now() WHERE id=%s", (user["id"], doc_id))
        con.execute("INSERT INTO edit_log (document_id, action, user_id) "
                    "VALUES (%s,'unvalidate',%s)", (doc_id, user["id"]))
        con.commit()
        out = doc_payload(con, doc_id)
    out["editable"] = True
    return out


@app.post("/api/documents/{doc_id}/review")
def submit_review(doc_id: int, user=Depends(require_monitor)):
    """Monitor marks a deed reviewed. review_corrected reflects whether the
    monitor changed any field during this review (tracked via edit_log)."""
    with connect() as con:
        doc = con.execute("SELECT status FROM documents WHERE id=%s", (doc_id,)).fetchone()
        if not doc or doc["status"] not in ("in_monitor_review", "validated"):
            raise HTTPException(400, "This deed is not awaiting review")
        # did the monitor correct anything during review?
        corrected = con.execute(
            "SELECT COUNT(*) c FROM edit_log WHERE document_id=%s AND user_id=%s "
            "AND action='edit'", (doc_id, user["id"])).fetchone()["c"] > 0
        con.execute(
            "UPDATE documents SET status='reviewed', review_by=%s, reviewed_at=now(), "
            "review_corrected=%s, locked_by=NULL, locked_at=NULL WHERE id=%s",
            (user["id"], corrected, doc_id))
        con.execute("INSERT INTO edit_log (document_id, action, user_id) "
                    "VALUES (%s,'reviewed',%s)", (doc_id, user["id"]))
        con.commit()
    return {"ok": True, "corrected": corrected}


# ---------- admin ----------

class ReopenIn(BaseModel):
    assign_to: int | None = None


@app.post("/api/documents/{doc_id}/reopen")
def reopen(doc_id: int, body: ReopenIn, user=Depends(require_admin)):
    """Admin-only: move a validated/flagged document back to pending so it can
    be edited again, optionally reassigning it for re-checking. History kept."""
    with connect() as con:
        doc = con.execute("SELECT status FROM documents WHERE id=%s", (doc_id,)).fetchone()
        if not doc:
            raise HTTPException(404, "Document not found")
        assign_sql, assign_params = "", []
        if body.assign_to is not None:
            if not con.execute("SELECT 1 FROM users WHERE id=%s AND role='expert'",
                               (body.assign_to,)).fetchone():
                raise HTTPException(404, "Expert not found")
            assign_sql = ", assigned_to=%s"; assign_params = [body.assign_to]
        con.execute(
            "UPDATE documents SET status='pending', validated_by=NULL, validated_at=NULL, "
            "flag_reason=NULL, locked_by=NULL, locked_at=NULL" + assign_sql +
            " WHERE id=%s", assign_params + [doc_id])
        con.execute("INSERT INTO edit_log (document_id, action, new_value, user_id) "
                    "VALUES (%s,'reopen',%s,%s)",
                    (doc_id, f"was {doc['status']}", user["id"]))
        con.commit()
    return {"ok": True}


class AssignOneIn(BaseModel):
    expert_id: int | None = None  # None = unassign


@app.post("/api/documents/{doc_id}/assign")
def assign_one(doc_id: int, body: AssignOneIn, user=Depends(require_admin)):
    """Assign (or unassign) a single specific deed to an expert."""
    with connect() as con:
        if body.expert_id is not None and not con.execute(
                "SELECT 1 FROM users WHERE id=%s AND role='expert'",
                (body.expert_id,)).fetchone():
            raise HTTPException(404, "Expert not found")
        con.execute("UPDATE documents SET assigned_to=%s WHERE id=%s",
                    (body.expert_id, doc_id))
        con.commit()
    return {"ok": True}


class AssignIn(BaseModel):
    expert_id: int
    count: int = 10
    book_no: int | None = None
    year: int | None = None


@app.post("/api/admin/assign")
def assign_work(body: AssignIn, user=Depends(require_admin)):
    """Assign up to `count` unassigned pending deeds (oldest first) to an expert,
    optionally filtered by book and/or year."""
    if body.count < 1 or body.count > 10000:
        raise HTTPException(400, "Count must be between 1 and 10000")
    filters, params = "", []
    if body.book_no is not None:
        filters += " AND book_no = %s"; params.append(body.book_no)
    if body.year is not None:
        filters += " AND year = %s"; params.append(body.year)
    with connect() as con:
        if not con.execute("SELECT 1 FROM users WHERE id=%s", (body.expert_id,)).fetchone():
            raise HTTPException(404, "Expert not found")
        rows = con.execute(
            f"UPDATE documents SET assigned_to = %s WHERE id IN ("
            f"  SELECT id FROM documents WHERE status='pending' AND assigned_to IS NULL"
            f"  {filters} ORDER BY year, id LIMIT %s FOR UPDATE SKIP LOCKED) "
            f"RETURNING id", [body.expert_id] + params + [body.count]).fetchall()
        con.commit()
    return {"assigned": len(rows)}


class UnassignIn(BaseModel):
    expert_id: int


@app.post("/api/admin/unassign")
def unassign_work(body: UnassignIn, user=Depends(require_admin)):
    """Return an expert's not-yet-started (pending) assignments to the pool."""
    with connect() as con:
        rows = con.execute(
            "UPDATE documents SET assigned_to = NULL "
            "WHERE assigned_to = %s AND status = 'pending' RETURNING id",
            (body.expert_id,)).fetchall()
        con.commit()
    return {"unassigned": len(rows)}


@app.get("/api/monitor/dashboard")
def monitor_dashboard(user=Depends(require_monitor)):
    """Analysis for the monitor: pending review count, reviewed totals,
    correction rate, and a per-expert breakdown of what was reviewed."""
    with connect() as con:
        awaiting = con.execute(
            "SELECT COUNT(*) c FROM documents WHERE status='in_monitor_review'").fetchone()["c"]
        reviewed = con.execute(
            "SELECT COUNT(*) c FROM documents WHERE status='reviewed'").fetchone()["c"]
        corrected = con.execute(
            "SELECT COUNT(*) c FROM documents WHERE status='reviewed' AND review_corrected").fetchone()["c"]
        today = con.execute(
            "SELECT COUNT(*) c FROM documents WHERE status='reviewed' "
            "AND reviewed_at::date = now()::date").fetchone()["c"]
        # per-expert: of this expert's validated deeds that reached review,
        # how many the monitor corrected
        per_expert = con.execute(
            "SELECT vu.full_name expert, "
            "COUNT(*) FILTER (WHERE d.status='reviewed') reviewed, "
            "COUNT(*) FILTER (WHERE d.status='reviewed' AND d.review_corrected) corrected, "
            "COUNT(*) FILTER (WHERE d.status='in_monitor_review') awaiting "
            "FROM documents d JOIN users vu ON vu.id = d.validated_by "
            "WHERE d.status IN ('in_monitor_review','reviewed') "
            "GROUP BY vu.full_name ORDER BY reviewed DESC").fetchall()
    corr_rate = round(corrected / reviewed * 100, 1) if reviewed else 0.0
    out_experts = []
    for e in per_expert:
        e = dict(e)
        e["error_rate"] = round(e["corrected"] / e["reviewed"] * 100, 1) if e["reviewed"] else 0.0
        out_experts.append(e)
    return {"awaiting": awaiting, "reviewed": reviewed, "corrected": corrected,
            "correction_rate": corr_rate, "reviewed_today": today, "experts": out_experts}


@app.get("/api/admin/dashboard")
def admin_dashboard(user=Depends(require_admin)):
    with connect() as con:
        by_status = {r["status"]: r["c"] for r in con.execute(
            "SELECT status, COUNT(*) c FROM documents GROUP BY status")}
        total = sum(by_status.values())
        experts = [dict(r) for r in con.execute("""
            SELECT u.id, u.full_name,
              (SELECT COUNT(*) FROM documents d WHERE d.assigned_to=u.id
                 AND d.status IN ('pending','in_review'))            assigned_remaining,
              (SELECT COUNT(*) FROM documents d WHERE d.assigned_to=u.id) assigned_total,
              (SELECT COUNT(*) FROM documents d WHERE d.validated_by=u.id) validated,
              (SELECT COUNT(*) FROM documents d WHERE d.validated_by=u.id
                 AND d.validated_at::date = current_date)             validated_today,
              (SELECT COUNT(*) FROM documents d WHERE d.validated_by=u.id
                 AND d.validated_at > now() - interval '7 days')      validated_7d,
              (SELECT COUNT(*) FROM edit_log e WHERE e.user_id=u.id
                 AND e.action='edit')                                 corrections,
              (SELECT COUNT(*) FROM edit_log e WHERE e.user_id=u.id
                 AND e.action='flag')                                 flags,
              (SELECT to_char(max(e.ts), 'DD Mon HH24:MI')
                 FROM edit_log e WHERE e.user_id=u.id)                last_active
            FROM users u WHERE u.role='expert' ORDER BY validated DESC, u.id"""
        ).fetchall()]
        books = [dict(r) for r in con.execute(
            "SELECT book_no, COUNT(*) total, "
            "COUNT(*) FILTER (WHERE status='validated') validated, "
            "COUNT(*) FILTER (WHERE status='flagged') flagged "
            "FROM documents GROUP BY book_no ORDER BY book_no").fetchall()]
        daily = [dict(r) for r in con.execute(
            "SELECT to_char(validated_at::date, 'DD Mon') AS day, COUNT(*) c "
            "FROM documents WHERE status='validated' "
            "AND validated_at > now() - interval '14 days' "
            "GROUP BY validated_at::date ORDER BY validated_at::date").fetchall()]
        flagged = [dict(r) for r in con.execute(
            "SELECT d.deed_number, d.flag_reason, "
            "(SELECT u.full_name FROM edit_log e JOIN users u ON u.id=e.user_id "
            " WHERE e.document_id=d.id AND e.action='flag' "
            " ORDER BY e.ts DESC LIMIT 1) flagged_by "
            "FROM documents d WHERE d.status='flagged' ORDER BY d.id").fetchall()]
        edits = con.execute(
            "SELECT COUNT(*) c FROM edit_log WHERE action='edit'").fetchone()["c"]
        unassigned = con.execute(
            "SELECT COUNT(*) c FROM documents WHERE status='pending' "
            "AND assigned_to IS NULL").fetchone()["c"]
    return {"by_status": by_status, "total": total, "experts": experts,
            "books": books, "daily": daily, "flagged": flagged,
            "total_edits": edits, "unassigned_pending": unassigned}


class NewUser(BaseModel):
    username: str
    password: str
    full_name: str
    role: str = "expert"


@app.get("/api/users")
def list_users(user=Depends(require_admin)):
    with connect() as con:
        rows = con.execute(
            "SELECT u.id, u.username, u.full_name, u.role, "
            "(SELECT COUNT(*) FROM documents d WHERE d.validated_by = u.id) validated "
            "FROM users u ORDER BY u.id").fetchall()
    return [dict(r) for r in rows]


@app.post("/api/users")
def create_user(body: NewUser, user=Depends(require_admin)):
    if body.role not in ("expert", "admin", "monitor"):
        raise HTTPException(400, "Role must be expert or admin")
    if len(body.password) < 6:
        raise HTTPException(400, "Password must be at least 6 characters")
    with connect() as con:
        try:
            con.execute(
                "INSERT INTO users (username, password_hash, full_name, role) "
                "VALUES (%s,%s,%s,%s)",
                (body.username.strip(), hash_pw(body.password),
                 body.full_name.strip(), body.role))
            con.commit()
        except Exception:
            raise HTTPException(409, "Username already exists")
    return {"ok": True}


class AdminPasswordReset(BaseModel):
    new_password: str


@app.post("/api/users/{user_id}/reset-password")
def admin_reset_password(user_id: int, body: AdminPasswordReset, user=Depends(require_admin)):
    """Admin-only: set a new password for a locked-out user, without
    needing their old one (unlike the self-service /change-password,
    which requires it). Only the password changes — username, role,
    document assignments, validation history, and everything else about
    the account is untouched. Existing sessions for that user are ended,
    so the reset takes effect immediately rather than leaving an old
    logged-in session usable until it expires on its own."""
    if len(body.new_password) < 6:
        raise HTTPException(400, "Password must be at least 6 characters")
    with connect() as con:
        target = con.execute("SELECT id FROM users WHERE id=%s", (user_id,)).fetchone()
        if not target:
            raise HTTPException(404, "User not found")
        con.execute("UPDATE users SET password_hash=%s WHERE id=%s",
                    (hash_pw(body.new_password), user_id))
        con.execute("DELETE FROM sessions WHERE user_id=%s", (user_id,))
        con.commit()
    return {"ok": True}


@app.delete("/api/users/{user_id}")
def delete_user(user_id: int, user=Depends(require_admin)):
    """Delete an account. Guards: can't delete yourself or the last admin.
    Their live assignments/locks are released; edit-log history is cleared."""
    if user_id == user["id"]:
        raise HTTPException(400, "You cannot delete your own account")
    with connect() as con:
        target = con.execute("SELECT role FROM users WHERE id=%s", (user_id,)).fetchone()
        if not target:
            raise HTTPException(404, "User not found")
        if target["role"] == "admin":
            admins = con.execute("SELECT COUNT(*) c FROM users WHERE role='admin'").fetchone()["c"]
            if admins <= 1:
                raise HTTPException(400, "Cannot delete the last admin account")
        con.execute("UPDATE documents SET assigned_to=NULL WHERE assigned_to=%s", (user_id,))
        con.execute("UPDATE documents SET status='pending', locked_by=NULL, locked_at=NULL "
                    "WHERE locked_by=%s AND status='in_review'", (user_id,))
        con.execute("UPDATE documents SET validated_by=NULL WHERE validated_by=%s", (user_id,))
        con.execute("UPDATE documents SET last_edited_by=NULL WHERE last_edited_by=%s", (user_id,))
        con.execute("DELETE FROM sessions WHERE user_id=%s", (user_id,))
        con.execute("DELETE FROM edit_log WHERE user_id=%s", (user_id,))
        con.execute("DELETE FROM users WHERE id=%s", (user_id,))
        con.commit()
    return {"ok": True}


@app.get("/api/stats")
def stats(user=Depends(current_user)):
    with connect() as con:
        by_status = {r["status"]: r["c"] for r in con.execute(
            "SELECT status, COUNT(*) c FROM documents GROUP BY status")}
        per_expert = [dict(r) for r in con.execute(
            "SELECT u.full_name, COUNT(*) validated FROM documents d "
            "JOIN users u ON u.id = d.validated_by WHERE d.status='validated' "
            "GROUP BY u.id, u.full_name ORDER BY validated DESC")]
        edits = con.execute(
            "SELECT COUNT(*) c FROM edit_log WHERE action='edit'").fetchone()["c"]
        flagged = [dict(r) for r in con.execute(
            "SELECT deed_number, flag_reason FROM documents WHERE status='flagged'")]
    return {"by_status": by_status, "per_expert": per_expert,
            "total_edits": edits, "flagged": flagged}


@app.get("/api/export")
def export_corrected(user=Depends(require_admin)):
    """Corrected dataset as JSON — the same per-page block structure that was
    ingested (one folder per deed, metadata/page_NNN.json), with each block's
    text replaced by the expert's corrected value. Returned as a ZIP."""
    from export_json import build_export_zip
    buf = build_export_zip()
    return StreamingResponse(
        buf, media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=corrected_deeds_json.zip"})


@app.get("/api/documents/{doc_id}/export")
def export_one(doc_id: int, user=Depends(require_admin)):
    """Corrected JSON for a single deed, as a ZIP of its per-page files."""
    from export_json import build_single_deed_json
    with connect() as con:
        d = con.execute("SELECT deed_number FROM documents WHERE id=%s", (doc_id,)).fetchone()
    if not d:
        raise HTTPException(404, "Document not found")
    buf = build_single_deed_json(d["deed_number"])
    safe = d["deed_number"].replace("/", "_")
    return StreamingResponse(
        buf, media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={safe}_corrected.zip"})


# static frontend last, so /api wins
app.mount("/", StaticFiles(directory="static", html=True), name="static")
