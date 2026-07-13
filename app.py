"""Deed validation backend (PostgreSQL).

Run:  uvicorn app:app --port 8000
Env:  DATABASE_URL=postgresql://deeds:deeds@localhost:5432/deeds
Seed logins (rotate in prod): expert1..expert3 / admin, password sarvam123
"""
import secrets
from pathlib import Path

from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from db import connect, init_db, hash_pw, check_pw, SESSION_HOURS
from export import build_export_workbook

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
    from ingest import find_pdf
    repaired = 0
    for d in missing:
        src = find_pdf("data", d["book_no"], d["reg_no"], d["year"])
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

    # 2) load data if empty
    try:
        con = connect()
        try:
            con.execute("SELECT pg_advisory_lock(424242)")
            n = con.execute("SELECT COUNT(*) c FROM documents").fetchone()["c"]
            if n:
                _ingest_status.update(state="done", documents=n, detail="already loaded")
                _repair_scans(con)
                return
        finally:
            con.close()

        found = sorted(glob.glob("data/**/*.xlsx", recursive=True))
        print(f"[startup] xlsx files found: {found}", flush=True)
        if not found:
            _ingest_status.update(state="error", detail="no .xlsx found under data/")
            print("[startup] ERROR: no Excel file found under data/", flush=True)
            return
        from ingest import ingest as run_ingest
        _ingest_status.update(state="running", detail=found[0])
        print(f"[startup] auto-ingesting {found[0]}", flush=True)
        run_ingest(found[0], str(Path(found[0]).parent))
        with connect() as con:
            n = con.execute("SELECT COUNT(*) c FROM documents").fetchone()["c"]
        _ingest_status.update(state="done", documents=n, detail=found[0])
        print(f"[startup] ingest complete — {n} documents", flush=True)
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
            "xlsx_found": sorted(glob.glob("data/**/*.xlsx", recursive=True)),
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


@app.post("/api/admin/reingest")
def admin_reingest(user=Depends(require_admin)):
    """Manually (re)run ingestion. Safe: existing deeds are skipped."""
    import threading
    threading.Thread(target=_auto_ingest, daemon=True).start()
    return {"ok": True, "message": "Ingestion started — check /api/ingest-status"}


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
        "SELECT id, section, label, ocr_value, current_value, multiline "
        "FROM fields WHERE document_id = %s ORDER BY position", (doc_id,)).fetchall()
    remaining = con.execute(
        "SELECT COUNT(*) c FROM documents WHERE status IN ('pending','in_review')"
    ).fetchone()["c"]
    doc = dict(doc)
    for k in ("locked_at", "validated_at"):
        doc[k] = str(doc[k]) if doc[k] else None
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
        if doc["status"] in ("validated", "flagged"):
            pass  # read-only
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
            con.execute(
                "UPDATE documents SET status='in_review', locked_by=%s, locked_at=now() "
                "WHERE id=%s", (user["id"], doc_id))
            editable = True
        con.commit()
        out = doc_payload(con, doc_id)
    out["editable"] = editable
    return out


@app.get("/api/documents/{doc_id}")
def get_document(doc_id: int, user=Depends(current_user)):
    with connect() as con:
        return doc_payload(con, doc_id)


@app.get("/api/documents/{doc_id}/pdf")
def get_pdf(doc_id: int, user=Depends(current_user)):
    with connect() as con:
        doc = con.execute("SELECT pdf_file FROM documents WHERE id = %s",
                          (doc_id,)).fetchone()
    if not doc or not doc["pdf_file"]:
        raise HTTPException(404, "No scan attached to this deed")
    path = Path("static/scans") / doc["pdf_file"]
    if not path.exists():
        raise HTTPException(404, "Scan file missing on disk")
    return FileResponse(path, media_type="application/pdf")


# ---------- search ----------

PARTY_NAME_SQL = ("(SELECT string_agg(current_value, E'\\n' ORDER BY position) FROM fields f "
                  "WHERE f.document_id = d.id AND f.label = 'Name' AND f.section LIKE %s)")


@app.get("/api/search")
def search(q: str = "", field: str = "deed_number", status: str = "",
           sort_by: str = "year", sort_order: str = "asc",
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
    # Experts only ever see deeds assigned to them; admins see everything.
    if user["role"] != "admin":
        where.append("d.assigned_to = %s"); params.append(user["id"])
    wsql = ("WHERE " + " AND ".join(where)) if where else ""
    order_col = {"year": "d.year", "deed_number": "d.deed_number",
                 "status": "d.status", "book_no": "d.book_no",
                 "last_edited": "d.last_edited_at"}.get(sort_by, "d.id")
    order = "DESC" if sort_order == "desc" else "ASC"
    per_page = max(1, min(per_page, 100))
    offset = (max(page, 1) - 1) * per_page

    with connect() as con:
        total = con.execute(
            f"SELECT COUNT(*) c FROM documents d {wsql}", params).fetchone()["c"]
        rows = con.execute(
            f"SELECT d.id, d.deed_number, d.deed_type, d.year, d.book_no, d.sr_office, "
            f"d.status, (d.pdf_file IS NOT NULL) has_pdf, u.full_name locked_name, "
            f"a.full_name assigned_name, a.id assigned_id, "
            f"le.full_name last_edited_name, to_char(d.last_edited_at, 'DD Mon HH24:MI') last_edited_at, "
            f"{PARTY_NAME_SQL} first_party, {PARTY_NAME_SQL} second_party "
            f"FROM documents d LEFT JOIN users u ON u.id = d.locked_by "
            f"LEFT JOIN users a ON a.id = d.assigned_to "
            f"LEFT JOIN users le ON le.id = d.last_edited_by "
            f"{wsql} ORDER BY {order_col} {order}, d.id LIMIT %s OFFSET %s",
            ["First party%", "Second party%"] + params + [per_page, offset]).fetchall()
    return {"total": total, "page": page, "per_page": per_page,
            "results": [dict(r) for r in rows]}


# ---------- edits & lifecycle ----------

class FieldPatch(BaseModel):
    value: str


def require_lock(con, doc_id, user):
    doc = con.execute("SELECT * FROM documents WHERE id = %s FOR UPDATE",
                      (doc_id,)).fetchone()
    if not doc:
        raise HTTPException(404, "Document not found")
    if doc["status"] != "in_review" or doc["locked_by"] != user["id"]:
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
        if f["current_value"] != body.value:
            con.execute("UPDATE fields SET current_value = %s WHERE id = %s",
                        (body.value, field_id))
            con.execute(
                "INSERT INTO edit_log (document_id, field_id, old_value, new_value, action, user_id) "
                "VALUES (%s,%s,%s,%s,'edit',%s)",
                (doc_id, field_id, f["current_value"], body.value, user["id"]))
            con.execute(
                "UPDATE documents SET locked_at = now(), "
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


@app.post("/api/documents/{doc_id}/skip")
def skip(doc_id: int, user=Depends(current_user)):
    with connect() as con:
        require_lock(con, doc_id, user)
        con.execute("UPDATE documents SET status='pending', locked_by=NULL, "
                    "locked_at=NULL WHERE id=%s", (doc_id,))
        con.commit()
    return {"ok": True}


# ---------- admin ----------

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
    if body.role not in ("expert", "admin"):
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
    """Corrected dataset in the same layout as the input Excel
    (one sheet per book, original columns, party strings reassembled),
    plus an Audit sheet listing every correction."""
    buf = build_export_workbook()
    return StreamingResponse(
        buf, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=corrected_deeds.xlsx"})


# static frontend last, so /api wins
app.mount("/", StaticFiles(directory="static", html=True), name="static")
