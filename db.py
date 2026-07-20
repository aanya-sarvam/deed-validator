"""PostgreSQL access layer (psycopg3) + schema + auth primitives.

Configure with env var:
  DATABASE_URL=postgresql://deeds:deeds@localhost:5432/deeds
"""
import os

import bcrypt
import psycopg
from psycopg.rows import dict_row

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://deeds:deeds@localhost:5432/deeds")

SESSION_HOURS = 12

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    username TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    full_name TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('expert','admin')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS sessions (
    token TEXT PRIMARY KEY,
    user_id BIGINT NOT NULL REFERENCES users(id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at TIMESTAMPTZ NOT NULL
);
CREATE TABLE IF NOT EXISTS documents (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    deed_number TEXT UNIQUE NOT NULL,
    deed_type TEXT,
    year INT,
    book_no INT,
    reg_no INT,
    sr_office TEXT,
    district TEXT,
    volume_no TEXT,
    executed_year INT,
    pdf_file TEXT,
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending','in_review','validated','flagged')),
    flag_reason TEXT,
    locked_by BIGINT REFERENCES users(id),
    locked_at TIMESTAMPTZ,
    validated_by BIGINT REFERENCES users(id),
    validated_at TIMESTAMPTZ
);
CREATE TABLE IF NOT EXISTS fields (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    document_id BIGINT NOT NULL REFERENCES documents(id),
    section TEXT NOT NULL,
    label TEXT NOT NULL,
    ocr_value TEXT NOT NULL DEFAULT '',
    current_value TEXT NOT NULL DEFAULT '',
    multiline BOOLEAN NOT NULL DEFAULT FALSE,
    position INT NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS edit_log (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    document_id BIGINT NOT NULL REFERENCES documents(id),
    field_id BIGINT REFERENCES fields(id),
    old_value TEXT,
    new_value TEXT,
    action TEXT NOT NULL,
    user_id BIGINT NOT NULL REFERENCES users(id),
    ts TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_docs_status  ON documents(status);
CREATE INDEX IF NOT EXISTS idx_docs_year    ON documents(year);
CREATE INDEX IF NOT EXISTS idx_fields_doc   ON fields(document_id);
CREATE INDEX IF NOT EXISTS idx_fields_label ON fields(label);
-- (previously indexed (label, current_value), but current_value can now hold
--  large table HTML from Akshar blocks, which exceeds the btree size limit)
DROP INDEX IF EXISTS idx_fields_name;
CREATE INDEX IF NOT EXISTS idx_log_doc      ON edit_log(document_id);

-- migrations for databases created before these columns existed
ALTER TABLE documents ADD COLUMN IF NOT EXISTS assigned_to BIGINT REFERENCES users(id);
CREATE INDEX IF NOT EXISTS idx_docs_assigned ON documents(assigned_to);
ALTER TABLE documents ADD COLUMN IF NOT EXISTS last_edited_by BIGINT REFERENCES users(id);
ALTER TABLE documents ADD COLUMN IF NOT EXISTS last_edited_at TIMESTAMPTZ;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS digitized_text TEXT;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS digitized_status TEXT NOT NULL DEFAULT 'not_started';
-- digitized_status: not_started | processing | ready | corrected | error

-- three-role support (admin / monitor / expert) and a review stage
ALTER TABLE users DROP CONSTRAINT IF EXISTS users_role_check;
ALTER TABLE users ADD CONSTRAINT users_role_check CHECK (role IN ('expert','admin','monitor'));

-- 'reviewed' is a post-validation state set by a monitor; 'assigned_to' a
-- validated doc that a monitor picked up is tracked via review_by
ALTER TABLE documents DROP CONSTRAINT IF EXISTS documents_status_check;
ALTER TABLE documents ADD CONSTRAINT documents_status_check
    CHECK (status IN ('pending','in_review','validated','flagged','reviewed','in_monitor_review'));
ALTER TABLE documents ADD COLUMN IF NOT EXISTS review_by BIGINT REFERENCES users(id);
ALTER TABLE documents ADD COLUMN IF NOT EXISTS reviewed_at TIMESTAMPTZ;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS review_corrected BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS sent_to_review_on DATE;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS sent_to_review_by BIGINT REFERENCES users(id);
ALTER TABLE documents ADD COLUMN IF NOT EXISTS sent_reason TEXT;  -- 'skip' | 'sample'

-- Odia value per field, alongside the (English) current_value
ALTER TABLE fields ADD COLUMN IF NOT EXISTS odia_value TEXT NOT NULL DEFAULT '';
-- field_kind: 'text' (normal field) or 'table' (current_value holds table HTML/JSON).
-- Blocks coming from Akshar carry their layout_tag so the metadata view can
-- mirror Akshar's block structure (Header / Paragraph / Table / …).
ALTER TABLE fields ADD COLUMN IF NOT EXISTS field_kind TEXT NOT NULL DEFAULT 'text';
ALTER TABLE fields ADD COLUMN IF NOT EXISTS layout_tag TEXT;
-- Preserve the original Akshar block (block_id, coordinates, confidence,
-- page_num, reading_order, ...) as JSON so corrected output can be exported
-- in exactly the same shape as the input, with only the text replaced.
ALTER TABLE fields ADD COLUMN IF NOT EXISTS src_block JSONB;
ALTER TABLE fields ADD COLUMN IF NOT EXISTS page_num INT;
-- Original grounding.json header (reg_no, book_label, deed_type, chunks...)
-- so corrected output can be exported in exactly the input shape.
ALTER TABLE documents ADD COLUMN IF NOT EXISTS src_meta JSONB;
"""

SEED_USERS = [
    ("expert1", "Aparna Mishra", "expert"),
    ("expert2", "Debasis Rout", "expert"),
    ("expert3", "Sunita Pradhan", "expert"),
    ("monitor", "Priya Sahoo", "monitor"),
    ("admin", "R. K. Mohapatra", "admin"),
]
DEMO_PASSWORD = "sarvam123"  # seed only — admin should rotate these on first login


def hash_pw(pw: str) -> str:
    return bcrypt.hashpw(pw.encode(), bcrypt.gensalt()).decode()


def check_pw(pw: str, pw_hash: str) -> bool:
    try:
        return bcrypt.checkpw(pw.encode(), pw_hash.encode())
    except ValueError:
        return False


def connect():
    return psycopg.connect(DATABASE_URL, row_factory=dict_row)


def init_db():
    con = connect()
    try:
        # Serialize concurrent workers: CREATE TABLE IF NOT EXISTS is not
        # race-safe in Postgres when several processes boot simultaneously.
        # Use a bounded wait so a stale advisory lock (left by a killed
        # process) can never freeze startup forever — if we can't get it
        # quickly, another worker is handling the schema, so proceed.
        con.execute("SET lock_timeout = '8s'")
        got = con.execute("SELECT pg_try_advisory_lock(424241) AS ok").fetchone()["ok"]
        con.execute(SCHEMA)
        with con.cursor() as cur:
            for u, n, r in SEED_USERS:
                cur.execute(
                    "INSERT INTO users (username, password_hash, full_name, role) "
                    "VALUES (%s,%s,%s,%s) ON CONFLICT (username) DO NOTHING",
                    (u, hash_pw(DEMO_PASSWORD), n, r))
        con.commit()
        if got:
            con.execute("SELECT pg_advisory_unlock(424241)")
    finally:
        con.close()
