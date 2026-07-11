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
CREATE INDEX IF NOT EXISTS idx_fields_name  ON fields(label, current_value);
CREATE INDEX IF NOT EXISTS idx_log_doc      ON edit_log(document_id);

-- migrations for databases created before these columns existed
ALTER TABLE documents ADD COLUMN IF NOT EXISTS assigned_to BIGINT REFERENCES users(id);
CREATE INDEX IF NOT EXISTS idx_docs_assigned ON documents(assigned_to);
"""

SEED_USERS = [
    ("expert1", "Aparna Mishra", "expert"),
    ("expert2", "Debasis Rout", "expert"),
    ("expert3", "Sunita Pradhan", "expert"),
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
        con.execute("SELECT pg_advisory_lock(424241)")
        con.execute(SCHEMA)
        if not con.execute("SELECT 1 FROM users LIMIT 1").fetchone():
            with con.cursor() as cur:
                cur.executemany(
                    "INSERT INTO users (username, password_hash, full_name, role) "
                    "VALUES (%s,%s,%s,%s)",
                    [(u, hash_pw(DEMO_PASSWORD), n, r) for u, n, r in SEED_USERS])
        con.commit()
    finally:
        con.close()  # releases the advisory lock
