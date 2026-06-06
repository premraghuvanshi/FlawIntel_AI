"""
backend/auth.py
──────────────────────────────────────────────────────────────────
Authentication & session management backed by SQLite.
Handles user registration, login validation, session tokens,
and all DB schema bootstrapping.
"""

import sqlite3
import hashlib
import hmac
import os
import uuid
import traceback
from datetime import datetime, timedelta
from pathlib import Path

# ── Path constants ────────────────────────────────────────────────
DB_PATH = Path(__file__).resolve().parent.parent / "data" / "storage.db"


# ─────────────────────────────────────────────────────────────────
# DB bootstrap
# ─────────────────────────────────────────────────────────────────

def _get_connection() -> sqlite3.Connection:
    """Return a thread-safe SQLite connection with WAL mode."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.row_factory = sqlite3.Row
    return conn


def bootstrap_schema() -> None:
    """
    Idempotently create all required tables.
    Called once at app startup.
    """
    ddl = """
    CREATE TABLE IF NOT EXISTS users (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        username    TEXT    UNIQUE NOT NULL,
        password_hash TEXT  NOT NULL,
        salt        TEXT    NOT NULL,
        created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS sessions (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id  TEXT    UNIQUE NOT NULL,
        username    TEXT    NOT NULL,
        created_at  TEXT    NOT NULL DEFAULT (datetime('now')),
        expires_at  TEXT    NOT NULL,
        FOREIGN KEY (username) REFERENCES users(username) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS analysis_history (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id      TEXT    NOT NULL,
        username        TEXT    NOT NULL,
        source          TEXT    NOT NULL,          -- 'csv' | 'url'
        raw_rows        INTEGER NOT NULL,
        filtered_rows   INTEGER NOT NULL,
        capped_rows     INTEGER NOT NULL,
        k_clusters      INTEGER NOT NULL,
        silhouette_score REAL   NOT NULL,
        engine_used     TEXT    NOT NULL,          -- 'LLM' | 'Heuristic'
        latency_seconds REAL    NOT NULL,
        created_at      TEXT    NOT NULL DEFAULT (datetime('now')),
        FOREIGN KEY (username) REFERENCES users(username) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS extracted_complaints (
        id                INTEGER PRIMARY KEY AUTOINCREMENT,
        analysis_id       INTEGER NOT NULL,
        cluster_id        INTEGER NOT NULL,
        feature_mentioned TEXT    NOT NULL,
        sentiment_score   REAL    NOT NULL,
        specific_complaint TEXT   NOT NULL,
        FOREIGN KEY (analysis_id) REFERENCES analysis_history(id) ON DELETE CASCADE
    );
    """
    try:
        conn = _get_connection()
        conn.executescript(ddl)
        conn.commit()
        conn.close()
    except Exception:
        traceback.print_exc()
        raise


# ─────────────────────────────────────────────────────────────────
# Password utilities
# ─────────────────────────────────────────────────────────────────

def _generate_salt() -> str:
    return os.urandom(32).hex()


def _hash_password(password: str, salt: str) -> str:
    """PBKDF2-HMAC-SHA256 with 260 000 iterations."""
    dk = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        260_000,
    )
    return dk.hex()


def _verify_password(password: str, salt: str, stored_hash: str) -> bool:
    candidate = _hash_password(password, salt)
    return hmac.compare_digest(candidate, stored_hash)


# ─────────────────────────────────────────────────────────────────
# User management
# ─────────────────────────────────────────────────────────────────

def register_user(username: str, password: str) -> tuple[bool, str]:
    """
    Returns (success: bool, message: str).
    Enforces minimum password length of 8 characters.
    """
    if not username or len(username.strip()) < 3:
        return False, "Username must be at least 3 characters."
    if not password or len(password) < 8:
        return False, "Password must be at least 8 characters."

    salt = _generate_salt()
    pw_hash = _hash_password(password, salt)

    try:
        conn = _get_connection()
        conn.execute(
            "INSERT INTO users (username, password_hash, salt) VALUES (?, ?, ?)",
            (username.strip(), pw_hash, salt),
        )
        conn.commit()
        conn.close()
        return True, "Account created successfully."
    except sqlite3.IntegrityError:
        return False, "Username already exists."
    except Exception:
        traceback.print_exc()
        return False, "Database error during registration."


def authenticate_user(username: str, password: str) -> tuple[bool, str]:
    """
    Returns (success: bool, session_id | error_message).
    On success creates a session valid for 24 hours.
    """
    if not username or not password:
        return False, "Username and password are required."

    try:
        conn = _get_connection()
        row = conn.execute(
            "SELECT password_hash, salt FROM users WHERE username = ?",
            (username.strip(),),
        ).fetchone()

        if row is None:
            conn.close()
            return False, "Invalid credentials."

        if not _verify_password(password, row["salt"], row["password_hash"]):
            conn.close()
            return False, "Invalid credentials."

        # Create session
        session_id = str(uuid.uuid4())
        expires_at = (datetime.utcnow() + timedelta(hours=24)).isoformat()
        conn.execute(
            "INSERT INTO sessions (session_id, username, expires_at) VALUES (?, ?, ?)",
            (session_id, username.strip(), expires_at),
        )
        conn.commit()
        conn.close()
        return True, session_id

    except Exception:
        traceback.print_exc()
        return False, "Authentication error."


def validate_session(session_id: str) -> tuple[bool, str]:
    """
    Returns (valid: bool, username | error_message).
    Purges expired sessions automatically.
    """
    try:
        conn = _get_connection()
        # Purge expired sessions
        conn.execute(
            "DELETE FROM sessions WHERE expires_at < ?",
            (datetime.utcnow().isoformat(),),
        )
        conn.commit()

        row = conn.execute(
            "SELECT username FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        conn.close()

        if row:
            return True, row["username"]
        return False, "Session expired or invalid."

    except Exception:
        traceback.print_exc()
        return False, "Session validation error."


def invalidate_session(session_id: str) -> None:
    """Logout: delete the session row."""
    try:
        conn = _get_connection()
        conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        conn.commit()
        conn.close()
    except Exception:
        traceback.print_exc()


# ─────────────────────────────────────────────────────────────────
# History persistence
# ─────────────────────────────────────────────────────────────────

def persist_analysis(
    session_id: str,
    username: str,
    source: str,
    raw_rows: int,
    filtered_rows: int,
    capped_rows: int,
    k_clusters: int,
    silhouette_score: float,
    engine_used: str,
    latency_seconds: float,
    complaints: list[dict],
) -> int | None:
    """
    Write a full analysis run to DB.
    Returns the analysis_id on success, None on failure.
    complaints: list of {cluster_id, Feature_Mentioned, Sentiment_Score, Specific_Complaint}
    """
    try:
        conn = _get_connection()
        cur = conn.execute(
            """INSERT INTO analysis_history
               (session_id, username, source, raw_rows, filtered_rows, capped_rows,
                k_clusters, silhouette_score, engine_used, latency_seconds)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (session_id, username, source, raw_rows, filtered_rows, capped_rows,
             k_clusters, round(silhouette_score, 6), engine_used, round(latency_seconds, 4)),
        )
        analysis_id = cur.lastrowid

        for c in complaints:
            conn.execute(
                """INSERT INTO extracted_complaints
                   (analysis_id, cluster_id, feature_mentioned, sentiment_score, specific_complaint)
                   VALUES (?, ?, ?, ?, ?)""",
                (analysis_id, c.get("cluster_id", 0),
                 c.get("Feature_Mentioned", ""),
                 round(float(c.get("Sentiment_Score", 0.0)), 4),
                 c.get("Specific_Complaint", "")),
            )

        conn.commit()
        conn.close()
        return analysis_id

    except Exception:
        traceback.print_exc()
        return None


def fetch_history(username: str, limit: int = 10) -> list[dict]:
    """Retrieve the N most recent analysis runs for a user."""
    try:
        conn = _get_connection()
        rows = conn.execute(
            """SELECT * FROM analysis_history
               WHERE username = ?
               ORDER BY created_at DESC LIMIT ?""",
            (username, limit),
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception:
        traceback.print_exc()
        return []


def fetch_complaints_for_analysis(analysis_id: int) -> list[dict]:
    """Retrieve all extracted complaints for a given analysis run."""
    try:
        conn = _get_connection()
        rows = conn.execute(
            "SELECT * FROM extracted_complaints WHERE analysis_id = ? ORDER BY cluster_id",
            (analysis_id,),
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception:
        traceback.print_exc()
        return []
