"""
db.py
======
Campayn AI Assistant — PostgreSQL layer

WHAT THIS MODULE DOES
----------------------
Postgres owns two things in this system:

  1. LangGraph checkpoints — via PostgresSaver (see api/graph.py).
     Replaces MemorySaver, which kept every checkpoint version for every
     session in process memory with no TTL and no eviction: roughly 115 KB
     per turn, cleared only by a restart. On a long-lived pod that is an
     unbounded leak, and with >1 worker each process had its own copy.

  2. Feedback + run log — durable rows for thumbs up/down and per-request
     telemetry (latency, provider that served it, whether MCQ was asked).
     Previously feedback only went to LangSmith, so it was gone if that
     was disabled.

Everything here FAILS SOFT. If Postgres is unreachable the app keeps
serving: the graph falls back to MemorySaver and feedback writes are
logged and dropped. A database outage must not take the assistant down.

Redis stays where it is — conversation history (api/graph.py) and the
intent-parse cache (api/cache.py). Creator profiles stay in pandas: the
in-memory filter runs in ~150ms, and a SQL round trip would be slower.

Author : Mohit Chauhan — Campayn Multimodal AI Assistant
"""

from __future__ import annotations

import os
from typing import Any, Optional

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://campayn:campayn@localhost:5432/campayn",
)
DB_ENABLED    = os.environ.get("DB_ENABLED", "true").lower() == "true"
DB_POOL_MIN   = int(os.environ.get("DB_POOL_MIN", "1"))
DB_POOL_MAX   = int(os.environ.get("DB_POOL_MAX", "5"))

# Module-level singletons — one pool per process, reused by every request.
_pool: Optional[Any] = None
_pool_failed = False          # remember failure so we don't retry on every call


# ─────────────────────────────────────────────────────────────────
# CONNECTION POOL
# ─────────────────────────────────────────────────────────────────

def get_pool():
    """
    Return a shared psycopg ConnectionPool, or None if Postgres is
    unavailable / disabled.

    The first failure is remembered (_pool_failed) so a dead database
    doesn't cost every later request a connection timeout. Restart the
    process to retry — same trade-off as a simple circuit breaker.
    """
    global _pool, _pool_failed

    if not DB_ENABLED or _pool_failed:
        return None
    if _pool is not None:
        return _pool

    try:
        from psycopg_pool import ConnectionPool

        _pool = ConnectionPool(
            conninfo   = DATABASE_URL,
            min_size   = DB_POOL_MIN,
            max_size   = DB_POOL_MAX,
            open       = True,
            kwargs     = {"autocommit": True},   # PostgresSaver expects autocommit
        )
        _pool.wait(timeout=5.0)                  # fail fast instead of hanging
        print(f"[DB] Postgres pool ready ({DB_POOL_MIN}-{DB_POOL_MAX} connections)")
        return _pool

    except Exception as e:
        print(f"[DB] ⚠ Postgres unavailable ({e}) — feedback + checkpoints degrade to in-memory")
        _pool_failed = True
        _pool = None
        return None


def close_pool() -> None:
    """Close the pool on shutdown so connections are released cleanly."""
    global _pool
    if _pool is not None:
        try:
            _pool.close()
            print("[DB] Postgres pool closed")
        except Exception as e:
            print(f"[DB] Pool close failed: {e}")
        _pool = None


# ─────────────────────────────────────────────────────────────────
# SCHEMA
# ─────────────────────────────────────────────────────────────────

# LangGraph's PostgresSaver creates its own checkpoint tables via .setup().
# These are only the application tables.
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS request_log (
    id            BIGSERIAL PRIMARY KEY,
    session_id    TEXT        NOT NULL,
    run_id        TEXT,
    role          TEXT        NOT NULL,
    query         TEXT        NOT NULL,
    needs_mcq     BOOLEAN     NOT NULL DEFAULT FALSE,
    provider      TEXT,
    total_ms      INTEGER,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS request_log_session_idx ON request_log (session_id, created_at DESC);
CREATE INDEX IF NOT EXISTS request_log_run_idx     ON request_log (run_id);

CREATE TABLE IF NOT EXISTS feedback (
    id          BIGSERIAL PRIMARY KEY,
    run_id      TEXT        NOT NULL,
    score       REAL        NOT NULL CHECK (score >= 0.0 AND score <= 1.0),
    comment     TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS feedback_run_idx ON feedback (run_id);
"""


def init_schema() -> bool:
    """
    Create application tables and LangGraph's checkpoint tables.
    Safe to call on every startup — everything is IF NOT EXISTS.
    Returns True if the schema is ready.
    """
    pool = get_pool()
    if pool is None:
        return False

    try:
        with pool.connection() as conn:
            conn.execute(SCHEMA_SQL)

        # PostgresSaver owns its own DDL; .setup() is idempotent.
        from langgraph.checkpoint.postgres import PostgresSaver
        PostgresSaver(pool).setup()

        print("[DB] Schema ready (request_log, feedback, langgraph checkpoints)")
        return True

    except Exception as e:
        print(f"[DB] ⚠ Schema init failed: {e}")
        return False


# ─────────────────────────────────────────────────────────────────
# WRITES — all fail soft, never raise into a request handler
# ─────────────────────────────────────────────────────────────────

def save_feedback(run_id: str, score: float, comment: Optional[str] = None) -> bool:
    """Persist a thumbs up/down. Returns True if it was stored."""
    pool = get_pool()
    if pool is None:
        return False
    try:
        with pool.connection() as conn:
            conn.execute(
                "INSERT INTO feedback (run_id, score, comment) VALUES (%s, %s, %s)",
                (run_id, score, comment),
            )
        return True
    except Exception as e:
        print(f"[DB] feedback insert failed: {e}")
        return False


def log_request(
    session_id: str,
    role:       str,
    query:      str,
    *,
    run_id:    Optional[str] = None,
    needs_mcq: bool          = False,
    provider:  Optional[str] = None,
    total_ms:  Optional[int] = None,
) -> None:
    """
    Record one request. Telemetry only — failures are logged and swallowed
    so a database problem can never fail a user's chat.
    """
    pool = get_pool()
    if pool is None:
        return
    try:
        with pool.connection() as conn:
            conn.execute(
                """
                INSERT INTO request_log
                    (session_id, run_id, role, query, needs_mcq, provider, total_ms)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (session_id, run_id, role, query[:2000], needs_mcq, provider, total_ms),
            )
    except Exception as e:
        print(f"[DB] request_log insert failed: {e}")


def health() -> str:
    """Health-check string for GET /health."""
    if not DB_ENABLED:
        return "disabled"
    pool = get_pool()
    if pool is None:
        return "unavailable (using in-memory fallback)"
    try:
        with pool.connection() as conn:
            conn.execute("SELECT 1")
        return "ok"
    except Exception as e:
        return f"error: {str(e)[:60]}"
