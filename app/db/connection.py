"""
Open and configure SQLite connections for the system

a quick note: a sqlite3 connection belongs to the thread that created it.
The web layer should open one per request and close it.
"""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

import sqlite_vec

_SCHEMA_PATH = Path(__file__).parent / "schema.sql"

_DEFAULT_DB_PATH = Path(__file__).resolve().parents[2] / "data" / "rag.db"


def database_path() -> Path:
    override = os.environ.get("RAG_DB_PATH")
    return Path(override).resolve() if override else _DEFAULT_DB_PATH


def connect(path: Path | None = None) -> sqlite3.Connection:
    path = path or database_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    _apply_pragmas(conn)
    return conn


def _apply_pragmas(conn: sqlite3.Connection) -> None:
    # Off by default in SQLite, needs to be on for ON DELETE to CASCADE PROPERLY
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA busy_timeout = 5000")
    conn.execute("PRAGMA synchronous = NORMAL")


def init_db(path: Path | None = None) -> sqlite3.Connection:
    conn = connect(path)
    conn.executescript(_SCHEMA_PATH.read_text(encoding="utf-8"))
    conn.commit()
    return conn


@contextmanager
def transaction(conn: sqlite3.Connection) -> Generator[sqlite3.Connection]:
    try:
        yield conn
    except Exception:
        conn.rollback()
        raise
    else:
        conn.commit()


if __name__ == "__main__":
    # Run: | uv run python -m app.db.connection | to get the entry point to the DB
    connection = init_db()
    tables = connection.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
    ).fetchall()
    print(f"{database_path()}: {len(tables)} tables")
    for row in tables:
        print(f"  {row['name']}")
    connection.close()
