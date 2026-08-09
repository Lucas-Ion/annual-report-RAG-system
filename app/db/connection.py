"""
Open and configure SQLite connections for the RAG System

Threading note: a sqlite3 connection belongs to the thread that created it.
The web layer should open one per request and close it, not share a global.
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
    """Resolve where the database lives.

    Returns:
        Absolute path to the SQLite file. It may not exist yet.
    """
    override = os.environ.get("RAG_DB_PATH")
    return Path(override).resolve() if override else _DEFAULT_DB_PATH


def connect(path: Path | None = None) -> sqlite3.Connection:
    """Open a fully configured connection to the database.

    Creates the file and its parent directory if they are missing.

    Args:
        path: Database file to open. Defaults to database_path().

    Returns:
        A connection with sqlite_vec loaded, the pragmas in _apply_pragmas
        applied, and rows returned as sqlite3.Row so callers can index columns
        by name instead of by position.

    Raises:
        AttributeError: If this Python build was compiled without support for
            loadable SQLite extensions, which sqlite_vec requires. The system
            Python shipped with macOS is the usual culprit; the uv managed
            interpreter is fine.
    """
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
    """Apply the settings the schema quietly depends on.

    None of these can live in schema.sql. SQLite scopes them to the connection,
    not the file, so a pragma written into the schema would apply once at
    creation time and never again.

    Args:
        conn: A freshly opened connection, before any application query runs.
    """
    # Off by default in SQLite, for compatibility with databases written before
    # foreign keys existed. Without this line every ON DELETE CASCADE in
    # schema.sql is decoration: deleting a document leaves its blocks, chunks
    # and facts orphaned in place, and nothing complains.
    conn.execute("PRAGMA foreign_keys = ON")

    # Lets the chat read while an ingest is still writing. Under the default
    # rollback journal a writer locks out every reader, which would freeze the
    # interface for the length of a 45 minute parse. This one is a property of
    # the file and only has to be set once, but setting it on every connection
    # costs nothing and removes the question of whether it took.
    conn.execute("PRAGMA journal_mode = WAL")

    # Ingest and the web server are separate processes competing for the single
    # writer slot. Without a timeout the loser raises "database is locked"
    # instantly; with one it waits and retries for five seconds first.
    conn.execute("PRAGMA busy_timeout = 5000")

    # WAL makes the default FULL sync stricter than this application needs.
    # NORMAL still survives a process crash and only risks losing the last
    # transaction if the machine itself loses power mid write, which is a fair
    # trade for a local tool inserting chunks by the thousand.
    conn.execute("PRAGMA synchronous = NORMAL")


def init_db(path: Path | None = None) -> sqlite3.Connection:
    """Create or update the database and hand back an open connection.

    Safe to call on every startup. Every statement in schema.sql is guarded by
    IF NOT EXISTS, so running this against a fully populated database does
    nothing at all. That property is what lets the project ship a seeded
    database instead of carrying a migrations framework.

    Args:
        path: Database file to create or open. Defaults to database_path().

    Returns:
        An open connection with the schema applied.
    """
    conn = connect(path)
    conn.executescript(_SCHEMA_PATH.read_text(encoding="utf-8"))
    conn.commit()
    return conn


@contextmanager
def transaction(conn: sqlite3.Connection) -> Generator[sqlite3.Connection]:
    """Group a set of writes so they either all land or none of them do.

    Ingest writes in batches of a few thousand rows. Committing row by row is
    orders of magnitude slower, and worse, it leaves the database holding half
    a document if the process dies partway through one.

    Args:
        conn: An open connection.

    Yields:
        The same connection, so the body of the with block has something to
        call execute on.

    Raises:
        Exception: Whatever the body raised, reraised after the rollback so the
            caller still sees the real failure.
    """
    try:
        yield conn
    except Exception:
        conn.rollback()
        raise
    else:
        conn.commit()


if __name__ == "__main__":
    # Creating the database is a one liner, so it gets a one liner entry point
    # rather than a script of its own:
    #
    #     uv run python -m app.db.connection
    connection = init_db()
    tables = connection.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
    ).fetchall()
    print(f"{database_path()}: {len(tables)} tables")
    for row in tables:
        print(f"  {row['name']}")
    connection.close()
