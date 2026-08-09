"""Environment loading for the application's entry points.

Called explicitly by the things that start the program, which is every script
in scripts/ and the web app. Not called on import of anything, because a
module that quietly rewrites os.environ when imported is a genuinely nasty
thing to debug: a test that sets a variable then imports a helper finds its
value has changed underneath it.

Individual settings stay where they are used. The database path is read in
db.connection, the embedding device in providers.embeddings, the API key in
providers.claude. Each of those has a sensible default and fails clearly
without one, which beats a single settings object that everything imports and
nothing can be tested without.
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

# app/config.py, so one level up is the repository root.
ENV_PATH = Path(__file__).resolve().parents[1] / ".env"


def load_environment(path: Path | None = None) -> bool:
    """Read .env into the environment, if there is one.

    Existing environment variables win. Someone who has exported a key in
    their shell means it, and a stale file on disk should not override it.

    Args:
        path: File to read. Defaults to .env at the repository root.

    Returns:
        True if a file was found and read, False if there was none. A missing
        .env is not an error: everything except the chat runs without one.
    """
    target = path or ENV_PATH
    if not target.is_file():
        return False
    load_dotenv(target, override=False)
    return True


# Where the source PDFs live. They are committed, so a fresh clone can open a
# report at the page a figure was taken from without downloading anything.
PDF_DIR = Path(__file__).resolve().parents[1] / "data" / "pdfs"


def resolve_pdf(filename: str, directory: Path = PDF_DIR) -> Path | None:
    """Resolve a stored filename to a PDF on disk.

    The containment check is the point. `filename` comes out of the database,
    and joining untrusted-ish text onto a directory is how a path traversal
    happens: a row whose filename is "../../.env" would otherwise be served
    happily. Resolving first and then confirming the result is still inside the
    directory closes that regardless of what the name contains.

    Args:
        filename: The document's stored filename.
        directory: Where source PDFs are kept.

    Returns:
        The path, or None if it escapes the directory or is not there. A
        missing file is an ordinary case: the database ships seeded and someone
        may well have removed the PDFs.
    """
    root = directory.resolve()
    candidate = (root / filename).resolve()
    if not candidate.is_relative_to(root):
        return None
    return candidate if candidate.is_file() else None
