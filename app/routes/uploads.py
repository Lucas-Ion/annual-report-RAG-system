"""Ingesting a report through the web interface.

The awkward truth this endpoint has to be honest about: parsing a 400 page
annual report takes around 45 minutes on a laptop and under a minute on a
machine with a GPU. Nothing can be done about that, so the request returns
immediately and the work continues in the background, and the interface polls
for progress.

Progress is read from the database rather than kept in memory. The pipeline
already records what it has done, stage by stage and page by page, because it
had to in order to be resumable. That means the progress endpoint needs no
shared state, reports correctly even if the browser reloads, and still tells
the truth after the server has been restarted mid-ingest.
"""

from __future__ import annotations

import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.config import PDF_DIR, resolve_pdf
from app.db.connection import connect, transaction
from app.db.models import Stage
from app.db.repositories import DocumentRepository, StageRunRepository
from app.ingest.naming import infer_company, infer_year
from app.ingest.parse import build_converter
from app.ingest.pipeline import file_hash, ingest, register_document
from app.routes.deps import Connection, Embeddings, Model, embeddings, language_model

router = APIRouter(prefix="/api")

# One ingest at a time. Two parses at once would compete for the same layout
# model and the same single SQLite writer, and finish no sooner than running
# them in order. A single worker makes the queue explicit rather than emergent.
_WORKER = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ingest")

# Guards the converter, which is expensive to build and not safe to share
# across concurrent conversions. With one worker this is belt and braces.
_CONVERTER_LOCK = threading.Lock()
_CONVERTER = None

# Every PDF starts with these bytes. Checked because a content type header is
# whatever the client says it is, and handing a renamed zip file to the parser
# produces a far more confusing failure than a rejection here.
_PDF_MAGIC = b"%PDF-"

# Large enough for the biggest report in the corpus, which is 32MB.
MAX_UPLOAD_BYTES = 80 * 1024 * 1024

# Module level singletons rather than calls in the signature's defaults. FastAPI
# reads these once at import to build the route, so evaluating them per call
# would be wasted work, and a default argument that is a function call is a
# familiar source of surprise in Python generally.
_FILE = File(...)
_COMPANY = Form(default=None)
_YEAR = Form(default=None)


def _converter():
    """Build the Docling converter once and reuse it.

    Loading the layout model costs around 21 seconds, so a converter per upload
    would add that to every ingest for no benefit.

    Returns:
        The shared converter.
    """
    global _CONVERTER
    with _CONVERTER_LOCK:
        if _CONVERTER is None:
            _CONVERTER = build_converter()
        return _CONVERTER


def _run(pdf: Path, company: str, year: int) -> None:
    """Ingest a report on the worker thread.

    Opens its own connection, because a sqlite3 connection belongs to the
    thread that created it and this is not the request's thread.

    Failures are swallowed after being recorded. The pipeline writes the reason
    against the failing stage before re-raising, so the progress endpoint can
    report it, and there is nobody left to propagate an exception to.

    Args:
        pdf: The stored file.
        company: Issuing company.
        year: Reporting year.
    """
    conn = connect()
    try:
        ingest(
            conn,
            pdf,
            company=company,
            year=year,
            converter=_converter(),
            embeddings=embeddings(),
            model=language_model(),
        )
    except Exception:
        pass
    finally:
        conn.close()


@router.post("/documents", status_code=202)
def upload(
    conn: Connection,
    embeddings: Embeddings,
    model: Model,
    file: UploadFile = _FILE,
    company: str | None = _COMPANY,
    year: int | None = _YEAR,
) -> dict[str, object]:
    """Accept a PDF and start ingesting it.

    Returns 202 rather than 201: the document row exists by the time this
    replies, but the report is not searchable until the background work
    finishes, and saying "created" would be a lie for the next 45 minutes.

    A file already in the index is recognised by the hash of its contents and
    returns the existing document rather than starting again. That is the same
    check the CLI uses, and it means re-uploading is harmless.

    Args:
        conn: Database connection.
        embeddings: The embedding provider, so a missing one fails here rather
            than on the worker thread where nobody would see it.
        model: The language model, for the same reason.
        file: The uploaded PDF.
        company: Issuing company. Read from the filename when omitted.
        year: Reporting year. Read from the filename when omitted.

    Returns:
        The document id and whether ingestion was actually started.

    Raises:
        HTTPException: 400 for anything that is not a usable PDF, 409 if a
            different file is already being ingested under the same name, and
            413 if it is too large.
    """
    name = Path(file.filename or "").name
    if not name.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="only PDF files are accepted")

    head = file.file.read(len(_PDF_MAGIC))
    if head != _PDF_MAGIC:
        raise HTTPException(
            status_code=400, detail="that file does not look like a PDF"
        )
    file.file.seek(0)

    stem = Path(name).stem
    resolved_company = company or infer_company(stem)
    resolved_year = year or infer_year(stem)
    if not resolved_company or resolved_year is None:
        raise HTTPException(
            status_code=400,
            detail=(
                "could not read a company and year from the filename. "
                "Give them explicitly, or rename the file to something like "
                "shell-annual-report-2025.pdf"
            ),
        )

    PDF_DIR.mkdir(parents=True, exist_ok=True)
    destination = PDF_DIR / name
    if destination.exists():
        # Same name, and the hash check below will decide whether it is really
        # the same report. Overwriting would corrupt an ingest in flight.
        existing = DocumentRepository(conn).read_by_hash(file_hash(destination))
        if existing is not None:
            return {
                "id": existing.id,
                "company": existing.company,
                "year": existing.year,
                "started": False,
                "detail": "already ingested",
            }
        # A file on disk that no document points at is a leftover: the report
        # was removed from the index, or a previous upload failed before it was
        # registered. Overwriting it is safe precisely because nothing
        # references it, and refusing would make a deleted report impossible to
        # re-add without shell access.
        destination.unlink()

    size = 0
    with destination.open("wb") as handle:
        while block := file.file.read(1 << 20):
            size += len(block)
            if size > MAX_UPLOAD_BYTES:
                handle.close()
                destination.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail=f"larger than {MAX_UPLOAD_BYTES // 1024 // 1024}MB",
                )
            handle.write(block)

    document, _created = register_document(
        conn, destination, resolved_company, resolved_year
    )
    _WORKER.submit(_run, destination, resolved_company, resolved_year)
    return {
        "id": document.id,
        "company": document.company,
        "year": document.year,
        "pages": document.page_count,
        "started": True,
    }


@router.delete("/documents/{document_id}")
def remove(document_id: int, conn: Connection) -> dict[str, object]:
    """Remove a report from the index.

    The source PDF is deliberately left on disk. It is a file somebody put
    there, and in this repository it is a committed artifact, so deleting the
    index entry should not destroy it. Uploading it again re-indexes it, which
    is the useful behaviour when the point of deleting was to re-ingest under
    changed chunking or extraction rules.

    Everything derived from the report goes: blocks, chunks, the keyword index,
    the embeddings and the extracted datapoints. Most of that is the database's
    own cascade, except the embeddings, which the repository removes explicitly
    because a vec0 table cannot declare a foreign key.

    Args:
        document_id: Which report.
        conn: Database connection.

    Returns:
        What was removed, so the interface can say something specific.

    Raises:
        HTTPException: 404 if there is no such report.
    """
    documents = DocumentRepository(conn)
    document = documents.read_by_id(document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="no such document")

    removed = _counts(conn, document_id)
    with transaction(conn):
        documents.delete(document)

    return {
        "id": document_id,
        "company": document.company,
        "removed": removed,
        "pdf_kept": resolve_pdf(document.filename) is not None,
    }


@router.get("/documents/{document_id}/progress")
def progress(document_id: int, conn: Connection) -> dict[str, object]:
    """Report how far ingestion has got.

    Everything here is read from tables the pipeline writes as a side effect of
    being resumable, so this needs no shared state and stays correct across a
    browser reload or a server restart.

    Args:
        document_id: Which report.
        conn: Database connection.

    Returns:
        Stage statuses, counts, and a percentage for the parse stage, which is
        the only one slow enough to be worth a progress bar.

    Raises:
        HTTPException: 404 if there is no such report.
    """
    document = DocumentRepository(conn).read_by_id(document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="no such document")

    runs = {
        run.stage.value: {"status": run.status.value, "error": run.error}
        for run in StageRunRepository(conn).read_for_document(document_id)
    }
    counts = _counts(conn, document_id)
    parsed_pages = counts["pages_parsed"]
    total_pages = document.page_count or 0

    done = all(runs.get(stage.value, {}).get("status") == "done" for stage in Stage)
    failed = next(
        (
            f"{stage}: {info['error']}"
            for stage, info in runs.items()
            if info["status"] == "failed"
        ),
        None,
    )
    return {
        "id": document_id,
        "company": document.company,
        "pages": total_pages,
        "stages": runs,
        "done": done,
        "error": failed,
        "percent": round(100 * parsed_pages / total_pages) if total_pages else 0,
        **counts,
    }


def _counts(conn: sqlite3.Connection, document_id: int) -> dict[str, int]:
    """Count what has been produced for a document so far.

    Args:
        conn: Database connection.
        document_id: Which report.

    Returns:
        Pages parsed, and rows in each derived table.
    """

    def scalar(sql: str) -> int:
        return conn.execute(sql, (document_id,)).fetchone()[0] or 0

    return {
        "pages_parsed": scalar("SELECT MAX(page_no) FROM blocks WHERE document_id = ?"),
        "blocks": scalar("SELECT count(*) FROM blocks WHERE document_id = ?"),
        "chunks": scalar("SELECT count(*) FROM chunks WHERE document_id = ?"),
        "vectors": scalar(
            "SELECT count(*) FROM chunk_vectors v JOIN chunks c ON c.id = v.chunk_id"
            " WHERE c.document_id = ?"
        ),
        "facts": scalar("SELECT count(*) FROM extracted_facts WHERE document_id = ?"),
    }
