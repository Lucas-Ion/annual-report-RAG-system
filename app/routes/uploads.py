"""Ingesting a report through the web interface."""

from __future__ import annotations

import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.config import PDF_DIR, resolve_pdf
from app.db.connection import connect, transaction
from app.db.models import Document, Stage, StageStatus
from app.db.repositories import DocumentRepository, StageRunRepository
from app.ingest.naming import infer_company, infer_year
from app.ingest.parse import build_converter
from app.ingest.pipeline import file_hash, ingest, register_document
from app.routes.deps import Connection, Embeddings, Model, embeddings, language_model

router = APIRouter(prefix="/api")

_WORKER = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ingest")
_CONVERTER_LOCK = threading.Lock()
_CONVERTER = None
_PDF_MAGIC = b"%PDF-"

MAX_UPLOAD_BYTES = 80 * 1024 * 1024

_FILE = File(...)
_COMPANY = Form(default=None)
_YEAR = Form(default=None)


def _converter():
    """Build the Docling converter once and reuse it.

    Returns:
        The shared converter.
    """
    global _CONVERTER
    with _CONVERTER_LOCK:
        if _CONVERTER is None:
            _CONVERTER = build_converter()
        return _CONVERTER


def _run(pdf: Path, company: str, year: int) -> None:
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
        existing = DocumentRepository(conn).read_by_hash(file_hash(destination))
        if existing is not None:
            return {
                "id": existing.id,
                "company": existing.company,
                "year": existing.year,
                "started": False,
                "detail": "already ingested",
            }
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


def _report(conn: sqlite3.Connection, document: Document) -> dict[str, object]:
    assert document.id is not None
    runs = {
        run.stage.value: run
        for run in StageRunRepository(conn).read_for_document(document.id)
    }
    counts = _counts(conn, document.id)
    total_pages = document.page_count or 0
    parsed = counts["pages_parsed"]

    running = next(
        (name for name, run in runs.items() if run.status.value == "running"), None
    )
    failed = next(
        (
            f"{name}: {run.error}"
            for name, run in runs.items()
            if run.status.value == "failed"
        ),
        None,
    )
    return {
        "id": document.id,
        "company": document.company,
        "pages": total_pages,
        "stages": {
            name: {"status": run.status.value, "error": run.error}
            for name, run in runs.items()
        },
        "running": running,
        "done": all(
            runs.get(stage.value) and runs[stage.value].status.value == "done"
            for stage in Stage
        ),
        "error": failed,
        "percent": round(100 * parsed / total_pages) if total_pages else 0,
        **counts,
    }


@router.get("/documents/in-progress")
def in_progress(conn: Connection) -> list[dict[str, object]]:
    documents = DocumentRepository(conn)
    runs = StageRunRepository(conn)
    unfinished = []
    for document in documents.read():
        if document.id is None:
            continue
        stages = runs.read_for_document(document.id)
        if not stages:
            continue
        if all(run.status is StageStatus.DONE for run in stages) and len(stages) == len(
            Stage
        ):
            continue
        unfinished.append(_report(conn, document))
    return unfinished


@router.delete("/documents/{document_id}")
def remove(document_id: int, conn: Connection) -> dict[str, object]:
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
    document = DocumentRepository(conn).read_by_id(document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="no such document")
    return _report(conn, document)


def _counts(conn: sqlite3.Connection, document_id: int) -> dict[str, int]:
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
