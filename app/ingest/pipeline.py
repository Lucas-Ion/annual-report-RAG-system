"""Runs the ingestion stages in order and log which ones are done."""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Callable, Sequence
from pathlib import Path

from docling.document_converter import DocumentConverter

from app.db.connection import transaction
from app.db.models import Document, Stage
from app.db.repositories import DocumentRepository, StageRunRepository
from app.ingest.chunk import chunk_document
from app.ingest.embed import embed_document
from app.ingest.extract import extract_document
from app.ingest.parse import (
    DEFAULT_BATCH_SIZE,
    build_converter,
    page_count,
    parse_document,
)
from app.providers.base import EmbeddingProvider, StructuredExtractor

DEFAULT_STAGES: tuple[Stage, ...] = (
    Stage.PARSE,
    Stage.CHUNK,
    Stage.EMBED,
    Stage.EXTRACT,
)

ProgressCallback = Callable[[Stage, str], None]


def file_hash(pdf: Path) -> str:
    digest = hashlib.sha256()
    with pdf.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def register_document(
    conn: sqlite3.Connection, pdf: Path, company: str, year: int
) -> tuple[Document, bool]:
    documents = DocumentRepository(conn)
    digest = file_hash(pdf)

    existing = documents.read_by_hash(digest)
    if existing is not None:
        return existing, False

    with transaction(conn):
        created = documents.create(
            Document(
                filename=pdf.name,
                file_hash=digest,
                company=company,
                year=year,
                page_count=page_count(pdf),
            )
        )
    return created, True


def ingest(
    conn: sqlite3.Connection,
    pdf: Path,
    *,
    company: str,
    year: int,
    converter: DocumentConverter | None = None,
    embeddings: EmbeddingProvider | None = None,
    model: StructuredExtractor | None = None,
    stages: Sequence[Stage] = DEFAULT_STAGES,
    parse_batch_size: int = DEFAULT_BATCH_SIZE,
    on_progress: ProgressCallback | None = None,
) -> Document:
    if not pdf.is_file():
        raise FileNotFoundError(f"no PDF at {pdf}")

    document, created = register_document(conn, pdf, company, year)
    assert document.id is not None  # register_document always returns a stored row

    def report(stage: Stage, message: str) -> None:
        if on_progress is not None:
            on_progress(stage, message)

    report(
        Stage.PARSE,
        f"{'registered' if created else 'already registered'} "
        f"as document {document.id} ({document.page_count} pages)",
    )

    runs = StageRunRepository(conn)
    for stage in stages:
        if runs.is_done(document.id, stage):
            report(stage, "already done, skipping")
            continue

        if stage is Stage.EMBED and embeddings is None:
            report(stage, "no embedding provider, skipping")
            continue

        if stage is Stage.EXTRACT and (model is None or embeddings is None):
            report(stage, "no language model or embedding provider, skipping")
            continue

        with transaction(conn):
            runs.start(document.id, stage)
        try:
            summary = _run_stage(
                conn,
                document,
                stage,
                pdf=pdf,
                converter=converter,
                embeddings=embeddings,
                model=model,
                parse_batch_size=parse_batch_size,
                report=report,
            )
        except Exception as exc:
            with transaction(conn):
                runs.fail(document.id, stage, repr(exc))
            report(stage, f"FAILED: {exc!r}")
            raise
        with transaction(conn):
            runs.finish(document.id, stage)
        report(stage, summary)

    # Re-read so the caller gets page_count as the parse stage recorded it,
    # rather than whatever was known when the row was first created.
    refreshed = DocumentRepository(conn).read_by_id(document.id)
    return refreshed or document


def _run_stage(
    conn: sqlite3.Connection,
    document: Document,
    stage: Stage,
    *,
    pdf: Path,
    converter: DocumentConverter | None,
    embeddings: EmbeddingProvider | None,
    model: StructuredExtractor | None,
    parse_batch_size: int,
    report: ProgressCallback,
) -> str:
    if stage is Stage.PARSE:
        written = parse_document(
            conn,
            document,
            pdf,
            converter=converter or build_converter(),
            batch_size=parse_batch_size,
            on_batch=lambda first, last, count: report(
                stage, f"  pages {first}-{last}: {count} blocks"
            ),
        )
        return f"{written} blocks written"

    if stage is Stage.CHUNK:
        return f"{chunk_document(conn, document)} chunks written"

    if stage is Stage.EMBED:
        assert embeddings is not None
        written = embed_document(
            conn,
            document,
            embeddings,
            on_batch=lambda done, total: report(stage, f"  {done}/{total} embedded"),
        )
        return f"{written} vectors written"

    if stage is Stage.EXTRACT:
        assert model is not None and embeddings is not None
        stored = extract_document(
            conn,
            document,
            model,
            embeddings,
            on_progress=lambda line: report(stage, line),
        )
        return f"{stored} facts stored"

    raise NotImplementedError(f"no runner for stage {stage.value}")
