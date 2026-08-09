"""Runs the ingest stages in order, and remembers which ones are done.

The single entry point for getting a PDF into the system. It exists as a
module rather than living inside the CLI because the web layer needs the same
thing when somebody uploads a report, and two implementations of "ingest a
document" would drift apart within a week.

Every stage is skipped if it has already succeeded, so calling this on a fully
ingested document is a fast no-op and calling it after an interruption picks
up where the last run stopped. That property is why stage bookkeeping lives
here rather than inside the stage modules: parse.py, chunk.py and embed.py
each do one job and know nothing about having been run before.
"""

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

# The order is not negotiable: chunking reads blocks, embedding reads chunks,
# and extraction searches those embeddings to find what to read.
DEFAULT_STAGES: tuple[Stage, ...] = (
    Stage.PARSE,
    Stage.CHUNK,
    Stage.EMBED,
    Stage.EXTRACT,
)

# Called with the stage and a line worth showing a human. Progress reporting
# is the caller's business, so a CLI prints it and a web request could push it
# down a socket, but the pipeline decides what is worth saying.
ProgressCallback = Callable[[Stage, str], None]


def file_hash(pdf: Path) -> str:
    """Fingerprint a file by its contents.

    Read in chunks rather than all at once. Reports are 15 to 31MB today,
    which would be fine to load whole, but nothing about this should care how
    large a filing gets.

    Args:
        pdf: Path to the file.

    Returns:
        Hex sha256 of the bytes.
    """
    digest = hashlib.sha256()
    with pdf.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def register_document(
    conn: sqlite3.Connection, pdf: Path, company: str, year: int
) -> tuple[Document, bool]:
    """Find or create the document row for a PDF.

    Identity is the hash of the file's contents, not its name. That is what
    makes re-ingesting safe: the same report under a different filename is
    recognised and its finished stages are not repeated, and an edited report
    under the same name is correctly treated as a new document.

    Args:
        conn: An open connection.
        pdf: Path to the PDF.
        company: Issuing company.
        year: Reporting year.

    Returns:
        The document, and whether it was newly created.
    """
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
    """Take a PDF all the way from a file to a searchable document.

    Args:
        conn: An open connection from db.connection.
        pdf: Path to the PDF.
        company: Issuing company, for display and for scoping questions.
        year: Reporting year.
        converter: A Docling converter to reuse across documents. One is built
            on demand if the parse stage runs and none was given, which costs
            about 21 seconds of model loading each time, so pass one when
            ingesting more than a single report.
        embeddings: The embedding provider. Without one the embed and extract
            stages are skipped rather than failing, which is what lets a caller
            parse and chunk without waiting on a 2GB model download.
        model: The language model. Without one the extract stage is skipped,
            so the pipeline runs end to end with no API key and no spend.
        stages: Which stages to attempt, in order.
        parse_batch_size: Pages per Docling conversion. Do not change this
            partway through a document, see parse.plan_batches.
        on_progress: Called with a stage and a line of human readable
            progress.

    Returns:
        The document, with page_count filled in.

    Raises:
        FileNotFoundError: If the PDF is not where it says it is.
        Exception: Whatever a stage raised, after recording the failure
            against that stage. Later stages are not attempted, since each one
            reads what the previous produced.
    """
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

        # Extraction needs both: the embeddings to find candidate excerpts and
        # the model to read them.
        if stage is Stage.EXTRACT and (model is None or embeddings is None):
            report(stage, "no language model or embedding provider, skipping")
            continue

        # Each transition gets its own transaction. Repositories never commit
        # on their own, and a finish() left uncommitted at the end of a run is
        # discarded, which would make a finished stage look like it had never
        # happened and cost the whole thing again on the next attempt.
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
    """Run one stage and describe what it did.

    Args:
        conn: An open connection.
        document: The document being processed.
        stage: Which stage to run.
        pdf: Path to the source PDF, needed by parsing.
        converter: A converter to reuse, or None to build one.
        embeddings: The embedding provider, needed by embedding.
        parse_batch_size: Pages per Docling conversion.
        report: Progress callback.

    Returns:
        A one line summary for the caller to display.

    Raises:
        NotImplementedError: For a stage this module does not yet know how to
            run, which today means extraction.
    """
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
        assert embeddings is not None  # guarded by the caller
        written = embed_document(
            conn,
            document,
            embeddings,
            on_batch=lambda done, total: report(stage, f"  {done}/{total} embedded"),
        )
        return f"{written} vectors written"

    if stage is Stage.EXTRACT:
        assert model is not None and embeddings is not None  # guarded above
        stored = extract_document(
            conn,
            document,
            model,
            embeddings,
            on_progress=lambda line: report(stage, line),
        )
        return f"{stored} facts stored"

    raise NotImplementedError(f"no runner for stage {stage.value}")
