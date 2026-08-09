"""Stage one of ingest: turn a PDF into blocks."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from pathlib import Path

import pymupdf
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc.document import DoclingDocument
from docling_core.types.doc.items.node import DocItem
from docling_core.types.doc.items.table.table import TableItem
from docling_core.types.doc.items.text import SectionHeaderItem, TitleItem

from app.db.connection import transaction
from app.db.models import Block, Document
from app.db.repositories import BlockRepository, DocumentRepository

DEFAULT_BATCH_SIZE = 25
SKIP_LABELS = frozenset({"page_header", "page_footer", "picture"})


def plan_batches(
    page_count: int, resume_after: int | None, size: int = DEFAULT_BATCH_SIZE
) -> list[tuple[int, int]]:
    """Work out which page ranges still need converting.

    Args:
        page_count: Total pages in the PDF.
        resume_after: Highest page number already stored for this document, or
            None if nothing has been parsed yet.
        size: Pages per batch.

    Returns:
        Inclusive (first_page, last_page) pairs still to be converted, in
        order. Empty when the document is already fully parsed.

    Raises:
        ValueError: If size is not positive, which would loop forever.
    """
    if size < 1:
        raise ValueError(f"batch size must be at least 1, got {size}")

    grid = [
        (first, min(first + size - 1, page_count))
        for first in range(1, page_count + 1, size)
    ]
    if resume_after is None:
        return grid
    return [(first, last) for first, last in grid if first > resume_after]


def to_blocks(document_id: int, parsed: DoclingDocument, first_seq: int) -> list[Block]:
    """Convert one Docling result into blocks, in reading order.

    Args:
        document_id: The document these blocks belong to.
        parsed: The result of one conversion, covering one batch of pages.
        first_seq: Reading order position to assign to the first block. A
            resumed parse continues numbering from where the last run stopped,
            because chunks are built by walking blocks in seq order and the
            column is unique per document.

    Returns:
        The blocks worth keeping, numbered consecutively from first_seq.
    """
    blocks: list[Block] = []
    seq = first_seq
    for item, _level in parsed.iterate_items():
        block = _to_block(item, parsed, document_id, seq)
        if block is not None:
            blocks.append(block)
            seq += 1
    return blocks


def _to_block(
    item: object, parsed: DoclingDocument, document_id: int, seq: int
) -> Block | None:
    """Convert a single Docling item, or return None if it should be dropped.

    Three separate reasons an item gets dropped, and all three are ordinary
    rather than exceptional, which is why this returns None instead of raising.

    Args:
        item: One item from iterate_items(). Typed loosely because Docling
            yields a NodeItem, and only the DocItem subclasses carry the
            provenance and text this function needs.
        parsed: The document the item came from. Needed because rendering a
            table requires the surrounding document, not just the item.
        document_id: The document these blocks belong to.
        seq: Reading order position for this block.

    Returns:
        The block, or None for items with no provenance, no usable text, or a
        label on the skip list.
    """

    if not isinstance(item, DocItem) or not item.prov:
        return None

    label = str(item.label)
    if label in SKIP_LABELS:
        return None

    if isinstance(item, TableItem):
        text = item.export_to_markdown(parsed)
    else:
        text = getattr(item, "text", "") or ""

    if not text.strip():
        return None
    if isinstance(item, TitleItem):
        level = 0
    elif isinstance(item, SectionHeaderItem):
        level = item.level
    else:
        level = None

    provenance = item.prov[0]
    box = provenance.bbox
    return Block(
        document_id=document_id,
        seq=seq,
        page_no=provenance.page_no,
        label=label,
        level=level,
        text=text,
        bbox=(box.l, box.t, box.r, box.b),
    )


# Shell


def build_converter() -> DocumentConverter:
    """Create the converter used for every batch.

    Returns:
        A converter configured for digital PDFs with table structure detection.
    """
    options = PdfPipelineOptions()
    options.do_ocr = False
    options.do_table_structure = True
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
    )


def page_count(pdf: Path) -> int:
    """Count the pages in a PDF.

    Args:
        pdf: Path to the PDF.

    Returns:
        Number of pages.
    """
    with pymupdf.open(pdf) as document:
        return document.page_count


def parse_document(
    conn: sqlite3.Connection,
    document: Document,
    pdf: Path,
    *,
    converter: DocumentConverter | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    on_batch: Callable[[int, int, int], None] | None = None,
) -> int:
    """Parse a PDF into blocks, resuming if a previous run was interrupted.

    Safe to call repeatedly. A fully parsed document does no work and returns
    zero, so the pipeline does not need to guard the call.

    Args:
        conn: An open connection from db.connection.
        document: The already registered document. Its id must be set.
        pdf: Path to the PDF on disk.
        converter: A converter to reuse. One is built if omitted, which is fine
            for a single document and wasteful for several.
        batch_size: Pages per conversion. Do not change this partway through a
            document, see plan_batches.
        on_batch: Called after each batch commits, with the first page, the
            last page, and how many blocks that batch produced. Purely for
            progress reporting, since a silent hour is hard to trust.

    Returns:
        How many blocks this call wrote. Zero means there was nothing left.

    Raises:
        ValueError: If the document has not been created yet, so has no id.
        FileNotFoundError: If the PDF is not where it says it is.
    """
    if document.id is None:
        raise ValueError("cannot parse a document that has not been created")
    if not pdf.is_file():
        raise FileNotFoundError(f"no PDF at {pdf}")

    documents = DocumentRepository(conn)
    blocks = BlockRepository(conn)

    total_pages = document.page_count or page_count(pdf)
    if document.page_count != total_pages:
        document.page_count = total_pages
        with transaction(conn):
            documents.update(document)

    resume_after = blocks.last_parsed_page(document.id)
    batches = plan_batches(total_pages, resume_after, batch_size)
    if not batches:
        return 0

    converter = converter or build_converter()
    seq = blocks.next_seq(document.id)
    written = 0

    for first, last in batches:
        result = converter.convert(pdf, page_range=(first, last))
        batch = to_blocks(document.id, result.document, seq)

        with transaction(conn):
            blocks.create_all(batch)

        seq += len(batch)
        written += len(batch)
        if on_batch is not None:
            on_batch(first, last, len(batch))

    return written
