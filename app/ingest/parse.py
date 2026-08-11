"""Stage one of ingest: turn a PDF into blocks."""

from __future__ import annotations

import importlib.util
import sqlite3
from collections.abc import Callable
from pathlib import Path

import pymupdf
import torch
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.settings import settings as docling_settings
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


def can_compile_models() -> bool:
    if not torch.cuda.is_available():
        return True
    return importlib.util.find_spec("triton") is not None


def build_converter() -> DocumentConverter:
    docling_settings.inference.compile_torch_models = can_compile_models()

    options = PdfPipelineOptions()
    options.do_ocr = False
    options.do_table_structure = True
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
    )


def page_count(pdf: Path) -> int:
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
