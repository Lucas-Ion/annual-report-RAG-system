"""Stage two of ingestion: turn a group blocks into the pieces retrieval searches."""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Sequence

from app.db.connection import transaction
from app.db.models import Block, Chunk, ChunkType, Document
from app.db.repositories import BlockRepository, ChunkRepository

TARGET_CHARS = 2_000
CEILING_CHARS = 6_000
SKIP_LABELS = frozenset({"document_index", "checkbox_selected", "checkbox_unselected"})
JOIN = "\n\n"
CHARS_PER_TOKEN = 4
_SEPARATOR = re.compile(r"^[\s|:-]+$")
_WHITESPACE = re.compile(r"\s+")


def estimate_tokens(text: str) -> int:
    return len(text) // CHARS_PER_TOKEN


def is_separator_row(line: str) -> bool:
    return bool(line.strip()) and _SEPARATOR.match(line) is not None


def collapse_table(text: str) -> str:
    lines: list[str] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        cells = line.split("|")
        if len(cells) < 3:
            lines.append(_WHITESPACE.sub(" ", line).strip())
            continue
        if is_separator_row(line):
            lines.append("|" + "|".join(" --- " for _ in cells[1:-1]) + "|")
            continue
        inner = [_WHITESPACE.sub(" ", cell).strip() for cell in cells[1:-1]]
        lines.append("| " + " | ".join(inner) + " |")
    return "\n".join(lines)


def _cells(line: str) -> list[str]:
    parts = line.split("|")
    return [cell.strip() for cell in parts[1:-1]] if len(parts) > 2 else [line.strip()]


def _render(grid: list[list[str]], columns: list[int]) -> str:
    return "\n".join(
        "| " + " | ".join(row[index] for index in columns) + " |" for row in grid
    )


def split_by_columns(text: str, ceiling: int) -> list[str]:
    grid = [_cells(line) for line in text.splitlines() if line.strip()]
    if not grid:
        return [text]

    width = max(len(row) for row in grid)
    grid = [row + [""] * (width - len(row)) for row in grid]

    pieces: list[str] = []
    group: list[int] = []
    for column in range(width):
        if group and len(_render(grid, [*group, column])) > ceiling:
            pieces.append(_render(grid, group))
            group = [column]
        else:
            group.append(column)
    if group:
        pieces.append(_render(grid, group))
    return pieces


def split_table(text: str, ceiling: int = CEILING_CHARS) -> list[str]:
    text = collapse_table(text)
    lines = text.splitlines()
    if len(text) <= ceiling:
        return [text]
    if len(lines) < 3:
        return split_by_columns(text, ceiling)

    header_height = 2 if is_separator_row(lines[1]) else 1
    header = "\n".join(lines[:header_height])
    body = [line for line in lines[header_height:] if line.strip()]

    by_row: list[str] = []
    current: list[str] = []
    size = len(header)
    for row in body:
        if current and size + len(row) + 1 > ceiling:
            by_row.append(header + "\n" + "\n".join(current))
            current = []
            size = len(header)
        current.append(row)
        size += len(row) + 1
    if current:
        by_row.append(header + "\n" + "\n".join(current))

    pieces: list[str] = []
    for piece in by_row or [text]:
        pieces.extend(
            [piece] if len(piece) <= ceiling else split_by_columns(piece, ceiling)
        )
    return pieces


def build_context_header(document: Document, section: str | None) -> str:
    parts = [document.company, f"Annual Report {document.year}"]
    if section:
        parts.append(section)
    return " | ".join(parts)


def build_chunks(document: Document, blocks: Sequence[Block]) -> list[Chunk]:
    if document.id is None:
        raise ValueError("cannot chunk a document that has not been created")
    document_id = document.id

    chunks: list[Chunk] = []
    buffer: list[Block] = []
    buffer_chars = 0
    section: str | None = None
    buffer_section: str | None = None
    held_caption: Block | None = None

    def emit(
        text: str, kind: ChunkType, first: int, last: int, sect: str | None
    ) -> None:
        chunks.append(
            Chunk(
                document_id=document_id,
                seq=len(chunks),
                page_start=first,
                page_end=last,
                section=sect,
                chunk_type=kind,
                context_header=build_context_header(document, sect),
                text=text,
                token_count=estimate_tokens(text),
            )
        )

    def buffer_add(block: Block) -> None:
        nonlocal buffer_chars, buffer_section
        if not buffer:
            buffer_section = section
        buffer.append(block)
        buffer_chars += len(block.text) + len(JOIN)

    def flush() -> None:
        nonlocal buffer, buffer_chars, buffer_section
        if not buffer:
            return
        emit(
            JOIN.join(block.text for block in buffer),
            ChunkType.PROSE,
            min(block.page_no for block in buffer),
            max(block.page_no for block in buffer),
            buffer_section,
        )
        buffer = []
        buffer_chars = 0
        buffer_section = None

    for block in blocks:
        if block.label in SKIP_LABELS:
            continue

        if block.label == "caption":
            if held_caption is not None:
                buffer_add(held_caption)
            held_caption = block
            continue

        if block.label == "table":
            flush()
            prefix = f"{held_caption.text}{JOIN}" if held_caption else ""
            held_caption = None
            room = max(CEILING_CHARS - len(prefix), TARGET_CHARS)
            for piece in split_table(block.text, room):
                emit(
                    prefix + piece,
                    ChunkType.TABLE,
                    block.page_no,
                    block.page_no,
                    section,
                )
            continue

        if held_caption is not None:
            buffer_add(held_caption)
            held_caption = None

        if block.label == "section_header":
            if buffer_chars >= TARGET_CHARS:
                flush()
            section = block.text

        if buffer and buffer_chars + len(block.text) > CEILING_CHARS:
            flush()
        buffer_add(block)
        if buffer_chars >= TARGET_CHARS:
            flush()

    if held_caption is not None:
        buffer_add(held_caption)
    flush()
    return chunks


def chunk_document(conn: sqlite3.Connection, document: Document) -> int:
    if document.id is None:
        raise ValueError("cannot chunk a document that has not been created")

    blocks = BlockRepository(conn).read_for_document(document.id)
    chunks = build_chunks(document, blocks)

    repository = ChunkRepository(conn)
    with transaction(conn):
        repository.delete_for_document(document.id)
        repository.create_all(chunks)
    return len(chunks)
