"""Stage two of ingest: group blocks into the pieces retrieval searches.

Chunking is where most of the judgement in this pipeline lives, and it is the
stage most likely to be rewritten. That is why almost all of it is a pure
function: build_chunks() takes a list of blocks and returns a list of chunks,
touching nothing else, so a rule can be changed and checked against a list
literal in milliseconds rather than by re-running an ingest.

The sizes below were chosen from the parsed corpus rather than from habit.
Across the five reports, tables are 4.4% of blocks and 49% of the text, with a
median of 1,385 characters and a maximum of 51,603. Prose blocks average 228
characters, and a section heading turns up every four or five blocks.
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Sequence

from app.db.connection import transaction
from app.db.models import Block, Chunk, ChunkType, Document
from app.db.repositories import BlockRepository, ChunkRepository

# Roughly 500 tokens. Small enough that a retrieved chunk is mostly answer
# rather than mostly surroundings, large enough to hold a whole argument.
TARGET_CHARS = 2_000

# Nothing is allowed past this. bge-m3 accepts 8,192 tokens, so this leaves
# generous headroom while keeping the model from silently truncating a chunk,
# which is the failure that matters: the tail of an over-long chunk is stored,
# displayed, and completely unsearchable.
CEILING_CHARS = 6_000

# Dropped here rather than during parsing, so changing this list costs a
# re-chunk of a few seconds instead of another pass over the PDFs.
#
# document_index is the table of contents, eleven blocks of section names and
# page numbers that match plenty of queries and answer none of them. The
# checkbox labels are form artefacts with a handful of characters each.
SKIP_LABELS = frozenset({"document_index", "checkbox_selected", "checkbox_unselected"})

# Blocks are joined with a blank line so the result reads like a document
# rather than a run-on paragraph.
JOIN = "\n\n"

# Only ever used for the token_count column, which is reporting and sizing, not
# correctness. Four characters per token is the usual rule of thumb for English
# prose; tables run denser because digits and pipes tokenize badly, so treat
# the number as an estimate and nothing more.
CHARS_PER_TOKEN = 4

# A markdown separator row: pipes, dashes, colons and spaces, nothing else.
_SEPARATOR = re.compile(r"^[\s|:-]+$")

# Any run of whitespace, including the newlines that appear inside table cells.
_WHITESPACE = re.compile(r"\s+")


def estimate_tokens(text: str) -> int:
    """Approximate how many tokens a string will become.

    Args:
        text: The text to measure.

    Returns:
        A rough token count. Deliberately not exact: getting that right means
        loading the embedding model's tokenizer, which is not worth a
        dependency for a column nothing depends on.
    """
    return len(text) // CHARS_PER_TOKEN


def is_separator_row(line: str) -> bool:
    """Report whether a markdown line is the dashes under a table header.

    Args:
        line: One line of a markdown table.

    Returns:
        True if the line is a separator rather than data.
    """
    return bool(line.strip()) and _SEPARATOR.match(line) is not None


def collapse_table(text: str) -> str:
    """Strip the column alignment padding out of a markdown table.

    Docling pads every cell with spaces so the columns line up when the
    markdown is read as plain text. On a wide table with one verbose column
    that is ruinous: measured across the five reports, 56% of all table text
    is padding, and one 51,603 character table turns out to hold 128
    characters of header spread over 12,900.

    That padding is pure cost. It is embedded, it is stored, it counts against
    the model's context window, and it carries no meaning whatsoever. Removing
    it takes table text from 4.16 MB to 1.84 MB and drops the number of tables
    exceeding the chunk ceiling from 204 to 30.

    The separator row is rebuilt rather than collapsed, because it is made of
    dashes rather than spaces and would otherwise survive at full width.

    Blocks keep the original. This runs at chunk time, so the parser's output
    stays a faithful record of what it produced.

    Args:
        text: The table as Docling emitted it.

    Returns:
        The same table with single spaces around every cell.
    """
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
    """Pull the cell contents out of one markdown table line.

    Args:
        line: A line beginning and ending with a pipe.

    Returns:
        The cells between the outer pipes, stripped.
    """
    parts = line.split("|")
    return [cell.strip() for cell in parts[1:-1]] if len(parts) > 2 else [line.strip()]


def _render(grid: list[list[str]], columns: list[int]) -> str:
    """Rebuild a markdown table from a subset of its columns.

    Args:
        grid: Rows of cells, every row the same width.
        columns: Which column indices to keep, in order.

    Returns:
        A valid markdown table holding only those columns.
    """
    return "\n".join(
        "| " + " | ".join(row[index] for index in columns) + " |" for row in grid
    )


def split_by_columns(text: str, ceiling: int) -> list[str]:
    """Split a table vertically, keeping whole columns together.

    The last resort, for tables whose individual rows exceed the ceiling on
    their own. The sustainability disclosure tables in these reports do this:
    eight columns wide, two rows deep, and every cell a paragraph of
    methodology notes. No amount of splitting by row helps.

    Cutting by column keeps each piece a valid table with its own matching
    header cells, which is the property that makes a fragment readable at all.

    Args:
        text: A collapsed markdown table.
        ceiling: Largest piece to aim for, in characters.

    Returns:
        One or more markdown tables, each a column subset of the original. A
        single cell larger than the ceiling still comes back over size, since
        a cell is the smallest thing left to cut.
    """
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
    """Turn a markdown table into pieces that each stand alone.

    Always collapses the alignment padding first, so every table in the index
    is stored in its compact form whether or not it needed splitting.

    Splitting then happens in two passes. Rows first, repeating the header row
    on every piece, because a fragment reading
    "| Net interest income | 5,214 | 4,908 |" is unusable on its own: nothing
    in it says which years those columns are or what unit the figures are in.
    Any piece still over size after that holds a single enormous row, so it
    goes through a second pass that cuts by column instead.

    Args:
        text: The table as Docling emitted it, padding included.
        ceiling: Largest piece to aim for, in characters.

    Returns:
        One or more markdown tables, collapsed, in reading order.
    """
    text = collapse_table(text)
    lines = text.splitlines()
    if len(text) <= ceiling:
        return [text]
    if len(lines) < 3:
        return split_by_columns(text, ceiling)

    # Docling emits a header row followed by a separator. Guard for the
    # separator being absent, which happens on malformed or single row tables.
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
    """Build the breadcrumb a chunk is embedded with.

    A chunk lifted out of page 300 of an annual report often reads as
    orphaned. "Increased by 12% to 4,208" gives an embedding model nothing to
    work with, and a question mentioning a company by name will not match it.
    Prefixing the company, the year and the section is frequently the
    difference between that chunk being findable and invisible.

    Args:
        document: The report the chunk came from.
        section: Nearest preceding heading, or None above the first one.

    Returns:
        Something like "ABN AMRO | Annual Report 2025 | Sustainability".
    """
    parts = [document.company, f"Annual Report {document.year}"]
    if section:
        parts.append(section)
    return " | ".join(parts)


def build_chunks(document: Document, blocks: Sequence[Block]) -> list[Chunk]:
    """Group a document's blocks into retrieval units.

    The rules, in the order they apply:

      * Skipped labels are dropped outright.
      * A caption is held back one block. If a table follows, the caption
        belongs to that table rather than to the prose before it, and carrying
        "Number of FTE per region" into the table chunk makes the table
        findable by a question that never mentions a column name.
      * A table is never mixed with prose. It closes whatever is open, then
        becomes its own chunk, split by rows if it exceeds the ceiling.
      * A heading is a candidate split point, taken only when what is already
        buffered has reached the target. Splitting at every heading would give
        chunks of around 900 characters, since headings appear every four or
        five blocks.
      * Anything else accumulates until the target is reached.

    A chunk is labelled with the section that was open when it started, not
    the one open when it ended, so a chunk spanning a heading is filed under
    the broader context it began in.

    Args:
        document: The report being chunked. Its id must be set.
        blocks: Its blocks, in reading order.

    Returns:
        Chunks numbered from zero, in document order.

    Raises:
        ValueError: If the document has no id.
    """
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
            # Leave room for the caption so a captioned table is not pushed
            # past the ceiling by its own label. The floor keeps a freakishly
            # long caption from collapsing the budget to nothing.
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
    """Rebuild a document's chunks from its stored blocks.

    Always starts from scratch rather than resuming. Chunking a whole report
    takes well under a second, and a partial rebuild would leave two
    generations of rules mixed together in one index, which is far worse than
    the cost of redoing it.

    Deleting the old chunks also discards their embeddings, which is correct:
    a vector computed from text that no longer exists is worse than no vector,
    because it still gets returned by searches.

    Args:
        conn: An open connection from db.connection.
        document: The report to chunk. Its id must be set.

    Returns:
        How many chunks were written.

    Raises:
        ValueError: If the document has no id.
    """
    if document.id is None:
        raise ValueError("cannot chunk a document that has not been created")

    blocks = BlockRepository(conn).read_for_document(document.id)
    chunks = build_chunks(document, blocks)

    repository = ChunkRepository(conn)
    with transaction(conn):
        repository.delete_for_document(document.id)
        repository.create_all(chunks)
    return len(chunks)
