"""Stage four of ingestion which is to pull named datapoints out with citation."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Sequence

from pydantic import BaseModel
from pydantic import Field as PydanticField

from app.db.connection import transaction
from app.db.models import Block, Chunk, Document, Fact
from app.db.repositories import BlockRepository, ChunkRepository, FactRepository
from app.ingest.fields import FIELDS, Field
from app.providers.base import EmbeddingProvider, StructuredExtractor
from app.retrieve import search_many
from app.verify import find_source, locate_page

EXTRACTOR_VERSION = "2026-08-09.2"
CANDIDATE_CHUNKS = 20


class ExtractedValue(BaseModel):

    value_raw: str = PydanticField(
        description=(
            "The value exactly as printed in the report, for example '87,870' "
            "or 'net zero by 2050'. Do not reformat numbers."
        )
    )
    value_numeric: float | None = PydanticField(
        default=None,
        description=(
            "The same value as a plain number, if it has one. '87,870' becomes "
            "87870. '1.4 billion' becomes 1400000000. Null when the value is "
            "not numeric."
        ),
    )
    unit: str | None = PydanticField(
        default=None,
        description=(
            "The unit or basis of measurement, for example 'FTE', 'headcount', "
            "'%', 'tCO2e', 'EUR million'. Null when there is none."
        ),
    )
    verbatim_quote: str = PydanticField(
        description=(
            "A short passage copied character for character out of one of the "
            "excerpts, containing this value. Copy it exactly. Do not correct, "
            "shorten, join or rephrase anything. This is checked against the "
            "source text and the extraction is discarded if it does not match."
        )
    )
    excerpt_number: int = PydanticField(
        description="Which numbered excerpt the quote was taken from."
    )
    confidence: float = PydanticField(
        default=0.5,
        description=(
            "How confident you are, from 0 to 1. Use a low value when the "
            "excerpts are ambiguous or the figure might be a subtotal."
        ),
    )


class Extraction(BaseModel):

    values: list[ExtractedValue] = PydanticField(
        default_factory=list,
        description=(
            "The values found. Empty when the excerpts do not contain the "
            "answer. An empty list is the correct response to excerpts that "
            "do not say; do not guess."
        ),
    )


SYSTEM_PROMPT = """\
You extract specific datapoints from corporate annual reports.

You are given numbered excerpts from one company's report and told what to \
look for. Answer only from those excerpts.

Rules that matter more than being helpful:

1. Every value must carry a quotation copied character for character from an \
excerpt. The quote is checked against the source text and the value is thrown \
away if it does not match exactly. Copy, do not retype.
2. If the excerpts do not contain the answer, return an empty list. A missing \
datapoint is a correct and useful result. An invented one is not.
3. Do not combine figures, calculate totals, or infer a value that is not \
written down.
4. When several candidate figures appear, choose the one the instruction asks \
for and say so through your confidence rather than hedging in the value.
"""


def build_prompt(document: Document, field: Field, chunks: Sequence[Chunk]) -> str:
    excerpts = "\n\n".join(
        f"--- excerpt {number} (page {chunk.page_start}"
        f"{f' to {chunk.page_end}' if chunk.page_end != chunk.page_start else ''}"
        f"{f', section: {chunk.section}' if chunk.section else ''}) ---\n"
        f"{chunk.text}"
        for number, chunk in enumerate(chunks, start=1)
    )
    expectation = (
        "There may be several. Record each one separately."
        if field.multiple
        else "There should be exactly one. If several candidates appear, pick "
        "the one the instruction describes and leave the rest out."
    )
    return (
        f"Company: {document.company}\n"
        f"Report: Annual Report {document.year}\n\n"
        f"Looking for: {field.label}\n\n"
        f"{field.instruction}\n\n"
        f"{expectation}\n\n"
        f"Excerpts:\n\n{excerpts}"
    )


def to_facts(
    extraction: Extraction,
    chunks: Sequence[Chunk],
    document: Document,
    field: Field,
    version: str = EXTRACTOR_VERSION,
    blocks: Sequence[Block] = (),
) -> tuple[list[Fact], list[str]]:
    if document.id is None:
        raise ValueError("cannot extract from a document that has not been created")

    facts: list[Fact] = []
    rejected: list[str] = []

    for value in extraction.values:
        source = find_source(value.verbatim_quote, chunks)
        if source is None:
            rejected.append(
                f"quote not found in any excerpt: {value.verbatim_quote[:70]!r}"
            )
            continue
        facts.append(
            Fact(
                document_id=document.id,
                field_key=field.key,
                value_raw=value.value_raw,
                value_numeric=value.value_numeric,
                unit=value.unit or field.unit_hint,
                verbatim_quote=value.verbatim_quote,
                page_no=locate_page(value.verbatim_quote, source, blocks),
                chunk_id=source.id,
                confidence=value.confidence,
                extractor_version=version,
            )
        )

    return facts, rejected


def _spanning_blocks(conn: sqlite3.Connection, chunks: Sequence[Chunk]) -> list[Block]:
    repository = BlockRepository(conn)
    found: list[Block] = []
    for chunk in chunks:
        if chunk.page_end != chunk.page_start:
            found.extend(
                repository.read_page_range(
                    chunk.document_id, chunk.page_start, chunk.page_end
                )
            )
    return found


def extract_document(
    conn: sqlite3.Connection,
    document: Document,
    model: StructuredExtractor,
    embeddings: EmbeddingProvider,
    *,
    fields: Sequence[Field] = FIELDS,
    candidates: int = CANDIDATE_CHUNKS,
    on_progress: Callable[[str], None] | None = None,
) -> int:
    if document.id is None:
        raise ValueError("cannot extract from a document that has not been created")

    def report(message: str) -> None:
        if on_progress is not None:
            on_progress(message)

    chunks = ChunkRepository(conn)
    facts_repository = FactRepository(conn)

    with transaction(conn):
        facts_repository.delete_for_document(document.id)

    stored = 0
    failed: list[str] = []
    for field in fields:
        excerpts = search_many(
            chunks,
            embeddings,
            field.queries,
            limit=candidates,
            document_id=document.id,
        )
        if not excerpts:
            report(f"  {field.key}: no candidate excerpts found")
            continue

        try:
            extraction = model.parse(
                system=SYSTEM_PROMPT,
                prompt=build_prompt(document, field, excerpts),
                schema=Extraction,
                max_tokens=field.max_tokens,
            )
        except Exception as exc:
            report(
                f"  {field.key}: FAILED, {exc!r}\n"
                f"      an invalid JSON error here almost always means the "
                f"reply was truncated. Raise max_tokens on this field."
            )
            failed.append(field.key)
            continue

        facts, rejected = to_facts(
            extraction,
            excerpts,
            document,
            field,
            blocks=_spanning_blocks(conn, excerpts),
        )

        with transaction(conn):
            facts_repository.create_all(facts)
        stored += len(facts)

        summary = f"  {field.key}: {len(facts)} stored"
        if rejected:
            summary += f", {len(rejected)} rejected as unverifiable"
        report(summary)
        for reason in rejected:
            report(f"      {reason}")

    if failed and len(failed) == len(fields):
        raise RuntimeError(f"every field failed to extract: {', '.join(failed)}")
    if failed:
        report(f"  {len(failed)} of {len(fields)} fields failed: {', '.join(failed)}")
    return stored
