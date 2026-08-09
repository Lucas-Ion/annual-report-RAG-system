"""Fixtures, and the fakes that keep the suite fast and free.

Two things must never happen in a test run: loading the 2GB embedding model,
and calling a paid API. Both are avoided by substituting the two Protocols in
app/providers/base.py, which is the reason those Protocols were split by
capability in the first place. A fake for extraction implements one method.

The fake embedder is not a stub returning zeros. It hashes words into
dimensions, so overlapping text really does produce closer vectors and a
retrieval test can assert that the right chunk came back rather than merely
that something did.
"""

from __future__ import annotations

import hashlib
import math
import sqlite3
from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest
from pydantic import BaseModel

from app.db.connection import init_db, transaction
from app.db.models import Block, Chunk, ChunkType, Document, Fact
from app.db.repositories import (
    BlockRepository,
    ChunkRepository,
    DocumentRepository,
    FactRepository,
)
from app.providers.embeddings import DIMENSIONS


def ident(entity: object) -> int:
    """Return a stored entity's id, narrowed from int or None.

    Repositories type ids as optional because an object built in memory has
    none yet. Anything a fixture hands back has been written, so the None case
    is impossible here, and one helper keeps the assertion out of every test.

    Args:
        entity: Something with an id attribute.

    Returns:
        The id.
    """
    value = getattr(entity, "id", None)
    assert isinstance(value, int), f"{entity!r} has no id"
    return value


class FakeEmbeddings:
    """A deterministic stand-in for bge-m3.

    Each word is hashed to a dimension and the vector is normalised, so two
    texts sharing words end up close together. That is a crude model of
    meaning and a perfectly good model of plumbing: it makes similarity
    assertions meaningful without a 2GB download.
    """

    @property
    def dimensions(self) -> int:
        """Vector width, matching the real provider and the schema."""
        return DIMENSIONS

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed several texts.

        Args:
            texts: The texts to embed.

        Returns:
            One unit length vector per input, in order.
        """
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        """Embed one text.

        Args:
            text: The text to embed.

        Returns:
            A unit length vector.
        """
        return self._vector(text)

    @staticmethod
    def _vector(text: str) -> list[float]:
        """Turn text into a unit length vector by hashing its words.

        Args:
            text: The text to embed.

        Returns:
            DIMENSIONS floats, normalised so distance behaves as it does with
            the real provider.
        """
        values = [0.0] * DIMENSIONS
        for word in text.casefold().split():
            digest = hashlib.blake2b(word.encode(), digest_size=4).digest()
            values[int.from_bytes(digest) % DIMENSIONS] += 1.0
        length = math.sqrt(sum(value * value for value in values)) or 1.0
        return [value / length for value in values]


class FakeModel:
    """A language model that says whatever the test told it to.

    Satisfies TextGenerator and StructuredExtractor without inheriting from
    either, which is the point of using Protocols: a class matching the shape
    is accepted, and the type checker still verifies the shape.
    """

    def __init__(
        self, reply: str = "an answer", parsed: BaseModel | None = None
    ) -> None:
        """Configure the canned responses.

        Args:
            reply: What complete() and stream() return.
            parsed: What parse() returns. Constructed from the schema when
                omitted, which gives an empty result.
        """
        self.reply = reply
        self.parsed = parsed
        self.prompts: list[str] = []

    def complete(self, *, system: str, prompt: str, max_tokens: int = 4096) -> str:
        """Return the canned reply, recording the prompt it was asked with."""
        self.prompts.append(prompt)
        return self.reply

    def stream(
        self, *, system: str, prompt: str, max_tokens: int = 4096
    ) -> Iterator[str]:
        """Yield the canned reply in fragments, as the real provider does."""
        self.prompts.append(prompt)
        for index in range(0, len(self.reply), 8):
            yield self.reply[index : index + 8]

    def parse[T: BaseModel](
        self, *, system: str, prompt: str, schema: type[T], max_tokens: int = 2048
    ) -> T:
        """Return the canned structured result."""
        self.prompts.append(prompt)
        return self.parsed if self.parsed is not None else schema()  # type: ignore[return-value]


@pytest.fixture
def conn(tmp_path, monkeypatch) -> Iterator[sqlite3.Connection]:
    """An empty database in a temporary directory.

    RAG_DB_PATH is set so that anything reaching for the default location
    inside the test lands here too, rather than on the developer's real
    database.

    Yields:
        An open connection with the schema applied.
    """
    path = tmp_path / "test.db"
    monkeypatch.setenv("RAG_DB_PATH", str(path))
    connection = init_db(path)
    try:
        yield connection
    finally:
        connection.close()


@pytest.fixture
def make_pdf(tmp_path):
    """Build a real, minimal PDF.

    Real because register_document counts pages with PyMuPDF, so a handful of
    bytes starting with %PDF is not enough. Page count doubles as a way to
    produce two files with different contents.

    Returns:
        A callable taking a name and a page count, returning the path.
    """

    def build(name: str = "acme-2025.pdf", pages: int = 3) -> Path:
        import pymupdf

        path = tmp_path / name
        document = pymupdf.open()
        for _ in range(pages):
            document.new_page()
        document.save(path)
        document.close()
        return path

    return build


@pytest.fixture
def embeddings() -> FakeEmbeddings:
    """The fake embedding provider."""
    return FakeEmbeddings()


@pytest.fixture
def model() -> FakeModel:
    """A language model returning a plain answer."""
    return FakeModel()


@pytest.fixture
def document(conn) -> Document:
    """One registered report, with no content yet."""
    with transaction(conn):
        return DocumentRepository(conn).create(
            Document(
                filename="acme-annual-report-2025.pdf",
                file_hash="hash-acme",
                company="Acme",
                year=2025,
                page_count=12,
            )
        )


@pytest.fixture
def seeded(conn, document, embeddings) -> Document:
    """A report with blocks, chunks, vectors and one extracted fact.

    Small enough to reason about: four blocks over three pages, where the
    headcount sentence deliberately sits on the second page of a chunk that
    spans a page break. That is the case that produced wrong page numbers in
    the real index, so the suite should always have one.

    Returns:
        The populated document.
    """
    assert document.id is not None
    blocks = BlockRepository(conn)
    chunks = ChunkRepository(conn)

    with transaction(conn):
        blocks.create_all(
            [
                Block(
                    document_id=document.id,
                    seq=0,
                    page_no=1,
                    label="title",
                    level=0,
                    text="Acme Annual Report 2025",
                ),
                Block(
                    document_id=document.id,
                    seq=1,
                    page_no=4,
                    label="section_header",
                    level=1,
                    text="Our people",
                ),
                Block(
                    document_id=document.id,
                    seq=2,
                    page_no=4,
                    label="text",
                    text="This section covers the workforce.",
                ),
                Block(
                    document_id=document.id,
                    seq=3,
                    page_no=5,
                    label="text",
                    text="The average number of employees in 2025 was 12,345.",
                ),
            ]
        )
        stored = chunks.create_all(
            [
                Chunk(
                    document_id=document.id,
                    seq=0,
                    page_start=1,
                    page_end=1,
                    chunk_type=ChunkType.PROSE,
                    context_header="Acme | Annual Report 2025",
                    text="Acme Annual Report 2025",
                ),
                # Spans pages 4 to 5. The figure is on page 5.
                Chunk(
                    document_id=document.id,
                    seq=1,
                    page_start=4,
                    page_end=5,
                    section="Our people",
                    chunk_type=ChunkType.PROSE,
                    context_header="Acme | Annual Report 2025 | Our people",
                    text="Our people\n\nThis section covers the workforce."
                    "\n\nThe average number of employees in 2025 was 12,345.",
                ),
            ]
        )
        chunks.set_embeddings(
            (chunk.id, embeddings.embed_query(chunk.embedding_text))
            for chunk in stored
            if chunk.id is not None
        )
        FactRepository(conn).create(
            Fact(
                document_id=document.id,
                field_key="fte",
                value_raw="12,345",
                value_numeric=12345.0,
                unit="FTE",
                verbatim_quote="The average number of employees in 2025 was 12,345.",
                page_no=5,
                chunk_id=stored[1].id,
                confidence=0.9,
                extractor_version="test",
            )
        )
    return document
