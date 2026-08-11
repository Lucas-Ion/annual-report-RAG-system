"""Fixtures"""

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
    value = getattr(entity, "id", None)
    assert isinstance(value, int), f"{entity!r} has no id"
    return value


class FakeEmbeddings:

    @property
    def dimensions(self) -> int:
        return DIMENSIONS

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    @staticmethod
    def _vector(text: str) -> list[float]:
        values = [0.0] * DIMENSIONS
        for word in text.casefold().split():
            digest = hashlib.blake2b(word.encode(), digest_size=4).digest()
            values[int.from_bytes(digest) % DIMENSIONS] += 1.0
        length = math.sqrt(sum(value * value for value in values)) or 1.0
        return [value / length for value in values]


class FakeModel:

    def __init__(
        self, reply: str = "an answer", parsed: BaseModel | None = None
    ) -> None:
        self.reply = reply
        self.parsed = parsed
        self.prompts: list[str] = []

    def complete(self, *, system: str, prompt: str, max_tokens: int = 4096) -> str:
        self.prompts.append(prompt)
        return self.reply

    def stream(
        self, *, system: str, prompt: str, max_tokens: int = 4096
    ) -> Iterator[str]:
        self.prompts.append(prompt)
        for index in range(0, len(self.reply), 8):
            yield self.reply[index : index + 8]

    def parse[T: BaseModel](
        self, *, system: str, prompt: str, schema: type[T], max_tokens: int = 2048
    ) -> T:
        self.prompts.append(prompt)
        return self.parsed if self.parsed is not None else schema()  # type: ignore[return-value]


@pytest.fixture
def conn(tmp_path, monkeypatch) -> Iterator[sqlite3.Connection]:
    path = tmp_path / "test.db"
    monkeypatch.setenv("RAG_DB_PATH", str(path))
    connection = init_db(path)
    try:
        yield connection
    finally:
        connection.close()


@pytest.fixture
def make_pdf(tmp_path):

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
    return FakeEmbeddings()


@pytest.fixture
def model() -> FakeModel:
    return FakeModel()


@pytest.fixture
def document(conn) -> Document:
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
