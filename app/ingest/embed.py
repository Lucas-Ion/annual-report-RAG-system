"""Stage three of ingestion which is to give every chunk a vector."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from itertools import batched

from app.db.connection import transaction
from app.db.models import Document
from app.db.repositories import ChunkRepository
from app.providers.base import EmbeddingProvider

DEFAULT_BATCH_SIZE = 100


def embed_document(
    conn: sqlite3.Connection,
    document: Document,
    provider: EmbeddingProvider,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    on_batch: Callable[[int, int], None] | None = None,
) -> int:
    if document.id is None:
        raise ValueError("cannot embed a document that has not been created")

    repository = ChunkRepository(conn)
    pending = repository.read_without_embeddings(document.id)
    if not pending:
        return 0

    written = 0
    for group in batched(pending, batch_size, strict=False):
        vectors = provider.embed_documents([chunk.embedding_text for chunk in group])

        pairs = [
            (chunk.id, vector)
            for chunk, vector in zip(group, vectors, strict=True)
            if chunk.id is not None
        ]
        with transaction(conn):
            repository.set_embeddings(pairs)

        written += len(pairs)
        if on_batch is not None:
            on_batch(written, len(pending))

    return written
