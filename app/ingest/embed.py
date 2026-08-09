"""Stage three of ingest: give every chunk a vector.

The shortest stage in the pipeline and almost entirely shell, because the
interesting part happens inside the embedding model. What is left here is
worth getting right anyway: only embedding what is missing, committing often
enough that an interruption is cheap, and pairing vectors back to the chunks
they came from without an off by one.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from itertools import batched

from app.db.connection import transaction
from app.db.models import Document
from app.db.repositories import ChunkRepository
from app.providers.base import EmbeddingProvider

# Chunks per commit. The model has its own internal batch size for the forward
# pass, so this is purely about how much work an interruption throws away.
# Embedding the whole corpus takes about six minutes, so a hundred chunks is
# roughly eight seconds of exposure.
DEFAULT_BATCH_SIZE = 100


def embed_document(
    conn: sqlite3.Connection,
    document: Document,
    provider: EmbeddingProvider,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    on_batch: Callable[[int, int], None] | None = None,
) -> int:
    """Embed a document's chunks, skipping any that already have a vector.

    Safe to call repeatedly. A fully embedded document does no work and
    returns zero, so an interrupted run resumes by simply being run again.

    Chunks are embedded through Chunk.embedding_text rather than Chunk.text,
    which prefixes the context header. That prefix is what lets a passage
    reading "increased by 12% to 4,208" be found by a question naming the
    company, and it must never reach the verbatim check, which is why it lives
    on the model as a property rather than being pasted in here.

    Args:
        conn: An open connection from db.connection.
        document: The report to embed. Its id must be set.
        provider: The embedding provider.
        batch_size: Chunks per commit.
        on_batch: Called after each commit with (embedded so far, total to
            do). For progress reporting only.

    Returns:
        How many vectors this call wrote. Zero means there was nothing left.

    Raises:
        ValueError: If the document has no id.
    """
    if document.id is None:
        raise ValueError("cannot embed a document that has not been created")

    repository = ChunkRepository(conn)
    pending = repository.read_without_embeddings(document.id)
    if not pending:
        return 0

    written = 0
    # strict=False because the final batch is nearly always short, which is
    # the ordinary case here rather than a mistake worth raising over.
    for group in batched(pending, batch_size, strict=False):
        vectors = provider.embed_documents([chunk.embedding_text for chunk in group])

        # zip with strict=True rather than trusting the lengths to line up. A
        # provider that silently dropped or reordered an input would otherwise
        # attach every vector in the batch to the wrong chunk, and the only
        # symptom would be retrieval that is subtly, unaccountably poor.
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
