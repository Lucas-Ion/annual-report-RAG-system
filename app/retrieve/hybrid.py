"""Combines the two searches into one ranked list."""

from __future__ import annotations

from collections.abc import Sequence

from app.db.models import Chunk, SearchHit
from app.db.repositories import ChunkRepository
from app.providers.base import EmbeddingProvider

RRF_K = 60
DEFAULT_CANDIDATES = 40
DOCUMENT_OVERFETCH = 6


def fuse(rankings: Sequence[Sequence[int]], k: int = RRF_K) -> list[int]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for position, chunk_id in enumerate(ranking):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + position + 1)
    return sorted(scores, key=lambda chunk_id: -scores[chunk_id])


def interleave(rankings: Sequence[Sequence[int]], limit: int) -> list[int]:
    seen: set[int] = set()
    merged: list[int] = []
    depth = max((len(ranking) for ranking in rankings), default=0)

    for position in range(depth):
        for ranking in rankings:
            if position >= len(ranking):
                continue
            chunk_id = ranking[position]
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            merged.append(chunk_id)
            if len(merged) >= limit:
                return merged
    return merged


def search(
    chunks: ChunkRepository,
    embeddings: EmbeddingProvider,
    query: str,
    *,
    limit: int = 8,
    document_id: int | None = None,
    candidates: int = DEFAULT_CANDIDATES,
) -> list[Chunk]:
    breadth = candidates * (DOCUMENT_OVERFETCH if document_id is not None else 1)

    lexical = chunks.read_by_keywords(query, limit=breadth)
    dense = chunks.read_by_similarity(embeddings.embed_query(query), limit=breadth)

    def ids(hits: Sequence[SearchHit]) -> list[int]:
        return [
            hit.chunk.id
            for hit in hits
            if hit.chunk.id is not None
            and (document_id is None or hit.chunk.document_id == document_id)
        ]

    ranked = fuse([ids(lexical), ids(dense)])
    return chunks.read_by_ids(ranked[:limit])


def search_many(
    chunks: ChunkRepository,
    embeddings: EmbeddingProvider,
    queries: Sequence[str],
    *,
    limit: int = 12,
    document_id: int | None = None,
    candidates: int = DEFAULT_CANDIDATES,
) -> list[Chunk]:
    breadth = candidates * (DOCUMENT_OVERFETCH if document_id is not None else 1)

    def ids(hits: Sequence[SearchHit]) -> list[int]:
        return [
            hit.chunk.id
            for hit in hits
            if hit.chunk.id is not None
            and (document_id is None or hit.chunk.document_id == document_id)
        ]

    per_query = [
        fuse(
            [
                ids(chunks.read_by_keywords(query, limit=breadth)),
                ids(
                    chunks.read_by_similarity(
                        embeddings.embed_query(query), limit=breadth
                    )
                ),
            ]
        )
        for query in queries
    ]
    return chunks.read_by_ids(interleave(per_query, limit))
