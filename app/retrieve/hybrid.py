"""Combines the two searches into one ranked list.

Keyword search and vector search fail in opposite directions, which is the
whole reason both exist. Keyword search finds "Scope 3" and misses "our
workforce" when the question said "employees". Vector search finds the
paraphrase and cheerfully returns Scope 1 when you asked about Scope 3,
because to an embedding model those sentences are nearly identical.

Measured on this corpus the two agree remarkably little. Asked "How many
employees does Heineken have?", their top three results overlapped on nothing
at all before the keyword query was cleaned up. That is not a defect, it is
the point: two searches that agreed would be one search.

Fusion is by rank rather than by score, which matters. BM25 returns unbounded
negative numbers and vector search returns distances, and no amount of
normalising makes those two comparable. Their orderings are comparable, and
that is all reciprocal rank fusion needs.
"""

from __future__ import annotations

from collections.abc import Sequence

from app.db.models import Chunk, SearchHit
from app.db.repositories import ChunkRepository
from app.providers.base import EmbeddingProvider

# The constant from the original reciprocal rank fusion paper. Its job is to
# stop the top result of either list from dominating: at k=60 the difference
# between rank 1 and rank 2 is small, so a chunk needs to place well in both
# searches to beat one that placed first in only one of them. That is exactly
# the behaviour wanted here, because either search alone is confidently wrong
# often enough to matter.
RRF_K = 60

# How many results to pull from each search before fusing. Larger than the
# final limit on purpose: a chunk ranked twelfth by both searches is a better
# answer than one ranked second by a single search, and it cannot win if it was
# never in the running.
DEFAULT_CANDIDATES = 40

# Extra breadth when the results are being narrowed to one document. A vec0
# table stores nothing but ids and vectors, so nearest neighbour search cannot
# be told to stay inside one report and the filtering has to happen afterwards.
# With five reports indexed, asking for six times as many candidates makes it
# very unlikely that filtering leaves the list short.
DOCUMENT_OVERFETCH = 6


def fuse(rankings: Sequence[Sequence[int]], k: int = RRF_K) -> list[int]:
    """Merge several ranked lists of chunk ids into one.

    Reciprocal rank fusion: every list awards each chunk 1/(k + position), and
    the scores are summed. A chunk appearing halfway down both lists beats one
    sitting at the top of a single list, which is the property that makes a
    confidently wrong search survivable.

    Args:
        rankings: One sequence of chunk ids per search, best first.
        k: Damping constant. Larger values flatten the advantage of being
            first, making agreement between searches count for more.

    Returns:
        Chunk ids, best first, each appearing once.
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for position, chunk_id in enumerate(ranking):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + position + 1)
    return sorted(scores, key=lambda chunk_id: -scores[chunk_id])


def interleave(rankings: Sequence[Sequence[int]], limit: int) -> list[int]:
    """Merge ranked lists by taking turns rather than by voting.

    The counterpart to fuse(), for a different situation, and mixing the two up
    costs real answers.

    fuse() is right when several methods are answering the same question:
    keyword and vector search over one query. Agreement between them is
    evidence, so summing their scores is exactly what you want.

    This is right when several queries are alternative ways of asking. There,
    summing punishes the specific. Searching an ABN AMRO report with four
    phrasings of "how many employees", the chunk containing the actual group
    figure ranked 4th for one phrasing and nowhere for the other three, while
    four departmental breakdowns ranked mid-table for all four. Fused, the
    breakdowns buried the answer: it fell from rank 4 to rank 38 and never
    reached the model. Taking turns guarantees every phrasing gets its best
    results in front of the model.

    Args:
        rankings: One ranked list of chunk ids per query, best first.
        limit: How many ids to return in total.

    Returns:
        Chunk ids, each appearing once, ordered by how well they placed within
        their own query rather than across all of them.
    """
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
    """Find the chunks most likely to answer a question.

    Args:
        chunks: The chunk repository.
        embeddings: The embedding provider, for the dense half.
        query: The question, in whatever words the user used.
        limit: How many chunks to return.
        document_id: Restrict to one report, or None to search everything.
        candidates: How many results to take from each search before fusing.

    Returns:
        Chunks, best first. Fewer than limit if the index holds fewer matches,
        and empty if the query had nothing searchable in it.
    """
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
    """Search several phrasings of the same information need and merge them.

    Extraction uses this. "How many people work here" is asked in a report as
    "average number of employees (FTE)", "total workforce", or a row in a table
    with no sentence around it at all, and no single query phrasing finds all
    three.

    Each query is searched properly on its own, keyword and vector fused
    together, and the resulting lists then take turns. See interleave() for why
    they take turns instead of being fused as well: doing it the other way
    round measurably buries the specific answer under generic near misses.

    Args:
        chunks: The chunk repository.
        embeddings: The embedding provider.
        queries: Different phrasings of the same need.
        limit: How many chunks to return in total. Each query gets roughly an
            equal share, so this is worth setting to several times the number
            of queries.
        document_id: Restrict to one report, or None for everything.
        candidates: How many results to take from each individual search.

    Returns:
        Chunks, best first, each appearing once.
    """
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
