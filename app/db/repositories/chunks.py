"""Storage and search for the retrieval unit.

This is the only repository that does more than move rows around, because
both halves of hybrid retrieval live here: keyword search through FTS5 and
nearest neighbour search through sqlite-vec. They stay in the repository
because both are pure SQL. Deciding how to combine their two result lists is
not, so that lives in app/retrieve/ instead.
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Iterable, Sequence

import sqlite_vec

from app.db.models import Chunk, ChunkType, SearchHit
from app.db.repositories.base import Repository

_FIELDS = (
    "id",
    "document_id",
    "seq",
    "page_start",
    "page_end",
    "section",
    "chunk_type",
    "context_header",
    "text",
    "token_count",
)
_COLUMNS = ", ".join(_FIELDS)
_JOINED_COLUMNS = ", ".join(f"c.{name}" for name in _FIELDS)

# FTS5 has its own little query language, so free text from a user is a syntax
# error waiting to happen: a stray double quote, a bare AND or NEAR, or an
# unbalanced bracket all raise mid query. Pulling out word characters and
# quoting each one turns any input into a plain list of literal terms, which is
# what somebody typing a question actually meant.
_WORD = re.compile(r"\w+", re.UNICODE)

# Dropped from keyword queries. Not an optimisation: with terms joined by OR,
# leaving these in actively breaks the search. Asking "How much did Shell spend
# on climate change adaptation in 2025?" with the full word list returns an ABN
# AMRO section titled "Snack or save?" as the top hit, because it happens to
# contain a lot of "how", "much" and "did". Removing them puts the right
# company's disclosures on top.
#
# Deliberately short, and question words rather than a general English stop
# list. An aggressive list starts discarding terms that carry real meaning in
# financial reporting, and a term wrongly dropped costs more than a common word
# wrongly kept.
_STOPWORDS = frozenset(
    """
    a about an and any are as at be been by can did do does for from had has
    have how i in into is it its many much of on or our that the their there
    these this to was we were what when where which who why will with would you
    your
    """.split()
)


def to_chunk(row: sqlite3.Row) -> Chunk:
    """Build a Chunk from a database row.

    Args:
        row: A row selected with _COLUMNS or _JOINED_COLUMNS.

    Returns:
        The equivalent domain object.
    """
    return Chunk(
        id=row["id"],
        document_id=row["document_id"],
        seq=row["seq"],
        page_start=row["page_start"],
        page_end=row["page_end"],
        section=row["section"],
        chunk_type=ChunkType(row["chunk_type"]),
        context_header=row["context_header"],
        text=row["text"],
        token_count=row["token_count"],
    )


def to_match_expression(text: str) -> str:
    """Turn free text into an FTS5 MATCH expression that cannot throw.

    Terms are joined with OR rather than AND on purpose. A question like "how
    much did Shell spend on climate change adaptation" has no chunk containing
    every one of those words, so AND would return nothing at all. OR returns
    anything sharing a term and lets BM25 do the ranking, which is exactly the
    job BM25 exists for: it weights rare words like "adaptation" far above
    common ones like "much".

    Question words are stripped first, for the reason set out on _STOPWORDS. A
    query made entirely of them falls back to using them anyway, since a search
    for something is more useful than a search for nothing.

    Args:
        text: Whatever the user typed.

    Returns:
        A safe MATCH expression, or an empty string if the input held no
        searchable words at all.
    """
    terms = _WORD.findall(text)
    meaningful = [term for term in terms if term.casefold() not in _STOPWORDS]
    return " OR ".join(f'"{term}"' for term in meaningful or terms)


class ChunkRepository(Repository[Chunk, int]):
    """The pieces of text that answers get retrieved from and quoted out of."""

    def read(self) -> list[Chunk]:
        """Return every chunk in the database.

        Returns:
            All chunks, ordered by document and then position.
        """
        rows = self._conn.execute(
            f"SELECT {_COLUMNS} FROM chunks ORDER BY document_id, seq"
        ).fetchall()
        return [to_chunk(row) for row in rows]

    def read_by_id(self, entity_id: int) -> Chunk | None:
        """Look up one chunk by row id.

        Args:
            entity_id: The chunk's id.

        Returns:
            The chunk, or None if there is no such row.
        """
        row = self._conn.execute(
            f"SELECT {_COLUMNS} FROM chunks WHERE id = ?", (entity_id,)
        ).fetchone()
        return to_chunk(row) if row else None

    def read_by_ids(self, chunk_ids: Sequence[int]) -> list[Chunk]:
        """Look up several chunks at once, preserving the order asked for.

        Retrieval produces a ranked list of ids and then needs the text behind
        them. SQL has no concept of the order an IN list was written in, so
        the rows are reordered in Python afterwards. Doing it any other way
        means either one query per id or a CASE expression built by string
        concatenation, and neither is worth it for a list of twenty.

        Args:
            chunk_ids: Ids to fetch, in the order they should come back.

        Returns:
            The chunks that exist, in the requested order. Ids with no
            matching row are quietly skipped rather than returned as None.
        """
        if not chunk_ids:
            return []
        placeholders = ", ".join("?" * len(chunk_ids))
        rows = self._conn.execute(
            f"SELECT {_COLUMNS} FROM chunks WHERE id IN ({placeholders})",
            tuple(chunk_ids),
        ).fetchall()
        by_id = {row["id"]: to_chunk(row) for row in rows}
        return [by_id[chunk_id] for chunk_id in chunk_ids if chunk_id in by_id]

    def read_for_document(self, document_id: int) -> list[Chunk]:
        """Return one document's chunks in order.

        Args:
            document_id: The document to read.

        Returns:
            Every chunk of that document, ordered by seq.
        """
        rows = self._conn.execute(
            f"SELECT {_COLUMNS} FROM chunks WHERE document_id = ? ORDER BY seq",
            (document_id,),
        ).fetchall()
        return [to_chunk(row) for row in rows]

    def read_without_embeddings(self, document_id: int) -> list[Chunk]:
        """Return chunks that have not been embedded yet.

        The resume point for the embed stage, in the same spirit as
        BlockRepository.last_parsed_page(). Embedding calls cost money and
        time, so a rerun should pick up only what is genuinely missing.

        Args:
            document_id: The document to check.

        Returns:
            That document's unembedded chunks, in order.
        """
        rows = self._conn.execute(
            f"""
            SELECT {_COLUMNS} FROM chunks
             WHERE document_id = ?
               AND id NOT IN (SELECT chunk_id FROM chunk_vectors)
             ORDER BY seq
            """,
            (document_id,),
        ).fetchall()
        return [to_chunk(row) for row in rows]

    def read_by_keywords(self, query: str, limit: int = 20) -> list[SearchHit]:
        """Search chunks by keyword, ranked by BM25.

        This is the half of retrieval that catches exact wording. Somebody
        asking about "Scope 3 emissions" wants the pages that literally say
        "Scope 3", and a purely semantic search will happily hand back
        something about Scope 1 instead because the two read almost alike.

        BM25 scores come out of SQLite negative, with more negative meaning a
        better match. That is genuinely how the extension reports it, and it
        catches people out constantly. The sign is flipped here so that the
        score obeys the same higher is better rule as every other search in
        this package.

        Args:
            query: Free text from the user. Sanitised before use, so anything
                is safe to pass.
            limit: How many results to return at most.

        Returns:
            Matching chunks, best first. Empty if the query held no searchable
            words.
        """
        expression = to_match_expression(query)
        if not expression:
            return []
        rows = self._conn.execute(
            f"""
            SELECT {_JOINED_COLUMNS}, bm25(chunks_fts) AS rank
              FROM chunks_fts
              JOIN chunks c ON c.id = chunks_fts.rowid
             WHERE chunks_fts MATCH ?
             ORDER BY rank
             LIMIT ?
            """,
            (expression, limit),
        ).fetchall()
        return [SearchHit(chunk=to_chunk(row), score=-row["rank"]) for row in rows]

    def read_by_similarity(
        self, embedding: Sequence[float], limit: int = 20
    ) -> list[SearchHit]:
        """Search chunks by meaning, ranked by vector distance.

        The half of retrieval that catches paraphrase. A report that never
        writes "employees" but talks throughout about "our workforce" is
        invisible to keyword search and obvious to this one.

        Note that there is no way to restrict this to a single document. A
        vec0 table stores nothing but ids and vectors, so its nearest
        neighbour search cannot be filtered by anything the chunks table
        knows. Scoping a question to one company therefore belongs in
        app/retrieve/, by asking for more results than are needed and
        discarding the ones from other reports.

        Args:
            embedding: The query embedded by the same model as the chunks, so
                exactly 1024 floats. A different length fails at the SQL layer
                rather than silently returning nonsense.
            limit: How many neighbours to return.

        Returns:
            The nearest chunks, closest first.
        """
        rows = self._conn.execute(
            f"""
            WITH nearest AS (
                SELECT chunk_id, distance
                  FROM chunk_vectors
                 WHERE embedding MATCH ?
                   AND k = ?
            )
            SELECT {_JOINED_COLUMNS}, nearest.distance AS distance
              FROM nearest
              JOIN chunks c ON c.id = nearest.chunk_id
             ORDER BY nearest.distance
            """,
            (sqlite_vec.serialize_float32(list(embedding)), limit),
        ).fetchall()
        return [SearchHit(chunk=to_chunk(row), score=-row["distance"]) for row in rows]

    def create(self, entity: Chunk) -> Chunk:
        """Store one chunk.

        The FTS5 index updates itself here, through the triggers declared in
        schema.sql. That is worth knowing because it means nothing in this
        file mentions chunks_fts on the write path, and forgetting to maintain
        the index is not a mistake anyone can make.

        Args:
            entity: The chunk to store. Its id should be None.

        Returns:
            The chunk with its id filled in, which the embed stage needs in
            order to attach a vector to it.
        """
        new_id = self._insert(
            """
            INSERT INTO chunks (
                document_id, seq, page_start, page_end, section,
                chunk_type, context_header, text, token_count
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entity.document_id,
                entity.seq,
                entity.page_start,
                entity.page_end,
                entity.section,
                str(entity.chunk_type),
                entity.context_header,
                entity.text,
                entity.token_count,
            ),
        )
        created = self.read_by_id(new_id)
        assert created is not None  # just inserted, inside the same transaction
        return created

    def update(self, entity: Chunk) -> Chunk:
        """Overwrite a stored chunk.

        Changing the text invalidates the chunk's embedding, since that was
        computed from the old wording. Nothing here enforces that, because a
        repository silently deleting a vector the caller did not mention would
        be worse than the problem. Callers that edit text should re-embed.

        Args:
            entity: The chunk to store, with its id set.

        Returns:
            The entity as given.

        Raises:
            ValueError: If the chunk has no id.
        """
        if entity.id is None:
            raise ValueError("cannot update a chunk that has not been created")
        self._conn.execute(
            """
            UPDATE chunks
               SET document_id = ?, seq = ?, page_start = ?, page_end = ?, section = ?,
                   chunk_type = ?, context_header = ?, text = ?, token_count = ?
             WHERE id = ?
            """,
            (
                entity.document_id,
                entity.seq,
                entity.page_start,
                entity.page_end,
                entity.section,
                str(entity.chunk_type),
                entity.context_header,
                entity.text,
                entity.token_count,
                entity.id,
            ),
        )
        return entity

    def delete(self, entity: Chunk) -> Chunk:
        """Remove one chunk and its embedding.

        The embedding has to be removed explicitly for the same reason it does
        in DocumentRepository.delete(): vec0 tables cannot declare foreign
        keys, so nothing cascades into chunk_vectors. The FTS5 index does look
        after itself, through a trigger.

        Args:
            entity: The chunk to remove, with its id set.

        Returns:
            The chunk that was removed.

        Raises:
            ValueError: If the chunk has no id.
        """
        if entity.id is None:
            raise ValueError("cannot delete a chunk that has not been created")
        self._conn.execute("DELETE FROM chunk_vectors WHERE chunk_id = ?", (entity.id,))
        self._conn.execute("DELETE FROM chunks WHERE id = ?", (entity.id,))
        return entity

    def delete_for_document(self, document_id: int) -> int:
        """Remove every chunk of one document, embeddings included.

        Used when rechunking. Blocks survive, so this is cheap: the expensive
        parse is not repeated, only the seconds of splitting that follow it.

        Args:
            document_id: The document to clear.

        Returns:
            How many chunks were removed.
        """
        self._conn.execute(
            """
            DELETE FROM chunk_vectors
             WHERE chunk_id IN (SELECT id FROM chunks WHERE document_id = ?)
            """,
            (document_id,),
        )
        cursor = self._conn.execute(
            "DELETE FROM chunks WHERE document_id = ?", (document_id,)
        )
        return cursor.rowcount

    def set_embedding(self, chunk_id: int, embedding: Sequence[float]) -> None:
        """Attach or replace a chunk's vector.

        Args:
            chunk_id: The chunk being embedded.
            embedding: Exactly 1024 floats from the embedding model.

        Raises:
            sqlite3.OperationalError: If the vector is not 1024 values long.
                The dimension is baked into the table, so swapping embedding
                models fails loudly on the first write rather than quietly
                poisoning the index.
        """
        self._conn.execute("DELETE FROM chunk_vectors WHERE chunk_id = ?", (chunk_id,))
        self._conn.execute(
            "INSERT INTO chunk_vectors (chunk_id, embedding) VALUES (?, ?)",
            (chunk_id, sqlite_vec.serialize_float32(list(embedding))),
        )

    def set_embeddings(self, pairs: Iterable[tuple[int, Sequence[float]]]) -> int:
        """Attach vectors to many chunks at once.

        Args:
            pairs: (chunk_id, embedding) tuples, usually straight from a batch
                call to the embedding model.

        Returns:
            How many vectors were written.
        """
        written = 0
        for chunk_id, embedding in pairs:
            self.set_embedding(chunk_id, embedding)
            written += 1
        return written
