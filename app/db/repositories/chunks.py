"""Storage and search for the retrieval unit"""

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
_WORD = re.compile(r"\w+", re.UNICODE)
_STOPWORDS = frozenset(
    """
    a about an and any are as at be been by can did do does for from had has
    have how i in into is it its many much of on or our that the their there
    these this to was we were what when where which who why will with would you
    your
    """.split()
)


def to_chunk(row: sqlite3.Row) -> Chunk:
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
    """Turn free text into an FTS5 MATCH expression 

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
    def read(self) -> list[Chunk]:
        rows = self._conn.execute(
            f"SELECT {_COLUMNS} FROM chunks ORDER BY document_id, seq"
        ).fetchall()
        return [to_chunk(row) for row in rows]

    def read_by_id(self, entity_id: int) -> Chunk | None:
        row = self._conn.execute(
            f"SELECT {_COLUMNS} FROM chunks WHERE id = ?", (entity_id,)
        ).fetchone()
        return to_chunk(row) if row else None

    def read_by_ids(self, chunk_ids: Sequence[int]) -> list[Chunk]:
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
        rows = self._conn.execute(
            f"SELECT {_COLUMNS} FROM chunks WHERE document_id = ? ORDER BY seq",
            (document_id,),
        ).fetchall()
        return [to_chunk(row) for row in rows]

    def read_without_embeddings(self, document_id: int) -> list[Chunk]:
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
        """This function searches chunks by keyword, ranked by BM25.

        Args:
            query: Free text from the user.
            limit: How many results to return at most.

        Returns:
            Matching chunks, best first (most negative). Empty if the query
            held no searchable words.
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

        Args:
            embedding: The query embedded by the same model as the chunks, so
                exactly 1024 floats.
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
        if entity.id is None:
            raise ValueError("cannot delete a chunk that has not been created")
        self._conn.execute("DELETE FROM chunk_vectors WHERE chunk_id = ?", (entity.id,))
        self._conn.execute("DELETE FROM chunks WHERE id = ?", (entity.id,))
        return entity

    def delete_for_document(self, document_id: int) -> int:
        """Remove every chunk of one document, embeddings included.

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
        self._conn.execute("DELETE FROM chunk_vectors WHERE chunk_id = ?", (chunk_id,))
        self._conn.execute(
            "INSERT INTO chunk_vectors (chunk_id, embedding) VALUES (?, ?)",
            (chunk_id, sqlite_vec.serialize_float32(list(embedding))),
        )

    def set_embeddings(self, pairs: Iterable[tuple[int, Sequence[float]]]) -> int:
        written = 0
        for chunk_id, embedding in pairs:
            self.set_embedding(chunk_id, embedding)
            written += 1
        return written
