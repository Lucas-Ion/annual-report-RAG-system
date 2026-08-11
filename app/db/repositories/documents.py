"""Storage for ingested reports."""

from __future__ import annotations

import sqlite3

from app.db.models import Document
from app.db.repositories.base import Repository

_COLUMNS = "id, filename, file_hash, company, year, page_count, created_at"


def to_document(row: sqlite3.Row) -> Document:
    return Document(
        id=row["id"],
        filename=row["filename"],
        file_hash=row["file_hash"],
        company=row["company"],
        year=row["year"],
        page_count=row["page_count"],
        created_at=row["created_at"],
    )


class DocumentRepository(Repository[Document, int]):
    def read(self) -> list[Document]:
        rows = self._conn.execute(
            f"SELECT {_COLUMNS} FROM documents ORDER BY created_at DESC, id DESC"
        ).fetchall()
        return [to_document(row) for row in rows]

    def read_by_id(self, entity_id: int) -> Document | None:
        row = self._conn.execute(
            f"SELECT {_COLUMNS} FROM documents WHERE id = ?", (entity_id,)
        ).fetchone()
        return to_document(row) if row else None

    def read_by_hash(self, file_hash: str) -> Document | None:
        row = self._conn.execute(
            f"SELECT {_COLUMNS} FROM documents WHERE file_hash = ?", (file_hash,)
        ).fetchone()
        return to_document(row) if row else None

    def read_by_company(self, company: str) -> list[Document]:
        rows = self._conn.execute(
            f"SELECT {_COLUMNS} FROM documents WHERE company = ? ORDER BY year",
            (company,),
        ).fetchall()
        return [to_document(row) for row in rows]

    def create(self, entity: Document) -> Document:
        new_id = self._insert(
            """
            INSERT INTO documents (filename, file_hash, company, year, page_count)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                entity.filename,
                entity.file_hash,
                entity.company,
                entity.year,
                entity.page_count,
            ),
        )
        created = self.read_by_id(new_id)
        assert created is not None  # just inserted, inside the same transaction
        return created

    def update(self, entity: Document) -> Document:
        if entity.id is None:
            raise ValueError("cannot update a document that has not been created")
        self._conn.execute(
            """
            UPDATE documents
               SET filename = ?, file_hash = ?, company = ?, year = ?, page_count = ?
             WHERE id = ?
            """,
            (
                entity.filename,
                entity.file_hash,
                entity.company,
                entity.year,
                entity.page_count,
                entity.id,
            ),
        )
        return entity

    def delete(self, entity: Document) -> Document:
        if entity.id is None:
            raise ValueError("cannot delete a document that has not been created")
        self._conn.execute(
            """
            DELETE FROM chunk_vectors
             WHERE chunk_id IN (SELECT id FROM chunks WHERE document_id = ?)
            """,
            (entity.id,),
        )
        self._conn.execute("DELETE FROM documents WHERE id = ?", (entity.id,))
        return entity
