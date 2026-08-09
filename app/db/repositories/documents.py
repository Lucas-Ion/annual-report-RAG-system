"""Storage for ingested reports."""

from __future__ import annotations

import sqlite3

from app.db.models import Document
from app.db.repositories.base import Repository

_COLUMNS = "id, filename, file_hash, company, year, page_count, created_at"


def to_document(row: sqlite3.Row) -> Document:
    """Build a Document from a database row.

    Args:
        row: A row selected with _COLUMNS.

    Returns:
        The equivalent domain object.
    """
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
    """The reports the system knows about."""

    def read(self) -> list[Document]:
        """Return every document, newest first.

        Newest first because this feeds the document list in the interface,
        and the report somebody just uploaded is the one they want to see.

        Returns:
            All documents.
        """
        rows = self._conn.execute(
            f"SELECT {_COLUMNS} FROM documents ORDER BY created_at DESC, id DESC"
        ).fetchall()
        return [to_document(row) for row in rows]

    def read_by_id(self, entity_id: int) -> Document | None:
        """Look up one document by row id.

        Args:
            entity_id: The document's id.

        Returns:
            The document, or None if there is no such row.
        """
        row = self._conn.execute(
            f"SELECT {_COLUMNS} FROM documents WHERE id = ?", (entity_id,)
        ).fetchone()
        return to_document(row) if row else None

    def read_by_hash(self, file_hash: str) -> Document | None:
        """Look up a document by the hash of its file contents.

        This is the duplicate check, and it is the reason ingest is safe to
        rerun. Hashing the bytes rather than trusting the filename means that
        "ABN_AMRO_2025.pdf" and "abn amro (1).pdf" are correctly recognised as
        the same report, and that an edited PDF with an unchanged name is
        correctly recognised as a different one.

        Args:
            file_hash: Hex sha256 of the PDF bytes.

        Returns:
            The existing document, or None if this file has not been seen.
        """
        row = self._conn.execute(
            f"SELECT {_COLUMNS} FROM documents WHERE file_hash = ?", (file_hash,)
        ).fetchone()
        return to_document(row) if row else None

    def read_by_company(self, company: str) -> list[Document]:
        """Return one company's reports, oldest year first.

        Ordered by year rather than by upload time, because the sensible thing
        to do with several years of the same company is compare them.

        Args:
            company: Company name, matched exactly.

        Returns:
            That company's documents.
        """
        rows = self._conn.execute(
            f"SELECT {_COLUMNS} FROM documents WHERE company = ? ORDER BY year",
            (company,),
        ).fetchall()
        return [to_document(row) for row in rows]

    def create(self, entity: Document) -> Document:
        """Register a new report.

        Args:
            entity: The document to store. Its id should be None.

        Returns:
            The document with its id and created_at filled in.

        Raises:
            sqlite3.IntegrityError: If a document with the same file_hash is
                already stored. Callers that want a quiet skip rather than an
                exception should check read_by_hash() first.
        """
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
        """Overwrite a stored document.

        In practice this is called once per ingest, by the parse stage, to
        record the page count it discovered.

        Args:
            entity: The document to store, with its id set.

        Returns:
            The entity as given.

        Raises:
            ValueError: If the document has no id.
        """
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
        """Remove a report and everything derived from it.

        Two statements, and the order matters.

        Foreign keys clean up blocks, chunks, facts and the full text index on
        their own, because those tables declare ON DELETE CASCADE and
        db.connection turns foreign key enforcement on. The vector table is
        the exception: sqlite-vec's vec0 tables cannot declare foreign keys at
        all, so its chunk_id column is a reference in name only and nothing
        cascades into it. Left alone, deleting a report would strand thousands
        of embeddings pointing at chunk ids that no longer exist, and the next
        search would happily return them.

        So the vectors go first, while the chunks that identify them still
        exist. Delete the document first and the subquery finds nothing, the
        statement succeeds, and the orphans are created silently. That failure
        leaves no trace, which is exactly why it is worth this many lines of
        comment.

        Args:
            entity: The document to remove, with its id set.

        Returns:
            The document that was removed.

        Raises:
            ValueError: If the document has no id.
        """
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
