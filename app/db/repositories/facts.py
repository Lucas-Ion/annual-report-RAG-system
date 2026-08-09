"""Storage for datapoints extracted at ingest time.

These are what the brief means by pre-extracted data being visible in the
application: the FTE count and the sustainability goals are pulled out when a
report is ingested and simply read back when somebody opens the page, so the
overview never waits on a model call.
"""

from __future__ import annotations

import sqlite3

from app.db.models import Fact
from app.db.repositories.base import Repository

_COLUMNS = (
    "id, document_id, field_key, value_raw, value_numeric, unit, "
    "verbatim_quote, page_no, chunk_id, confidence, extractor_version, created_at"
)


def to_fact(row: sqlite3.Row) -> Fact:
    """Build a Fact from a database row.

    Args:
        row: A row selected with _COLUMNS.

    Returns:
        The equivalent domain object.
    """
    return Fact(
        id=row["id"],
        document_id=row["document_id"],
        field_key=row["field_key"],
        value_raw=row["value_raw"],
        value_numeric=row["value_numeric"],
        unit=row["unit"],
        verbatim_quote=row["verbatim_quote"],
        page_no=row["page_no"],
        chunk_id=row["chunk_id"],
        confidence=row["confidence"],
        extractor_version=row["extractor_version"],
        created_at=row["created_at"],
    )


class FactRepository(Repository[Fact, int]):
    """Datapoints read out of reports ahead of time.

    Note what this repository does not do: it never checks that a quote really
    appears in its chunk. That check belongs to the extraction stage, which
    has the chunk in hand and can reject a bad extraction before it ever
    reaches storage. A repository that silently dropped rows it disapproved of
    would be a much harder thing to debug.
    """

    def read(self) -> list[Fact]:
        """Return every stored fact.

        Returns:
            All facts, grouped by document and field.
        """
        rows = self._conn.execute(
            f"""
            SELECT {_COLUMNS} FROM extracted_facts
             ORDER BY document_id, field_key, id
            """
        ).fetchall()
        return [to_fact(row) for row in rows]

    def read_by_id(self, entity_id: int) -> Fact | None:
        """Look up one fact by row id.

        Args:
            entity_id: The fact's id.

        Returns:
            The fact, or None if there is no such row.
        """
        row = self._conn.execute(
            f"SELECT {_COLUMNS} FROM extracted_facts WHERE id = ?", (entity_id,)
        ).fetchone()
        return to_fact(row) if row else None

    def read_for_document(self, document_id: int) -> list[Fact]:
        """Return everything extracted from one report.

        This backs the per document detail view.

        Args:
            document_id: The document to read.

        Returns:
            Its facts, grouped by field. Expect several rows for the same
            field: a company has one FTE figure but usually a handful of
            sustainability goals, which is why the schema deliberately allows
            repeats rather than enforcing one row per field.
        """
        rows = self._conn.execute(
            f"""
            SELECT {_COLUMNS} FROM extracted_facts
             WHERE document_id = ?
             ORDER BY field_key, id
            """,
            (document_id,),
        ).fetchall()
        return [to_fact(row) for row in rows]

    def read_by_field(self, field_key: str) -> list[Fact]:
        """Return one field across every report.

        The comparison view. Asking for "fte" gives you the headcount of all
        five companies side by side, which is the kind of thing the extraction
        table exists to make cheap.

        Args:
            field_key: The field to pull, matching the extraction registry.

        Returns:
            Every stored value of that field, ordered by document.
        """
        rows = self._conn.execute(
            f"""
            SELECT {_COLUMNS} FROM extracted_facts
             WHERE field_key = ?
             ORDER BY document_id, id
            """,
            (field_key,),
        ).fetchall()
        return [to_fact(row) for row in rows]

    def create(self, entity: Fact) -> Fact:
        """Store one extracted fact.

        Args:
            entity: The fact to store. Its id should be None, and its
                verbatim_quote should already have been checked against the
                chunk it came from.

        Returns:
            The fact with its id and created_at filled in.
        """
        new_id = self._insert(
            """
            INSERT INTO extracted_facts (
                document_id, field_key, value_raw, value_numeric, unit,
                verbatim_quote, page_no, chunk_id, confidence, extractor_version
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entity.document_id,
                entity.field_key,
                entity.value_raw,
                entity.value_numeric,
                entity.unit,
                entity.verbatim_quote,
                entity.page_no,
                entity.chunk_id,
                entity.confidence,
                entity.extractor_version,
            ),
        )
        created = self.read_by_id(new_id)
        assert created is not None  # just inserted, inside the same transaction
        return created

    def update(self, entity: Fact) -> Fact:
        """Overwrite a stored fact.

        Args:
            entity: The fact to store, with its id set.

        Returns:
            The entity as given.

        Raises:
            ValueError: If the fact has no id.
        """
        if entity.id is None:
            raise ValueError("cannot update a fact that has not been created")
        self._conn.execute(
            """
            UPDATE extracted_facts
               SET document_id = ?, field_key = ?, value_raw = ?, value_numeric = ?,
                   unit = ?, verbatim_quote = ?, page_no = ?, chunk_id = ?,
                   confidence = ?, extractor_version = ?
             WHERE id = ?
            """,
            (
                entity.document_id,
                entity.field_key,
                entity.value_raw,
                entity.value_numeric,
                entity.unit,
                entity.verbatim_quote,
                entity.page_no,
                entity.chunk_id,
                entity.confidence,
                entity.extractor_version,
                entity.id,
            ),
        )
        return entity

    def delete(self, entity: Fact) -> Fact:
        """Remove one fact.

        Args:
            entity: The fact to remove, with its id set.

        Returns:
            The fact that was removed.

        Raises:
            ValueError: If the fact has no id.
        """
        if entity.id is None:
            raise ValueError("cannot delete a fact that has not been created")
        self._conn.execute("DELETE FROM extracted_facts WHERE id = ?", (entity.id,))
        return entity

    def delete_for_document(self, document_id: int) -> int:
        """Remove every fact extracted from one report.

        Called at the start of a re-extraction. Clearing first rather than
        relying on extractor_version keeps the table honest: a field that used
        to be extracted and no longer is disappears, instead of lingering as a
        stale row that the interface would happily still display.

        Args:
            document_id: The document to clear.

        Returns:
            How many facts were removed.
        """
        cursor = self._conn.execute(
            "DELETE FROM extracted_facts WHERE document_id = ?", (document_id,)
        )
        return cursor.rowcount
