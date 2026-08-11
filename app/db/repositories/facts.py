"""Storage for datapoints extracted at ingest time"""

from __future__ import annotations

import sqlite3

from app.db.models import Fact
from app.db.repositories.base import Repository

_COLUMNS = (
    "id, document_id, field_key, value_raw, value_numeric, unit, "
    "verbatim_quote, page_no, chunk_id, confidence, extractor_version, created_at"
)


def to_fact(row: sqlite3.Row) -> Fact:
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
    def read(self) -> list[Fact]:
        rows = self._conn.execute(
            f"""
            SELECT {_COLUMNS} FROM extracted_facts
             ORDER BY document_id, field_key, id
            """
        ).fetchall()
        return [to_fact(row) for row in rows]

    def read_by_id(self, entity_id: int) -> Fact | None:
        row = self._conn.execute(
            f"SELECT {_COLUMNS} FROM extracted_facts WHERE id = ?", (entity_id,)
        ).fetchone()
        return to_fact(row) if row else None

    def read_for_document(self, document_id: int) -> list[Fact]:
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

        For example asking for "fte" gives you the headcount of all
        five companies side by side.

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
        if entity.id is None:
            raise ValueError("cannot delete a fact that has not been created")
        self._conn.execute("DELETE FROM extracted_facts WHERE id = ?", (entity.id,))
        return entity

    def delete_for_document(self, document_id: int) -> int:
        cursor = self._conn.execute(
            "DELETE FROM extracted_facts WHERE document_id = ?", (document_id,)
        )
        return cursor.rowcount
