"""Storage for the parser's output"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable

from app.db.models import Block
from app.db.repositories.base import Repository

_COLUMNS = "id, document_id, seq, page_no, label, level, text, bbox"
_INSERT = """
    INSERT INTO blocks (document_id, seq, page_no, label, level, text, bbox)
    VALUES (?, ?, ?, ?, ?, ?, ?)
"""


def to_block(row: sqlite3.Row) -> Block:
    bbox = json.loads(row["bbox"]) if row["bbox"] else None
    return Block(
        id=row["id"],
        document_id=row["document_id"],
        seq=row["seq"],
        page_no=row["page_no"],
        label=row["label"],
        level=row["level"],
        text=row["text"],
        bbox=tuple(bbox) if bbox else None,
    )


def _to_values(
    entity: Block,
) -> tuple[int, int, int, str, int | None, str, str | None]:
    return (
        entity.document_id,
        entity.seq,
        entity.page_no,
        entity.label,
        entity.level,
        entity.text,
        json.dumps(list(entity.bbox)) if entity.bbox else None,
    )


class BlockRepository(Repository[Block, int]):

    def read(self) -> list[Block]:
        rows = self._conn.execute(
            f"SELECT {_COLUMNS} FROM blocks ORDER BY document_id, seq"
        ).fetchall()
        return [to_block(row) for row in rows]

    def read_by_id(self, entity_id: int) -> Block | None:
        row = self._conn.execute(
            f"SELECT {_COLUMNS} FROM blocks WHERE id = ?", (entity_id,)
        ).fetchone()
        return to_block(row) if row else None

    def read_for_document(self, document_id: int) -> list[Block]:
        rows = self._conn.execute(
            f"SELECT {_COLUMNS} FROM blocks WHERE document_id = ? ORDER BY seq",
            (document_id,),
        ).fetchall()
        return [to_block(row) for row in rows]

    def read_page_range(self, document_id: int, first: int, last: int) -> list[Block]:
        rows = self._conn.execute(
            f"""
            SELECT {_COLUMNS} FROM blocks
             WHERE document_id = ? AND page_no BETWEEN ? AND ?
             ORDER BY seq
            """,
            (document_id, first, last),
        ).fetchall()
        return [to_block(row) for row in rows]

    def last_parsed_page(self, document_id: int) -> int | None:
        row = self._conn.execute(
            "SELECT MAX(page_no) AS last_page FROM blocks WHERE document_id = ?",
            (document_id,),
        ).fetchone()
        return row["last_page"]

    def next_seq(self, document_id: int) -> int:
        row = self._conn.execute(
            "SELECT MAX(seq) AS last_seq FROM blocks WHERE document_id = ?",
            (document_id,),
        ).fetchone()
        return 0 if row["last_seq"] is None else row["last_seq"] + 1

    def create(self, entity: Block) -> Block:
        new_id = self._insert(_INSERT, _to_values(entity))
        created = self.read_by_id(new_id)
        assert created is not None  # just inserted, inside the same transaction
        return created

    def create_all(self, entities: Iterable[Block]) -> list[Block]:
        stored = list(entities)
        self._conn.executemany(_INSERT, [_to_values(entity) for entity in stored])
        return stored

    def update(self, entity: Block) -> Block:
        if entity.id is None:
            raise ValueError("cannot update a block that has not been created")
        self._conn.execute(
            """
            UPDATE blocks
               SET document_id = ?, seq = ?, page_no = ?, label = ?,
                   level = ?, text = ?, bbox = ?
             WHERE id = ?
            """,
            (*_to_values(entity), entity.id),
        )
        return entity

    def delete(self, entity: Block) -> Block:
        if entity.id is None:
            raise ValueError("cannot delete a block that has not been created")
        self._conn.execute("DELETE FROM blocks WHERE id = ?", (entity.id,))
        return entity

    def delete_for_document(self, document_id: int) -> int:
        cursor = self._conn.execute(
            "DELETE FROM blocks WHERE document_id = ?", (document_id,)
        )
        return cursor.rowcount
