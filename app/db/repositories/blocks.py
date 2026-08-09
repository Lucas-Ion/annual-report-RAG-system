"""Storage for the parser's output.

Blocks are the most expensive thing this system produces, at roughly 5.5
seconds of layout analysis per page, and the highest volume table by a wide
margin. Both facts shape this file: the bulk insert is overridden for speed,
and there is a method whose only job is to let an interrupted parse work out
where to resume.
"""

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
    """Build a Block from a database row.

    Args:
        row: A row selected with _COLUMNS.

    Returns:
        The equivalent domain object, with bbox decoded back into a tuple.
    """
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
    """Flatten a Block into the tuple an INSERT expects.

    SQLite has no array type, so the bounding box is stored as a JSON string.
    That is a fair trade here: nothing ever queries on a coordinate, the box
    is only ever read back whole, and the alternative of four separate columns
    would clutter the table for no benefit.

    Args:
        entity: The block to flatten.

    Returns:
        Its column values, in the order the INSERT lists them.
    """
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
    """Raw parser output, one row per item, in reading order."""

    def read(self) -> list[Block]:
        """Return every block in the database.

        Present because the base contract asks for it, but be careful: five
        annual reports come to tens of thousands of rows. Almost every real
        caller wants read_for_document() instead.

        Returns:
            All blocks, ordered by document and then reading order.
        """
        rows = self._conn.execute(
            f"SELECT {_COLUMNS} FROM blocks ORDER BY document_id, seq"
        ).fetchall()
        return [to_block(row) for row in rows]

    def read_by_id(self, entity_id: int) -> Block | None:
        """Look up one block by row id.

        Args:
            entity_id: The block's id.

        Returns:
            The block, or None if there is no such row.
        """
        row = self._conn.execute(
            f"SELECT {_COLUMNS} FROM blocks WHERE id = ?", (entity_id,)
        ).fetchone()
        return to_block(row) if row else None

    def read_for_document(self, document_id: int) -> list[Block]:
        """Return one document's blocks in reading order.

        This is what the chunking stage consumes. Reading order matters more
        than it might sound: chunking works by walking forward and cutting at
        headings, so a shuffled list would produce chunks that mix unrelated
        sections together.

        Args:
            document_id: The document to read.

        Returns:
            Every block of that document, ordered by seq.
        """
        rows = self._conn.execute(
            f"SELECT {_COLUMNS} FROM blocks WHERE document_id = ? ORDER BY seq",
            (document_id,),
        ).fetchall()
        return [to_block(row) for row in rows]

    def read_page_range(self, document_id: int, first: int, last: int) -> list[Block]:
        """Return the blocks on a span of pages, inclusive at both ends.

        Args:
            document_id: The document to read.
            first: First page number, counting from 1.
            last: Last page number, included in the result.

        Returns:
            Blocks on those pages, in reading order.
        """
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
        """Report the highest page number already stored for a document.

        This is the resume point. Parsing writes in batches of pages rather
        than all at once, so a run killed at page 380 of 434 has 380 pages of
        work safely on disk. Asking this question first turns what would be a
        45 minute restart into a 6 minute finish.

        Args:
            document_id: The document being parsed.

        Returns:
            The last page with at least one block stored, or None if nothing
            has been parsed yet. Note that the batch containing this page is
            the one to redo, since it may have been committed only partly.

        """
        row = self._conn.execute(
            "SELECT MAX(page_no) AS last_page FROM blocks WHERE document_id = ?",
            (document_id,),
        ).fetchone()
        return row["last_page"]

    def next_seq(self, document_id: int) -> int:
        """Report the next free position in a document's reading order.

        Needed for the same reason as last_parsed_page(): a resumed parse has
        to carry on numbering where the previous run stopped, or it collides
        with the UNIQUE (document_id, seq) constraint.

        Args:
            document_id: The document being parsed.

        Returns:
            One past the highest seq stored, or 0 for a document with no
            blocks yet.
        """
        row = self._conn.execute(
            "SELECT MAX(seq) AS last_seq FROM blocks WHERE document_id = ?",
            (document_id,),
        ).fetchone()
        return 0 if row["last_seq"] is None else row["last_seq"] + 1

    def create(self, entity: Block) -> Block:
        """Store one block.

        Args:
            entity: The block to store. Its id should be None.

        Returns:
            The block with its id filled in.
        """
        new_id = self._insert(_INSERT, _to_values(entity))
        created = self.read_by_id(new_id)
        assert created is not None  # just inserted, inside the same transaction
        return created

    def create_all(self, entities: Iterable[Block]) -> list[Block]:
        """Store many blocks in one round trip.

        Overridden because this is the highest volume insert in the system and
        the base class implementation would read every row back individually
        to collect its id. Nothing needs a block's id, so executemany is used
        and the ids are simply not returned.

        Args:
            entities: The blocks to store, already in reading order.

        Returns:
            The blocks as given, with ids still None. This is the one place in
            the package that deliberately breaks the "returns the stored
            entity" convention, and it is worth the inconsistency: filling
            those ids in would mean a second query over tens of thousands of
            rows to populate a field nothing reads.
        """
        # Materialised up front because entities may be a generator, and a
        # generator can only be walked once. Building the values would consume
        # it and leave nothing to return.
        stored = list(entities)
        self._conn.executemany(_INSERT, [_to_values(entity) for entity in stored])
        return stored

    def update(self, entity: Block) -> Block:
        """Overwrite a stored block.

        Rarely used. Blocks are a faithful record of what the parser produced,
        so editing one usually means the right fix is upstream, in parsing.

        Args:
            entity: The block to store, with its id set.

        Returns:
            The entity as given.

        Raises:
            ValueError: If the block has no id.
        """
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
        """Remove one block.

        Args:
            entity: The block to remove, with its id set.

        Returns:
            The block that was removed.

        Raises:
            ValueError: If the block has no id.
        """
        if entity.id is None:
            raise ValueError("cannot delete a block that has not been created")
        self._conn.execute("DELETE FROM blocks WHERE id = ?", (entity.id,))
        return entity

    def delete_for_document(self, document_id: int) -> int:
        """Remove every block of one document.

        Used when a parse is being redone from scratch rather than resumed,
        for instance after changing parser settings.

        Args:
            document_id: The document to clear.

        Returns:
            How many blocks were removed.
        """
        cursor = self._conn.execute(
            "DELETE FROM blocks WHERE document_id = ?", (document_id,)
        )
        return cursor.rowcount
