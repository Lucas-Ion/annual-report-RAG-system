"""A contract every repository in this package implements."""

from __future__ import annotations

import sqlite3
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence


class Repository[T, K](ABC):
    """T is the domain object, K is the type of its identity.

    StageRunRepository is an exception however, 
    since a stage run is identified by (document_id, stage).
    """
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    @abstractmethod
    def read(self) -> list[T]:
        """Every entity, in whatever order suits the type."""

    @abstractmethod
    def read_by_id(self, entity_id: K) -> T | None:
        """Look up one entity, or None if nothing has that identity."""

    @abstractmethod
    def create(self, entity: T) -> T:
        """Store a new entity, returning it with its id and defaults filled in."""

    @abstractmethod
    def update(self, entity: T) -> T:
        """Overwrite the stored entity."""

    @abstractmethod
    def delete(self, entity: T) -> T:
        """Remove the entity and hand it back, so a caller can report on it."""

    def create_all(self, entities: Iterable[T]) -> list[T]:
        """Store many entities, in order."""

        return [self.create(entity) for entity in entities]

    def _insert(self, sql: str, values: Sequence[object]) -> int:
        """Run an INSERT and return the row id SQLite assigned to it."""
        cursor = self._conn.execute(sql, values)
        if cursor.lastrowid is None:
            raise RuntimeError(f"statement produced no row id: {sql.strip()[:60]}")
        return cursor.lastrowid
