"""The contract every repository in this package implements.

The pattern buys two things. First, an abstraction over storage: callers say
what they want ("the chunks for this document") and never how it is kept, so
moving off SQLite would mean rewriting this package and nothing above it.
Second, one home per kind of object, which is what stops the alternative from
happening: a single sprawling data access class that everything imports and
nobody wants to touch.

Three deliberate departures from the textbook version, each with a reason:

  * No generic Criteria object. A criteria abstraction general enough to
    express both a keyword search and a nearest vector search ends up leaking
    the storage engine straight back into the caller, which defeats the point
    of having the abstraction. Concrete repositories add plainly named read
    methods instead, so ChunkRepository offers read_by_similarity() rather
    than read(SomeOpaqueCriteria(...)). Less general, far easier to use, and
    the type checker can actually help you.

  * create_all() sits on the base class. The textbook shape writes one entity
    at a time, which is fine for an article but not for an ingest that inserts
    blocks by the ten thousand.

  * Repositories are not thread safe and are not meant to be long lived. A
    SQLite connection belongs to the thread that opened it, so the web layer
    builds these per request, uses them, and drops them. Constructing one is
    an attribute assignment, so this costs nothing.
"""

from __future__ import annotations

import sqlite3
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence


class Repository[T, K](ABC):
    """A collection of one kind of domain object, backed by the database.

    The two type parameters are the same idea as the article's
    Repository<T, K>: T is the domain object being stored, K is the type of
    its identity. Most repositories here are Repository[Something, int],
    because most identities are row ids.

    Subclasses implement the five methods below against real SQL. They are
    free to add domain specific read methods on top, and most of them do.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        """Bind the repository to an open connection.

        The connection is borrowed, never owned. Opening it, committing it and
        closing it stay the caller's business, and that is what allows several
        repositories to take part in a single transaction: hand the same
        connection to all of them and wrap the lot in one with block.

        Args:
            conn: An open connection from db.connection.connect(). It must
                have come from there, because a raw sqlite3.connect() has
                foreign keys disabled and no vector support.
        """
        self._conn = conn

    @abstractmethod
    def read(self) -> list[T]:
        """Return every entity in the collection.

        Returns:
            All entities, in whatever order the concrete repository considers
            natural for the type.
        """

    @abstractmethod
    def read_by_id(self, entity_id: K) -> T | None:
        """Look up a single entity by its identity.

        Returning None rather than raising is deliberate. A missing row is an
        ordinary outcome in this application (a stale link, a document that
        was deleted while a page was open), not an exceptional one, and
        forcing every caller into a try block for the normal case makes the
        code worse.

        Args:
            entity_id: The identity to look for.

        Returns:
            The entity, or None if nothing has that identity.
        """

    @abstractmethod
    def create(self, entity: T) -> T:
        """Write a new entity.

        Args:
            entity: The object to store. Its id is expected to be None.

        Returns:
            A copy of the entity with its identity and any database assigned
            defaults filled in. The argument itself is left untouched, so a
            caller holding the original is never surprised by it changing
            underneath them.
        """

    @abstractmethod
    def update(self, entity: T) -> T:
        """Overwrite the stored entity with this one.

        Args:
            entity: The object to store, with its identity already set.

        Returns:
            The entity, for symmetry with create().

        Raises:
            ValueError: If the entity has no identity, since there would be
                nothing to overwrite.
        """

    @abstractmethod
    def delete(self, entity: T) -> T:
        """Remove the entity.

        Args:
            entity: The object to remove, with its identity already set.

        Returns:
            The object that was removed, so a caller can log or report on it
            without having to hold a second reference.

        Raises:
            ValueError: If the entity has no identity.
        """

    def create_all(self, entities: Iterable[T]) -> list[T]:
        """Write many entities, in the order given.

        The default is a plain loop over create(), which is correct everywhere
        but issues one statement per row. Repositories that insert at real
        volume override this with executemany. Note that nothing here commits:
        wrap the call in db.connection.transaction() so a failure halfway
        through leaves no half written document behind.

        Args:
            entities: The objects to store.

        Returns:
            The stored entities with their identities filled in, in the same
            order they were given.
        """
        return [self.create(entity) for entity in entities]

    def _insert(self, sql: str, values: Sequence[object]) -> int:
        """Run an INSERT and return the row id SQLite assigned to it.

        This exists to deal with one awkward corner of the sqlite3 API.
        cursor.lastrowid is typed as int or None, because it really is None
        after a statement that inserted nothing. Every caller in this package
        has just run an INSERT, so that case cannot arise, but a type checker
        has no way of knowing and will flag the next line of every create()
        method in the package.

        Handling it once here is better than seven copies of the same
        assertion, and turning the impossible case into a loud error is better
        than quietly passing None into a lookup that would then return nothing.

        Args:
            sql: An INSERT statement using ? placeholders.
            values: The values for those placeholders.

        Returns:
            The id of the row just inserted.

        Raises:
            RuntimeError: If SQLite reported no row id, which would mean the
                statement was not an INSERT at all.
        """
        cursor = self._conn.execute(sql, values)
        if cursor.lastrowid is None:
            raise RuntimeError(f"statement produced no row id: {sql.strip()[:60]}")
        return cursor.lastrowid
