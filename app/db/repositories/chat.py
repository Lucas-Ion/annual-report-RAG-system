"""Storage for conversations, their messages, and the sources behind answers.

Three small repositories rather than one, following the same rule as the rest
of the package: one class per kind of object. Citations could arguably have
been folded into messages, since a citation only ever exists as part of an
answer, but they are read on their own often enough (rendering the sources
panel, counting how many answers were verifiable) that a separate class stays
simpler than a message repository with two entity types inside it.
"""

from __future__ import annotations

import sqlite3

from app.db.models import Citation, Conversation, Message, Role
from app.db.repositories.base import Repository

_CONVERSATION_COLUMNS = "id, title, created_at"
_MESSAGE_COLUMNS = "id, conversation_id, role, content, created_at"
_CITATION_COLUMNS = "id, message_id, chunk_id, quote, page_no, verified"


def to_conversation(row: sqlite3.Row) -> Conversation:
    """Build a Conversation from a database row.

    Args:
        row: A row selected with _CONVERSATION_COLUMNS.

    Returns:
        The equivalent domain object.
    """
    return Conversation(id=row["id"], title=row["title"], created_at=row["created_at"])


def to_message(row: sqlite3.Row) -> Message:
    """Build a Message from a database row.

    Args:
        row: A row selected with _MESSAGE_COLUMNS.

    Returns:
        The equivalent domain object.
    """
    return Message(
        id=row["id"],
        conversation_id=row["conversation_id"],
        role=Role(row["role"]),
        content=row["content"],
        created_at=row["created_at"],
    )


def to_citation(row: sqlite3.Row) -> Citation:
    """Build a Citation from a database row.

    SQLite has no boolean type, so verified is stored as 0 or 1 and converted
    back here. Doing it in one place is the reason no caller ever has to
    remember that.

    Args:
        row: A row selected with _CITATION_COLUMNS.

    Returns:
        The equivalent domain object.
    """
    return Citation(
        id=row["id"],
        message_id=row["message_id"],
        chunk_id=row["chunk_id"],
        quote=row["quote"],
        page_no=row["page_no"],
        verified=bool(row["verified"]),
    )


class ConversationRepository(Repository[Conversation, int]):
    """Chat threads."""

    def read(self) -> list[Conversation]:
        """Return every conversation, most recent first.

        Returns:
            All conversations, in the order a sidebar would list them.
        """
        rows = self._conn.execute(
            f"SELECT {_CONVERSATION_COLUMNS} FROM conversations ORDER BY id DESC"
        ).fetchall()
        return [to_conversation(row) for row in rows]

    def read_by_id(self, entity_id: int) -> Conversation | None:
        """Look up one conversation.

        Args:
            entity_id: The conversation's id.

        Returns:
            The conversation, or None if there is no such row.
        """
        row = self._conn.execute(
            f"SELECT {_CONVERSATION_COLUMNS} FROM conversations WHERE id = ?",
            (entity_id,),
        ).fetchone()
        return to_conversation(row) if row else None

    def create(self, entity: Conversation) -> Conversation:
        """Start a new thread.

        Args:
            entity: The conversation to store. A title is optional, since one
                is usually only worth generating after the first question has
                been asked.

        Returns:
            The conversation with its id and created_at filled in.
        """
        new_id = self._insert(
            "INSERT INTO conversations (title) VALUES (?)", (entity.title,)
        )
        created = self.read_by_id(new_id)
        assert created is not None  # just inserted, inside the same transaction
        return created

    def update(self, entity: Conversation) -> Conversation:
        """Overwrite a stored conversation, in practice to set its title.

        Args:
            entity: The conversation to store, with its id set.

        Returns:
            The entity as given.

        Raises:
            ValueError: If the conversation has no id.
        """
        if entity.id is None:
            raise ValueError("cannot update a conversation that has not been created")
        self._conn.execute(
            "UPDATE conversations SET title = ? WHERE id = ?", (entity.title, entity.id)
        )
        return entity

    def delete(self, entity: Conversation) -> Conversation:
        """Remove a thread and everything in it.

        Messages and their citations go too, by cascade. Nothing manual is
        needed here, because unlike the vector table these are all ordinary
        tables with real foreign keys.

        Args:
            entity: The conversation to remove, with its id set.

        Returns:
            The conversation that was removed.

        Raises:
            ValueError: If the conversation has no id.
        """
        if entity.id is None:
            raise ValueError("cannot delete a conversation that has not been created")
        self._conn.execute("DELETE FROM conversations WHERE id = ?", (entity.id,))
        return entity


class MessageRepository(Repository[Message, int]):
    """Individual turns within threads."""

    def read(self) -> list[Message]:
        """Return every message in every conversation.

        Returns:
            All messages, in conversation and then chronological order.
        """
        rows = self._conn.execute(
            f"SELECT {_MESSAGE_COLUMNS} FROM messages ORDER BY conversation_id, id"
        ).fetchall()
        return [to_message(row) for row in rows]

    def read_by_id(self, entity_id: int) -> Message | None:
        """Look up one message.

        Args:
            entity_id: The message's id.

        Returns:
            The message, or None if there is no such row.
        """
        row = self._conn.execute(
            f"SELECT {_MESSAGE_COLUMNS} FROM messages WHERE id = ?", (entity_id,)
        ).fetchone()
        return to_message(row) if row else None

    def read_for_conversation(self, conversation_id: int) -> list[Message]:
        """Return one thread's messages, oldest first.

        Ordered by id rather than by created_at. The timestamps come from
        datetime('now'), which only has a resolution of one second, so a
        question and its answer can easily share one. Row id is monotonic and
        never ties, which makes it the more reliable clock here.

        Args:
            conversation_id: The thread to read.

        Returns:
            Its messages in the order they were said.
        """
        rows = self._conn.execute(
            f"""
            SELECT {_MESSAGE_COLUMNS} FROM messages
             WHERE conversation_id = ?
             ORDER BY id
            """,
            (conversation_id,),
        ).fetchall()
        return [to_message(row) for row in rows]

    def create(self, entity: Message) -> Message:
        """Store one turn.

        Args:
            entity: The message to store. Its id should be None.

        Returns:
            The message with its id and created_at filled in.
        """
        new_id = self._insert(
            "INSERT INTO messages (conversation_id, role, content) VALUES (?, ?, ?)",
            (entity.conversation_id, str(entity.role), entity.content),
        )
        created = self.read_by_id(new_id)
        assert created is not None  # just inserted, inside the same transaction
        return created

    def update(self, entity: Message) -> Message:
        """Overwrite a stored message.

        Exists for the base contract more than for use. Editing chat history
        after the fact is not something this application does.

        Args:
            entity: The message to store, with its id set.

        Returns:
            The entity as given.

        Raises:
            ValueError: If the message has no id.
        """
        if entity.id is None:
            raise ValueError("cannot update a message that has not been created")
        self._conn.execute(
            """
            UPDATE messages
               SET conversation_id = ?, role = ?, content = ?
             WHERE id = ?
            """,
            (entity.conversation_id, str(entity.role), entity.content, entity.id),
        )
        return entity

    def delete(self, entity: Message) -> Message:
        """Remove one message and its citations.

        Args:
            entity: The message to remove, with its id set.

        Returns:
            The message that was removed.

        Raises:
            ValueError: If the message has no id.
        """
        if entity.id is None:
            raise ValueError("cannot delete a message that has not been created")
        self._conn.execute("DELETE FROM messages WHERE id = ?", (entity.id,))
        return entity


class CitationRepository(Repository[Citation, int]):
    """The sources sitting behind an answer."""

    def read(self) -> list[Citation]:
        """Return every citation ever recorded.

        Returns:
            All citations, grouped by message.
        """
        rows = self._conn.execute(
            f"SELECT {_CITATION_COLUMNS} FROM citations ORDER BY message_id, id"
        ).fetchall()
        return [to_citation(row) for row in rows]

    def read_by_id(self, entity_id: int) -> Citation | None:
        """Look up one citation.

        Args:
            entity_id: The citation's id.

        Returns:
            The citation, or None if there is no such row.
        """
        row = self._conn.execute(
            f"SELECT {_CITATION_COLUMNS} FROM citations WHERE id = ?", (entity_id,)
        ).fetchone()
        return to_citation(row) if row else None

    def read_for_message(self, message_id: int) -> list[Citation]:
        """Return the sources attached to one answer.

        Returns unverified citations too. Filtering them out is a display
        decision and belongs to whatever is rendering the answer, not to
        storage, and keeping the failures visible here is what lets you count
        how often verification fails at all.

        Args:
            message_id: The answer to read sources for.

        Returns:
            Its citations, in the order they were attached.
        """
        rows = self._conn.execute(
            f"""
            SELECT {_CITATION_COLUMNS} FROM citations
             WHERE message_id = ?
             ORDER BY id
            """,
            (message_id,),
        ).fetchall()
        return [to_citation(row) for row in rows]

    def create(self, entity: Citation) -> Citation:
        """Store one source.

        Args:
            entity: The citation to store. Its verified flag should already
                reflect whether the quote was found in the cited chunk.

        Returns:
            The citation with its id filled in.
        """
        new_id = self._insert(
            """
            INSERT INTO citations (message_id, chunk_id, quote, page_no, verified)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                entity.message_id,
                entity.chunk_id,
                entity.quote,
                entity.page_no,
                int(entity.verified),
            ),
        )
        created = self.read_by_id(new_id)
        assert created is not None  # just inserted, inside the same transaction
        return created

    def update(self, entity: Citation) -> Citation:
        """Overwrite a stored citation.

        Args:
            entity: The citation to store, with its id set.

        Returns:
            The entity as given.

        Raises:
            ValueError: If the citation has no id.
        """
        if entity.id is None:
            raise ValueError("cannot update a citation that has not been created")
        self._conn.execute(
            """
            UPDATE citations
               SET message_id = ?, chunk_id = ?, quote = ?, page_no = ?, verified = ?
             WHERE id = ?
            """,
            (
                entity.message_id,
                entity.chunk_id,
                entity.quote,
                entity.page_no,
                int(entity.verified),
                entity.id,
            ),
        )
        return entity

    def delete(self, entity: Citation) -> Citation:
        """Remove one citation.

        Args:
            entity: The citation to remove, with its id set.

        Returns:
            The citation that was removed.

        Raises:
            ValueError: If the citation has no id.
        """
        if entity.id is None:
            raise ValueError("cannot delete a citation that has not been created")
        self._conn.execute("DELETE FROM citations WHERE id = ?", (entity.id,))
        return entity
