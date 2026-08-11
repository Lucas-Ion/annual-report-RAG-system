"""Storage for conversations, their messages, and the sources behind answers."""

from __future__ import annotations

import sqlite3

from app.db.models import Citation, Conversation, Message, Role
from app.db.repositories.base import Repository

_CONVERSATION_COLUMNS = "id, title, created_at"
_MESSAGE_COLUMNS = "id, conversation_id, role, content, created_at"
_CITATION_COLUMNS = "id, message_id, chunk_id, quote, page_no, verified"


def to_conversation(row: sqlite3.Row) -> Conversation:
    return Conversation(id=row["id"], title=row["title"], created_at=row["created_at"])


def to_message(row: sqlite3.Row) -> Message:
    return Message(
        id=row["id"],
        conversation_id=row["conversation_id"],
        role=Role(row["role"]),
        content=row["content"],
        created_at=row["created_at"],
    )


def to_citation(row: sqlite3.Row) -> Citation:
    return Citation(
        id=row["id"],
        message_id=row["message_id"],
        chunk_id=row["chunk_id"],
        quote=row["quote"],
        page_no=row["page_no"],
        verified=bool(row["verified"]),
    )


class ConversationRepository(Repository[Conversation, int]):
    def read(self) -> list[Conversation]:
        rows = self._conn.execute(
            f"SELECT {_CONVERSATION_COLUMNS} FROM conversations ORDER BY id DESC"
        ).fetchall()
        return [to_conversation(row) for row in rows]

    def read_by_id(self, entity_id: int) -> Conversation | None:
        row = self._conn.execute(
            f"SELECT {_CONVERSATION_COLUMNS} FROM conversations WHERE id = ?",
            (entity_id,),
        ).fetchone()
        return to_conversation(row) if row else None

    def create(self, entity: Conversation) -> Conversation:
        new_id = self._insert(
            "INSERT INTO conversations (title) VALUES (?)", (entity.title,)
        )
        created = self.read_by_id(new_id)
        assert created is not None  # just inserted, inside the same transaction
        return created

    def update(self, entity: Conversation) -> Conversation:
        if entity.id is None:
            raise ValueError("cannot update a conversation that has not been created")
        self._conn.execute(
            "UPDATE conversations SET title = ? WHERE id = ?", (entity.title, entity.id)
        )
        return entity

    def delete(self, entity: Conversation) -> Conversation:
        if entity.id is None:
            raise ValueError("cannot delete a conversation that has not been created")
        self._conn.execute("DELETE FROM conversations WHERE id = ?", (entity.id,))
        return entity


class MessageRepository(Repository[Message, int]):
    def read(self) -> list[Message]:
        rows = self._conn.execute(
            f"SELECT {_MESSAGE_COLUMNS} FROM messages ORDER BY conversation_id, id"
        ).fetchall()
        return [to_message(row) for row in rows]

    def read_by_id(self, entity_id: int) -> Message | None:
        row = self._conn.execute(
            f"SELECT {_MESSAGE_COLUMNS} FROM messages WHERE id = ?", (entity_id,)
        ).fetchone()
        return to_message(row) if row else None

    def read_for_conversation(self, conversation_id: int) -> list[Message]:
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
        new_id = self._insert(
            "INSERT INTO messages (conversation_id, role, content) VALUES (?, ?, ?)",
            (entity.conversation_id, str(entity.role), entity.content),
        )
        created = self.read_by_id(new_id)
        assert created is not None  # just inserted, inside the same transaction
        return created

    def update(self, entity: Message) -> Message:
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
        if entity.id is None:
            raise ValueError("cannot delete a message that has not been created")
        self._conn.execute("DELETE FROM messages WHERE id = ?", (entity.id,))
        return entity


class CitationRepository(Repository[Citation, int]):
    def read(self) -> list[Citation]:
        rows = self._conn.execute(
            f"SELECT {_CITATION_COLUMNS} FROM citations ORDER BY message_id, id"
        ).fetchall()
        return [to_citation(row) for row in rows]

    def read_by_id(self, entity_id: int) -> Citation | None:
        row = self._conn.execute(
            f"SELECT {_CITATION_COLUMNS} FROM citations WHERE id = ?", (entity_id,)
        ).fetchone()
        return to_citation(row) if row else None

    def read_for_message(self, message_id: int) -> list[Citation]:
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
        assert created is not None
        return created

    def update(self, entity: Citation) -> Citation:
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
        if entity.id is None:
            raise ValueError("cannot delete a citation that has not been created")
        self._conn.execute("DELETE FROM citations WHERE id = ?", (entity.id,))
        return entity
