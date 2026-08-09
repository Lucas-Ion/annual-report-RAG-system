"""Every repository in the system, and the one place to import them from.

Callers write:

    from app.db.repositories import ChunkRepository

rather than reaching into the module a class happens to live in today. That
keeps the file layout free to change without a rename rippling through the
pipeline and the routes.
"""

from app.db.repositories.base import Repository
from app.db.repositories.blocks import BlockRepository
from app.db.repositories.chat import (
    CitationRepository,
    ConversationRepository,
    MessageRepository,
)
from app.db.repositories.chunks import ChunkRepository
from app.db.repositories.documents import DocumentRepository
from app.db.repositories.facts import FactRepository
from app.db.repositories.stage_runs import StageRunRepository

__all__ = [
    "BlockRepository",
    "ChunkRepository",
    "CitationRepository",
    "ConversationRepository",
    "DocumentRepository",
    "FactRepository",
    "MessageRepository",
    "Repository",
    "StageRunRepository",
]
