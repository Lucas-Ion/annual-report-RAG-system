"""Everything this system talks to that is not a file or the database.

Two ports, both defined in base.py as Protocols and both with exactly one
implementation today. The Protocols exist so a test can substitute a fake
without loading a 2GB model or spending money, not because a second real
implementation is expected.
"""

from app.providers.base import (
    EmbeddingProvider,
    LanguageModel,
    StructuredExtractor,
    TextGenerator,
)
from app.providers.claude import ClaudeProvider, MissingApiKey
from app.providers.embeddings import BGEEmbeddings

__all__ = [
    "BGEEmbeddings",
    "ClaudeProvider",
    "EmbeddingProvider",
    "LanguageModel",
    "StructuredExtractor",
    "TextGenerator",
    "MissingApiKey",
]
