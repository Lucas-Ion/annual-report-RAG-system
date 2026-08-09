"""The two model shaped holes in this application.

Everything else in the system is deterministic and offline. These two ports
are where it reaches out to something that costs money, needs a network, or
takes gigabytes of disk, which is exactly why they are named and kept thin.

Protocols rather than base classes, because the point here is not shared
implementation. It is that a test can hand the pipeline a dictionary of
canned answers and get the same type checking as the real thing, without
loading a 2GB model or spending a cent. Nothing needs to inherit from these:
a class matching the shape satisfies them.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Protocol, runtime_checkable

from pydantic import BaseModel


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Turns text into vectors for the dense half of retrieval."""

    @property
    def dimensions(self) -> int:
        """Length of the vectors this provider returns.

        Checked against the width baked into the chunk_vectors table, since a
        mismatch there is a confusing failure at insert time and a silently
        useless index if it somehow gets through.
        """
        ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed chunks for storage.

        Args:
            texts: The chunk texts, already carrying their context headers.

        Returns:
            One vector per input, in the same order.
        """
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed a question for searching.

        Separate from embed_documents because many embedding models expect
        queries to be prefixed with an instruction and documents not to be,
        and getting that backwards degrades retrieval quietly rather than
        loudly. bge-m3 happens to need no prefix on either, but the two stay
        distinct so that swapping the model cannot introduce that bug.

        Args:
            text: The user's question.

        Returns:
            One vector.
        """
        ...


@runtime_checkable
class TextGenerator(Protocol):
    """Writes prose. What the chat interface needs."""

    def complete(self, *, system: str, prompt: str, max_tokens: int = 4096) -> str:
        """Answer a single question and return the whole reply.

        Args:
            system: Instructions that frame the task.
            prompt: The question, with whatever context it needs.
            max_tokens: Ceiling on the reply length.

        Returns:
            The reply as plain text.
        """
        ...

    def stream(
        self, *, system: str, prompt: str, max_tokens: int = 4096
    ) -> Iterator[str]:
        """Answer a single question, yielding text as it is written.

        Args:
            system: Instructions that frame the task.
            prompt: The question, with whatever context it needs.
            max_tokens: Ceiling on the reply length.

        Yields:
            Fragments of the reply, in order.
        """
        ...


@runtime_checkable
class StructuredExtractor(Protocol):
    """Returns a typed object rather than prose. What extraction needs."""

    def parse[T: BaseModel](
        self, *, system: str, prompt: str, schema: type[T], max_tokens: int = 2048
    ) -> T:
        """Read the prompt and fill in the schema.

        Args:
            system: Instructions that frame the task.
            prompt: The material to read.
            schema: A Pydantic model describing the expected shape.
            max_tokens: Ceiling on the reply length.

        Returns:
            An instance of the schema.
        """
        ...


@runtime_checkable
class LanguageModel(TextGenerator, StructuredExtractor, Protocol):
    """Both halves at once, which is what a real provider offers.

    Split into the two above so that call sites can ask for only what they
    use. Extraction takes a StructuredExtractor, so a test fake for it is one
    method rather than three, and a fake that quietly returned the wrong thing
    from an unused method cannot exist.
    """
