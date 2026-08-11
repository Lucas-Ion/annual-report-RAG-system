""" These two ports
are where it reaches out to something that costs money, needs a network, or
takes gigabytes of disk, which is exactly why they are named and kept thin."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Protocol, runtime_checkable

from pydantic import BaseModel


@runtime_checkable
class EmbeddingProvider(Protocol):

    @property
    def dimensions(self) -> int:
        """Length of the vectors this provider returns."""
        ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed chunks for storage.

        Args:
            texts: The chunk texts, already carrying with their context headers.

        Returns:
            One vector per input, in the same order.
        """
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed a question for searching.

        Args:
            text: The user's question.

        Returns:
            One vector.
        """
        ...


@runtime_checkable
class TextGenerator(Protocol):

    def complete(self, *, system: str, prompt: str, max_tokens: int = 4096) -> str:
        """Answer a single question and return the whole reply.

        Args:
            system: Instructions that frame the task.
            prompt: The question, with whatever context it needs.
            max_tokens: Ceiling on the reply length.

        Returns:
            The reply as text.
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
    use.
    """
