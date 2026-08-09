"""The generation half: Claude, behind a deliberately narrow interface.

Two jobs, and they are different enough to be separate methods. Extraction
needs a typed object back and gets it through structured outputs, so a
malformed reply fails at the API rather than halfway through parsing. Chat
needs prose, streamed, so the interface is not blank for the several seconds
an answer takes to write.

Nothing here knows about annual reports, chunks or citations. Prompts belong
to the callers in app/ingest and app/chat, which keeps this file a transport
and lets those be tested by reading a string rather than by making a request.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from functools import cached_property

from anthropic import Anthropic
from pydantic import BaseModel

# The 1M token context is irrelevant here, since retrieval hands over eight
# chunks. What matters is instruction following on a task where the failure
# mode is a confident wrong number, and where a quote has to come back
# character for character out of the source.
DEFAULT_MODEL = "claude-opus-5"

# Enough for an answer with several quoted passages. Extraction overrides this
# downwards, because a field extractor that wants 4,000 tokens has
# misunderstood the question.
DEFAULT_MAX_TOKENS = 4096


class MissingApiKey(RuntimeError):
    """Raised when no API key is configured.

    Its own type so the web layer can tell the difference between "not set up
    yet", which deserves an explanation and a link, and a real failure from
    the API, which deserves a stack trace in the log.
    """


class ClaudeProvider:
    """Claude behind the LanguageModel port."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
    ) -> None:
        """Configure the provider without contacting anything.

        Args:
            api_key: The key, or None to read ANTHROPIC_API_KEY from the
                environment when the client is first needed.
            model: Model id.
        """
        self._api_key = api_key
        self._model = model

    @cached_property
    def _client(self) -> Anthropic:
        """Build the client on first use.

        Deferred so that constructing a provider stays free. The application
        builds one at startup and only some requests ever reach the model, and
        a missing key should not stop the document browser from working.

        Returns:
            A configured client.

        Raises:
            MissingApiKey: If no key was passed and none is in the
                environment.
        """
        key = self._api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise MissingApiKey(
                "ANTHROPIC_API_KEY is not set. Put it in .env at the "
                "repository root, or export it before starting the app."
            )
        return Anthropic(api_key=key)

    @property
    def model(self) -> str:
        """Which model this provider calls.

        Returns:
            The model id.
        """
        return self._model

    def complete(
        self, *, system: str, prompt: str, max_tokens: int = DEFAULT_MAX_TOKENS
    ) -> str:
        """Answer once and return the whole reply.

        Args:
            system: Instructions framing the task.
            prompt: The question, with its retrieved context.
            max_tokens: Ceiling on the reply.

        Returns:
            The reply text, with any non-text content blocks dropped.

        Raises:
            ValueError: If the budget ran out before any text was produced.
                This model thinks before it answers, and thinking spends the
                same allowance, so a tight max_tokens can be consumed entirely
                by a thinking block. Returning the resulting empty string
                looked exactly like a model that had nothing to say, and made
                every chat in this application come out untitled.
        """
        message = self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(block.text for block in message.content if block.type == "text")
        if not text and message.stop_reason == "max_tokens":
            raise ValueError(
                f"the reply hit max_tokens ({max_tokens}) before producing any "
                f"text. Thinking uses the same allowance, so raise it."
            )
        return text

    def stream(
        self, *, system: str, prompt: str, max_tokens: int = DEFAULT_MAX_TOKENS
    ) -> Iterator[str]:
        """Answer once, yielding text as it arrives.

        What the chat interface uses. An answer over eight chunks of an annual
        report takes several seconds to write, and a blank screen for that long
        reads as broken.

        Args:
            system: Instructions framing the task.
            prompt: The question, with its retrieved context.
            max_tokens: Ceiling on the reply.

        Yields:
            Fragments of the reply, in order. Concatenating everything yielded
            gives the same string complete() would have returned.
        """
        with self._client.messages.stream(
            model=self._model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            yield from stream.text_stream

    def parse[T: BaseModel](
        self, *, system: str, prompt: str, schema: type[T], max_tokens: int = 2048
    ) -> T:
        """Extract structured data and get it back as a typed object.

        Uses structured outputs, so the model is constrained to the schema
        rather than asked politely to follow it. That difference matters here:
        the alternative is parsing JSON out of prose and discovering at ingest
        time that one report in five produced a trailing comma.

        Note that structured outputs and Claude's native citations cannot be
        used in the same request. Extraction therefore carries its own
        verbatim_quote field and the quote is checked against the source chunk
        afterwards, which is a stronger guarantee anyway because it is our
        check rather than the model's claim.

        Args:
            system: Instructions framing the extraction.
            prompt: The chunks to read, and what to look for.
            schema: A Pydantic model describing the expected shape.
            max_tokens: Ceiling on the reply.

        Returns:
            An instance of the schema.

        Raises:
            ValueError: If the reply carried no parsed output. The SDK types
                this as optional because a reply can stop early, on the token
                limit or a refusal, and returning None into an extraction
                pipeline would push the failure several stages downstream.
        """
        message = self._client.messages.parse(
            model=self._model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            output_format=schema,
        )
        if message.parsed_output is None:
            raise ValueError(
                f"no structured output in the reply, stop reason "
                f"{message.stop_reason!r}"
            )
        return message.parsed_output
