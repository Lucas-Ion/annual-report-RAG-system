"""The generation component using Claude"""

from __future__ import annotations

import os
from collections.abc import Iterator
from functools import cached_property

from anthropic import Anthropic
from pydantic import BaseModel

DEFAULT_MODEL = "claude-opus-5"

DEFAULT_MAX_TOKENS = 4096


class MissingApiKey(RuntimeError):
    """Raised when no API key is configured."""


class ClaudeProvider:
    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
    ) -> None:
        self._api_key = api_key
        self._model = model

    @cached_property
    def _client(self) -> Anthropic:
        key = self._api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise MissingApiKey(
                "ANTHROPIC_API_KEY is not set. Put it in .env at the "
                "repository root, or export it before starting the app."
            )
        return Anthropic(api_key=key)

    @property
    def model(self) -> str:
        return self._model

    def complete(
        self, *, system: str, prompt: str, max_tokens: int = DEFAULT_MAX_TOKENS
    ) -> str:
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
