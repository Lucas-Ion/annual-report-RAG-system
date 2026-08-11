"""Checking that a quotation really came from where it claims to."""

from __future__ import annotations

import re
from collections.abc import Sequence

from app.db.models import Block, Chunk

_WHITESPACE = re.compile(r"\s+")


def normalise_for_match(text: str) -> str:
    return _WHITESPACE.sub(" ", text).strip()


def find_source(quote: str, chunks: Sequence[Chunk]) -> Chunk | None:
    needle = normalise_for_match(quote)
    if not needle:
        return None
    for chunk in chunks:
        if needle in normalise_for_match(chunk.text):
            return chunk
    return None


def locate_page(quote: str, chunk: Chunk, blocks: Sequence[Block] = ()) -> int:
    if chunk.page_end == chunk.page_start:
        return chunk.page_start

    needle = normalise_for_match(quote)
    if not needle:
        return chunk.page_start

    for block in blocks:
        if block.document_id != chunk.document_id:
            continue
        if not chunk.page_start <= block.page_no <= chunk.page_end:
            continue
        if needle in normalise_for_match(block.text):
            return block.page_no
    return chunk.page_start
