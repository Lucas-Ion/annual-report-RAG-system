"""Checking that a quotation really came from where it claims to.

Shared by extraction and by chat, which make the same promise about different
kinds of output: a figure on the reports page and a citation in a streamed
answer are both only shown because the text was found in the indexed source.
Keeping one implementation means the two cannot drift into disagreeing about
what counts as a match.

Everything here is pure. Callers fetch whatever rows are needed and pass them
in, so the rules can be tested against list literals in microseconds.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from app.db.models import Block, Chunk

# Whitespace only. Line breaks, the padding inside table cells and the
# difference between one space and three are artefacts of how a PDF was laid
# out, and no model reproduces them. Every other character has to match, so
# numbers, punctuation and wording are still compared exactly: 87,871 does not
# match a source saying 87,870.
_WHITESPACE = re.compile(r"\s+")


def normalise_for_match(text: str) -> str:
    """Reduce text to the form used for comparing a quote against its source.

    Args:
        text: A quote, a chunk, or a block.

    Returns:
        The text with whitespace runs collapsed and the ends stripped.
    """
    return _WHITESPACE.sub(" ", text).strip()


def find_source(quote: str, chunks: Sequence[Chunk]) -> Chunk | None:
    """Find which chunk a quotation actually came from.

    Every chunk is searched rather than only the one that was cited. Models
    misnumber their references fairly often while quoting perfectly accurately,
    and rejecting a genuine quote over a bookkeeping slip throws away good
    data. What is never relaxed is that the text has to exist somewhere in what
    the model was actually shown.

    Args:
        quote: The claimed quotation.
        chunks: The excerpts that were given to the model.

    Returns:
        The chunk containing the quote, or None if none does, which means the
        claim is unverifiable.
    """
    needle = normalise_for_match(quote)
    if not needle:
        return None
    for chunk in chunks:
        if needle in normalise_for_match(chunk.text):
            return chunk
    return None


def locate_page(quote: str, chunk: Chunk, blocks: Sequence[Block] = ()) -> int:
    """Work out which page a quotation is actually printed on.

    A chunk is often built from blocks on two pages, and taking its first page
    is then wrong for anything quoted from the second. Measured across the
    stored facts, every incorrect page number came from exactly this: the page
    was right for 14 quotations and one too low for 7, and all 7 came from
    chunks that crossed a page break.

    That matters more than it sounds, because every page number in the
    interface is a link that opens the PDF at that page. A citation that lands
    the reader one page early undermines the exact thing it exists to prove.

    Blocks are the fix because they carry a real page number and are finer
    grained than chunks. A chunk that sits on one page needs no lookup at all,
    which is the common case.

    Args:
        quote: The quotation.
        chunk: The chunk it was found in.
        blocks: Candidate blocks. Only those inside the chunk's own page range
            are considered, so passing a wider selection is harmless. Ignored
            entirely when the chunk does not span pages.

    Returns:
        The page the quotation is on. Falls back to the chunk's first page when
        no block contains it, which happens when chunking joined text across a
        block boundary.
    """
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
