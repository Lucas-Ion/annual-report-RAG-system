"""The verification rules, which are what every displayed figure rests on."""

from __future__ import annotations

from app.db.models import Block, Chunk, ChunkType
from app.verify import find_source, locate_page, normalise_for_match


def chunk(page_start: int, page_end: int, text: str, document_id: int = 7) -> Chunk:
    """Build a chunk for a test."""
    return Chunk(
        id=1,
        document_id=document_id,
        seq=0,
        page_start=page_start,
        page_end=page_end,
        chunk_type=ChunkType.PROSE,
        context_header="Acme | Annual Report 2025",
        text=text,
    )


def block(page_no: int, text: str, document_id: int = 7) -> Block:
    """Build a block for a test."""
    return Block(
        document_id=document_id, seq=0, page_no=page_no, label="text", text=text
    )


class TestNormalise:
    def test_collapses_whitespace_runs(self):
        assert normalise_for_match("a  \n\t b") == "a b"

    def test_strips_the_ends(self):
        assert normalise_for_match("  a b  ") == "a b"

    def test_leaves_everything_else_alone(self):
        assert normalise_for_match("87,870 (2024: 88,497)") == "87,870 (2024: 88,497)"


class TestFindSource:
    sources = [
        chunk(1, 1, "We employ 12,345 people."),
        chunk(9, 9, "Our workforce is our strength."),
    ]

    def test_finds_an_exact_quote(self):
        assert find_source("employ 12,345", self.sources) is self.sources[0]

    def test_searches_every_chunk_not_just_the_first(self):
        assert find_source("our strength", self.sources) is self.sources[1]

    def test_tolerates_whitespace_differences(self):
        assert find_source("We  employ\n12,345", self.sources) is self.sources[0]

    def test_rejects_an_invented_figure(self):
        assert find_source("We employ 12,346 people.", self.sources) is None

    def test_rejects_a_paraphrase(self):
        assert find_source("Acme has 12,345 staff", self.sources) is None

    def test_rejects_an_empty_quote(self):
        assert find_source("   ", self.sources) is None


class TestLocatePage:
    """The bug this exists for: a chunk spanning a page break.

    Stored page numbers are links that open the PDF, so landing a reader one
    page early undermines the exact thing the citation exists to prove.
    """

    spanning = chunk(4, 5, "text from page four\n\nthe figure was 12,345")
    blocks = [block(4, "text from page four"), block(5, "the figure was 12,345")]

    def test_single_page_chunk_needs_no_lookup(self):
        assert locate_page("anything", chunk(9, 9, "x"), []) == 9

    def test_quote_on_the_second_page_returns_the_second_page(self):
        assert locate_page("was 12,345", self.spanning, self.blocks) == 5

    def test_quote_on_the_first_page_returns_the_first(self):
        assert locate_page("from page four", self.spanning, self.blocks) == 4

    def test_falls_back_when_no_block_holds_the_quote(self):
        assert locate_page("not in any block", self.spanning, self.blocks) == 4

    def test_ignores_blocks_from_another_document(self):
        elsewhere = [block(5, "was 12,345", document_id=99)]
        assert locate_page("was 12,345", self.spanning, elsewhere) == 4

    def test_ignores_blocks_outside_the_chunks_page_range(self):
        assert locate_page("was 12,345", self.spanning, [block(300, "was 12,345")]) == 4

    def test_tolerates_whitespace_differences(self):
        assert locate_page("the  figure\nwas 12,345", self.spanning, self.blocks) == 5
