"""Batch planning, block mapping, and chunking."""

from __future__ import annotations

import pytest

from app.db.models import Block, ChunkType, Document
from app.ingest.chunk import (
    CEILING_CHARS,
    build_chunks,
    build_context_header,
    collapse_table,
    is_separator_row,
    split_table,
)
from app.ingest.parse import plan_batches

DOCUMENT = Document(id=1, filename="x.pdf", file_hash="h", company="Acme", year=2025)


def block(seq: int, label: str, text: str, page: int = 1) -> Block:
    return Block(document_id=1, seq=seq, page_no=page, label=label, text=text)


class TestPlanBatches:

    def test_a_fresh_document_covers_every_page_once(self):
        pages = [
            p
            for first, last in plan_batches(434, None, 25)
            for p in range(first, last + 1)
        ]
        assert pages == list(range(1, 435))

    def test_the_last_batch_is_clipped(self):
        assert plan_batches(434, None, 25)[-1] == (426, 434)

    def test_resuming_skips_completed_batches(self):
        assert plan_batches(434, 25, 25)[0] == (26, 50)

    def test_resuming_mid_batch_skips_that_whole_batch(self):
        assert plan_batches(434, 30, 25)[0] == (51, 75)

    def test_a_batch_whose_tail_pages_were_blank_still_counts_as_done(self):
        assert plan_batches(434, 398, 25)[0] == (401, 425)

    def test_a_finished_document_has_nothing_to_do(self):
        assert plan_batches(434, 434, 25) == []

    def test_an_empty_document(self):
        assert plan_batches(0, None, 25) == []

    def test_a_zero_batch_size_is_refused(self):
        with pytest.raises(ValueError):
            plan_batches(10, None, 0)


class TestCollapseTable:

    def test_removes_cell_padding(self):
        padded = (
            "| Name          | Value        |\n|-------|------|\n| Revenue | 1,204 |"
        )
        assert collapse_table(padded).splitlines()[0] == "| Name | Value |"

    def test_keeps_every_value(self):
        padded = "| Name    | Value |\n|----|----|\n| Revenue  | 1,204 |"
        collapsed = collapse_table(padded)
        assert all(word in collapsed for word in ("Name", "Value", "Revenue", "1,204"))

    def test_rebuilds_the_separator_rather_than_collapsing_it(self):
        wide = "| a | b |\n|" + "-" * 400 + "|" + "-" * 400 + "|\n| 1 | 2 |"
        assert len(collapse_table(wide).splitlines()[1]) < 30

    def test_flattens_newlines_inside_a_cell(self):
        assert "\n" not in collapse_table("| a\nb | c |").splitlines()[0]

    def test_recognises_a_separator_row(self):
        assert is_separator_row("|---|:--:|---|")
        assert not is_separator_row("| 1 | 2 |")


class TestSplitTable:
    header = "| item | 2025 | 2024 |\n| --- | --- | --- |"

    def test_a_small_table_is_left_alone(self):
        small = "| a | b |\n| --- | --- |\n| 1 | 2 |"
        assert split_table(small) == [small]

    def test_splits_by_row_and_repeats_the_header(self):
        rows = "\n".join(
            f"| row {i} with padding text | {i}00 | {i}50 |" for i in range(60)
        )
        pieces = split_table(self.header + "\n" + rows, 600)
        assert len(pieces) > 1
        assert all(piece.startswith("| item") for piece in pieces)

    def test_loses_no_rows_and_duplicates_none(self):
        rows = "\n".join(
            f"| row {i} with padding text | {i}00 | {i}50 |" for i in range(60)
        )
        pieces = split_table(self.header + "\n" + rows, 600)
        seen = {
            line
            for piece in pieces
            for line in piece.splitlines()
            if line.startswith("| row ")
        }
        assert len(seen) == 60

    def test_falls_back_to_splitting_by_column_when_a_row_is_too_long(self):
        wide = (
            "| "
            + " | ".join(f"h{i}" for i in range(8))
            + " |\n"
            + "|"
            + "|".join("---" for _ in range(8))
            + "|\n"
            + "| "
            + " | ".join("x" * 1500 for _ in range(8))
            + " |"
        )
        pieces = split_table(wide, 3000)
        assert len(pieces) > 1
        assert all(len(piece) <= 3000 for piece in pieces)
        assert all(piece.startswith("|") and piece.endswith("|") for piece in pieces)


class TestBuildChunks:
    def test_no_blocks_no_chunks(self):
        assert build_chunks(DOCUMENT, []) == []

    def test_a_heading_becomes_the_chunks_section(self):
        chunks = build_chunks(
            DOCUMENT,
            [
                block(0, "section_header", "Our people"),
                block(1, "text", "We employ 12,345 people."),
            ],
        )
        assert len(chunks) == 1
        assert chunks[0].section == "Our people"
        assert "Our people" in chunks[0].text

    def test_the_context_header_is_never_part_of_the_text(self):
        chunks = build_chunks(DOCUMENT, [block(0, "text", "body")])
        assert chunks[0].context_header not in chunks[0].text
        assert chunks[0].embedding_text.startswith(chunks[0].context_header)

    def test_a_table_never_shares_a_chunk_with_prose(self):
        chunks = build_chunks(
            DOCUMENT,
            [
                block(0, "text", "before"),
                block(1, "table", "| a |\n| --- |\n| 1 |"),
                block(2, "text", "after"),
            ],
        )
        assert [c.chunk_type for c in chunks] == [
            ChunkType.PROSE,
            ChunkType.TABLE,
            ChunkType.PROSE,
        ]

    def test_a_caption_is_absorbed_into_the_table_it_introduces(self):
        chunks = build_chunks(
            DOCUMENT,
            [
                block(0, "caption", "Table 5: FTE by region"),
                block(1, "table", "| region |\n| --- |\n| NL |"),
            ],
        )
        assert len(chunks) == 1
        assert "Table 5" in chunks[0].text

    def test_a_caption_with_no_table_after_it_stays_prose(self):
        chunks = build_chunks(
            DOCUMENT, [block(0, "caption", "Figure 2"), block(1, "text", "body")]
        )
        assert len(chunks) == 1
        assert chunks[0].chunk_type is ChunkType.PROSE

    def test_nothing_exceeds_the_ceiling(self):
        chunks = build_chunks(
            DOCUMENT, [block(i, "text", "x" * 400) for i in range(30)]
        )
        assert len(chunks) > 1
        assert all(len(c.text) <= CEILING_CHARS for c in chunks)

    def test_sequence_numbers_are_gapless_from_zero(self):
        chunks = build_chunks(
            DOCUMENT, [block(i, "text", "x" * 400) for i in range(30)]
        )
        assert [c.seq for c in chunks] == list(range(len(chunks)))

    def test_noise_labels_are_dropped(self):
        chunks = build_chunks(
            DOCUMENT,
            [
                block(0, "text", "keep"),
                block(1, "document_index", "table of contents noise"),
                block(2, "checkbox_selected", "x"),
            ],
        )
        assert len(chunks) == 1
        assert "contents" not in chunks[0].text

    def test_the_page_range_spans_its_blocks(self):
        chunks = build_chunks(
            DOCUMENT, [block(0, "text", "a", page=4), block(1, "text", "b", page=6)]
        )
        assert (chunks[0].page_start, chunks[0].page_end) == (4, 6)


class TestContextHeader:
    def test_includes_the_section_when_there_is_one(self):
        assert build_context_header(DOCUMENT, "Sustainability") == (
            "Acme | Annual Report 2025 | Sustainability"
        )

    def test_omits_it_when_there_is_not(self):
        assert build_context_header(DOCUMENT, None) == "Acme | Annual Report 2025"
