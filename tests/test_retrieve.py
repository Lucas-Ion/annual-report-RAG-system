"""Testing Rank fusion, round robin merging, and reading a company out of a question."""

from __future__ import annotations

import pytest

from app.db.models import Document
from app.retrieve import aliases, detect_document, fuse, interleave


class TestFuse:

    def test_empty_input(self):
        assert fuse([]) == []

    def test_a_single_list_keeps_its_order(self):
        assert fuse([[5, 3, 9]]) == [5, 3, 9]

    def test_never_repeats_an_id(self):
        assert sorted(fuse([[1, 2], [2, 1], [1, 2]])) == [1, 2]

    def test_agreement_beats_a_single_strong_placing(self):
        merged = fuse([[9, 8, 7], [6, 5, 7]])
        assert merged.index(7) < merged.index(9)
        assert merged.index(7) < merged.index(6)


class TestInterleave:

    def test_takes_turns(self):
        assert interleave([[1, 2, 3], [4, 5, 6]], 4) == [1, 4, 2, 5]

    def test_dedupes_across_lists(self):
        assert interleave([[1, 2], [1, 3]], 4) == [1, 2, 3]

    def test_respects_the_limit(self):
        assert len(interleave([[1, 2, 3], [4, 5, 6]], 3)) == 3

    def test_handles_ragged_lists(self):
        assert interleave([[1], [2, 3, 4]], 9) == [1, 2, 3, 4]

    def test_empty_input(self):
        assert interleave([], 5) == []

    def test_a_specific_answer_survives_generic_competition(self):
        rankings = [[99, 42], [99, 50], [99, 51], [99, 52]]
        assert 42 in interleave(rankings, 5)
        assert fuse(rankings).index(42) > fuse(rankings).index(99)


def report(company: str) -> Document:
    return Document(
        id=hash(company) % 1000,
        filename="x.pdf",
        file_hash=company,
        company=company,
        year=2025,
    )


DOCUMENTS = [
    report(name) for name in ("ABN AMRO", "ASML", "CM", "Heineken N.V.", "Shell")
]


class TestAliases:
    def test_strips_a_corporate_suffix(self):
        assert "heineken" in aliases("Heineken N.V.")

    def test_keeps_the_full_name_too(self):
        assert "heineken n.v." in aliases("Heineken N.V.")

    def test_leaves_a_plain_name_alone(self):
        assert aliases("ABN AMRO") == {"abn amro"}


class TestDetectDocument:
    @pytest.mark.parametrize(
        ("question", "expected"),
        [
            ("How much did Shell spend on climate adaptation in 2025?", "Shell"),
            ("How much did SHELL spend?", "Shell"),
            ("and Heineken's employee count?", "Heineken N.V."),
            ("What is Heineken N.V.'s net zero target?", "Heineken N.V."),
            ("How many employees does ABN AMRO have?", "ABN AMRO"),
            ("asml climate targets", "ASML"),
            ("What is CM's headcount?", "CM"),
        ],
    )
    def test_finds_the_named_company(self, question, expected):
        found = detect_document(question, DOCUMENTS)
        assert found is not None and found.company == expected

    @pytest.mark.parametrize(
        "question",
        [
            "Compare Shell and ASML emissions targets",
            "Which company has the most employees?",
            "What is the measurement in cm for this?",
            "Tell me about shellfish processing",
        ],
    )
    def test_declines_to_narrow(self, question):
        assert detect_document(question, DOCUMENTS) is None
