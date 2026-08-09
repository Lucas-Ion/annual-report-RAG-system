"""Extraction, and the gate that decides what is allowed to be stored.

The model here is a fake, so these tests cost nothing and are deterministic.
What they check is the part that is ours: whether a claimed quotation survives
being looked for in the text the model was actually shown.
"""

from __future__ import annotations

from app.db.models import Chunk, ChunkType, Document
from app.db.repositories import FactRepository
from app.ingest.extract import (
    ExtractedValue,
    Extraction,
    build_prompt,
    extract_document,
    to_facts,
)
from app.ingest.fields import FTE, SUSTAINABILITY_GOAL

DOCUMENT = Document(id=1, filename="x.pdf", file_hash="h", company="Acme", year=2025)


def chunk(chunk_id: int, text: str, page: int = 8) -> Chunk:
    """Build an excerpt for a test."""
    return Chunk(
        id=chunk_id,
        document_id=1,
        seq=chunk_id,
        page_start=page,
        page_end=page,
        chunk_type=ChunkType.TABLE,
        context_header="Acme | Annual Report 2025",
        text=text,
    )


REAL = chunk(10, "| Average number of employees (FTE) | 12,345 | 12,900 |")
OTHER = chunk(11, "Our people are our strength.", page=31)
EXCERPTS = [REAL, OTHER]


def value(**overrides) -> ExtractedValue:
    """Build an extracted value, correct unless overridden."""
    return ExtractedValue(
        **{
            "value_raw": "12,345",
            "value_numeric": 12345.0,
            "unit": "FTE",
            "verbatim_quote": "Average number of employees (FTE) | 12,345",
            "excerpt_number": 1,
            "confidence": 0.95,
            **overrides,
        }
    )


class TestVerificationGate:
    def test_a_quoted_value_is_stored(self):
        facts, rejected = to_facts(
            Extraction(values=[value()]), EXCERPTS, DOCUMENT, FTE
        )
        assert len(facts) == 1 and rejected == []

    def test_an_invented_figure_is_rejected(self):
        """The failure this whole stage exists to prevent. The model is
        fluent, the number is plausible, and it is not in the report."""
        made_up = value(
            value_raw="99,999",
            verbatim_quote="Average number of employees (FTE) | 99,999",
        )
        facts, rejected = to_facts(
            Extraction(values=[made_up]), EXCERPTS, DOCUMENT, FTE
        )
        assert facts == []
        assert len(rejected) == 1 and "99,999" in rejected[0]

    def test_a_paraphrase_is_rejected(self):
        near = value(verbatim_quote="Acme employed 12,345 people on average")
        facts, _ = to_facts(Extraction(values=[near]), EXCERPTS, DOCUMENT, FTE)
        assert facts == []

    def test_a_wrong_excerpt_number_does_not_lose_a_real_quote(self):
        """Models misnumber references while quoting accurately. The quote is
        looked for in every excerpt, and the page comes from where it was
        actually found rather than from what was claimed."""
        misnumbered = value(verbatim_quote="our strength", excerpt_number=99)
        facts, rejected = to_facts(
            Extraction(values=[misnumbered]), EXCERPTS, DOCUMENT, FTE
        )
        assert rejected == []
        assert facts[0].page_no == 31 and facts[0].chunk_id == 11

    def test_good_and_bad_values_in_one_reply_are_separated(self):
        extraction = Extraction(values=[value(), value(verbatim_quote="invented")])
        facts, rejected = to_facts(extraction, EXCERPTS, DOCUMENT, FTE)
        assert len(facts) == 1 and len(rejected) == 1

    def test_an_empty_reply_is_a_valid_answer(self):
        """A missing datapoint is a correct result. An invented one is not."""
        facts, rejected = to_facts(Extraction(), EXCERPTS, DOCUMENT, FTE)
        assert facts == [] and rejected == []

    def test_the_fields_unit_is_used_when_the_model_gives_none(self):
        facts, _ = to_facts(
            Extraction(values=[value(unit=None)]), EXCERPTS, DOCUMENT, FTE
        )
        assert facts[0].unit == "FTE"

    def test_every_stored_fact_is_stamped_with_the_extractor_version(self):
        facts, _ = to_facts(Extraction(values=[value()]), EXCERPTS, DOCUMENT, FTE)
        assert facts[0].extractor_version


class TestPrompt:
    def test_names_the_company_and_the_year(self):
        prompt = build_prompt(DOCUMENT, FTE, EXCERPTS)
        assert "Acme" in prompt and "2025" in prompt

    def test_numbers_the_excerpts_with_their_pages(self):
        prompt = build_prompt(DOCUMENT, FTE, EXCERPTS)
        assert "--- excerpt 1 (page 8" in prompt
        assert "--- excerpt 2 (page 31" in prompt

    def test_carries_the_fields_instruction(self):
        """The instruction encodes what was learned from the corpus, such as
        preferring an exact figure over a rounded one."""
        assert "Prefer an exact figure" in build_prompt(DOCUMENT, FTE, EXCERPTS)

    def test_a_single_valued_field_asks_for_one_answer(self):
        assert "exactly one" in build_prompt(DOCUMENT, FTE, EXCERPTS)

    def test_a_multi_valued_field_asks_for_several(self):
        assert "several" in build_prompt(DOCUMENT, SUSTAINABILITY_GOAL, EXCERPTS)


class TestExtractDocument:
    def test_a_failing_field_does_not_lose_the_others(self, conn, seeded, embeddings):
        """A truncated reply arrives as a JSON parse error. Before this, one
        bad field abandoned the whole document and the finished fields with it.
        """

        class HalfBroken:
            def __init__(self):
                self.calls = 0

            def parse(self, *, system, prompt, schema, max_tokens=2048):
                self.calls += 1
                if self.calls == 1:
                    raise ValueError("Invalid JSON: EOF while parsing a string")
                return schema(
                    values=[
                        ExtractedValue(
                            value_raw="net zero by 2040",
                            verbatim_quote="average number of employees",
                            excerpt_number=1,
                            confidence=0.8,
                        )
                    ]
                )

        stored = extract_document(conn, seeded, HalfBroken(), embeddings)
        assert stored == 1
        assert FactRepository(conn).read_for_document(seeded.id)

    def test_previous_facts_are_cleared_before_re_extraction(
        self, conn, seeded, embeddings, model
    ):
        """A field that used to extract and no longer does should disappear,
        not linger as a stale row the interface still displays."""
        assert FactRepository(conn).read_for_document(seeded.id)
        extract_document(conn, seeded, model, embeddings)
        assert FactRepository(conn).read_for_document(seeded.id) == []
