"""Answering: testing citation parsing, page attribution, and follow-up handling."""

from __future__ import annotations

from app.chat.answer import (
    ask,
    fact_backed_chunks,
    fallback_title,
    parse_citations,
    retrieval_query,
    spanning_blocks,
    start_conversation,
)
from app.chat.prompts import build_prompt, format_sources
from app.db.connection import transaction
from app.db.models import Block, Chunk, ChunkType, Conversation, Message, Role
from app.db.repositories import ConversationRepository
from tests.conftest import FakeModel, ident


def chunk(
    chunk_id: int, text: str, page_start: int, page_end: int | None = None
) -> Chunk:
    return Chunk(
        id=chunk_id,
        document_id=7,
        seq=chunk_id,
        page_start=page_start,
        page_end=page_end or page_start,
        chunk_type=ChunkType.PROSE,
        context_header="Acme | Annual Report 2025",
        text=text,
    )


SOURCES = [
    chunk(10, "We aim to be a net-zero emissions business by 2050.", 358),
    chunk(11, "Scope 1 and 2 emissions halved by 2030 against a 2016 baseline.", 87),
]


class TestParseCitations:
    def test_a_quoted_marker_is_verified(self):
        found = parse_citations(
            'Acme targets [1: "net-zero emissions business by 2050"].', SOURCES
        )
        assert len(found) == 1
        assert found[0].verified is True
        assert found[0].page_no == 358 and found[0].chunk_id == 10

    def test_typographic_quotes_are_accepted(self):
        found = parse_citations("x [2: “halved by 2030”].", SOURCES)
        assert found[0].verified is True

    def test_an_invented_quotation_is_not_verified(self):
        found = parse_citations(
            'Acme will halve emissions [1: "net-zero by 2049"].', SOURCES
        )
        assert found[0].verified is False

    def test_a_misnumbered_but_real_quote_is_verified_against_the_right_page(self):
        found = parse_citations('[9: "halved by 2030"]', SOURCES)
        assert found[0].verified is True and found[0].page_no == 87

    def test_a_bare_marker_is_recorded_but_not_verified(self):
        found = parse_citations("a claim [2].", SOURCES)
        assert found[0].verified is False and found[0].page_no == 87

    def test_a_marker_pointing_nowhere_is_dropped(self):
        assert parse_citations("[9]", SOURCES) == []

    def test_several_markers_are_all_parsed(self):
        found = parse_citations('a [1: "by 2050"] and b [2: "by 2030"]', SOURCES)
        assert len(found) == 2

    def test_prose_with_no_markers_produces_no_citations(self):
        assert parse_citations("a plain answer", SOURCES) == []

    def test_a_quotation_is_pinned_to_the_page_it_is_printed_on(self):
        spanning = chunk(12, "tail of page four\n\nthe figure was 12,345", 4, 5)
        blocks = [
            Block(
                document_id=7, seq=0, page_no=4, label="text", text="tail of page four"
            ),
            Block(
                document_id=7,
                seq=1,
                page_no=5,
                label="text",
                text="the figure was 12,345",
            ),
        ]
        found = parse_citations('[1: "the figure was 12,345"]', [spanning], blocks)
        assert found[0].page_no == 5


class TestRetrievalQuery:
    HISTORY = [
        Message(
            conversation_id=1,
            role=Role.USER,
            content="How many employees does Shell have?",
        ),
        Message(conversation_id=1, role=Role.ASSISTANT, content="85,000."),
    ]

    def test_a_bare_follow_up_borrows_the_previous_question(self):
        assert "Shell" in retrieval_query("what about 2024?", self.HISTORY)

    def test_a_question_naming_its_own_subject_stands_alone(self):
        assert (
            retrieval_query("and Heineken?", self.HISTORY, scoped=True)
            == "and Heineken?"
        )

    def test_a_full_question_is_left_alone(self):
        question = "What are ASML's climate targets for 2030?"
        assert retrieval_query(question, self.HISTORY) == question

    def test_the_first_question_has_nothing_to_borrow(self):
        assert retrieval_query("hello?", []) == "hello?"


class TestPrompts:
    def test_sources_are_numbered_to_match_the_citation_syntax(self):
        rendered = format_sources(SOURCES)
        assert "--- source 1 |" in rendered and "--- source 2 |" in rendered

    def test_each_source_carries_its_company_and_page(self):
        rendered = format_sources(SOURCES)
        assert "Acme" in rendered and "page 358" in rendered

    def test_with_no_sources_the_prompt_says_so(self):
        assert "No excerpts were found" in build_prompt("anything", [])

    def test_history_is_included_when_there_is_some(self):
        prompt = build_prompt("q", SOURCES, TestRetrievalQuery.HISTORY)
        assert "Earlier in this conversation" in prompt


class TestContextAssembly:
    def test_extraction_evidence_is_offered_back_to_chat(self, conn, seeded):
        found = fact_backed_chunks(conn, seeded.id)
        assert found and "12,345" in found[0].text

    def test_only_page_spanning_chunks_cost_a_block_lookup(self, conn, seeded):
        single = chunk(1, "x", 9)
        assert spanning_blocks(conn, [single]) == []

    def test_a_full_turn_is_stored_atomically(self, conn, seeded, model, embeddings):
        with transaction(conn):
            thread = ConversationRepository(conn).create(Conversation(title="t"))
        answer = ask(
            conn,
            "how many employees?",
            conversation_id=ident(thread),
            model=model,
            embeddings=embeddings,
        )
        assert answer.message.id is not None
        rows = conn.execute("SELECT role, content FROM messages ORDER BY id").fetchall()
        assert [r["role"] for r in rows] == ["user", "assistant"]


class TestTitles:

    def test_a_title_comes_from_the_question_when_there_is_no_model(self, conn):
        thread = start_conversation(conn, "How many employees does Heineken have?")
        assert thread.title and thread.title != "Untitled"
        assert "Heineken" in thread.title

    def test_the_models_suggestion_is_preferred(self, conn):
        thread = start_conversation(
            conn, "How many employees?", FakeModel(reply="Heineken Employee Count")
        )
        assert thread.title == "Heineken Employee Count"

    def test_surrounding_quotes_are_stripped(self):
        assert fallback_title('"a title"') == '"a title"'

    def test_a_model_failure_falls_back_rather_than_losing_the_thread(self, conn):
        class Broken:
            def complete(self, **kwargs):
                raise RuntimeError("hit max_tokens before producing any text")

            def stream(self, **kwargs):
                yield ""

        thread = start_conversation(conn, "What is the KLM fleet size?", Broken())
        assert thread.title and "Untitled" not in thread.title

    def test_a_long_question_is_shortened(self):
        title = fallback_title(
            "How much did Shell spend on climate change adaptation "
            "during the 2025 reporting year in total"
        )
        assert title.endswith("…") and len(title.split()) <= 7

    def test_an_empty_question_still_gets_a_name(self):
        assert fallback_title("   ") == "New conversation"
