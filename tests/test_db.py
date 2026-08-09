"""The persistence layer.

Two properties matter more than the CRUD: that deleting a document takes its
embeddings with it, and that a failed transaction leaves nothing behind. Both
fail silently if they break, which is why they are tested rather than assumed.
"""

from __future__ import annotations

import pytest

from app.db.connection import transaction
from app.db.models import (
    Block,
    Citation,
    Conversation,
    Document,
    Fact,
    Message,
    Role,
    Stage,
    StageStatus,
)
from app.db.repositories import (
    BlockRepository,
    ChunkRepository,
    CitationRepository,
    ConversationRepository,
    DocumentRepository,
    FactRepository,
    MessageRepository,
    StageRunRepository,
)
from tests.conftest import ident


class TestConnection:
    def test_foreign_keys_are_enforced(self, conn):
        """Off by default in SQLite. Without it every ON DELETE CASCADE in the
        schema is decoration and orphans accumulate silently."""
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1

    def test_write_ahead_logging_is_on(self, conn):
        """Otherwise a running ingest blocks every read and the interface
        appears to hang for the length of a parse."""
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"

    def test_the_schema_is_idempotent(self, conn):
        """Applied on every startup, which is what replaces migrations."""
        from app.db.connection import _SCHEMA_PATH

        conn.executescript(_SCHEMA_PATH.read_text())

    def test_rows_are_addressable_by_column_name(self, conn):
        assert conn.execute("SELECT 1 AS n").fetchone()["n"] == 1


class TestDocuments:
    def test_create_fills_in_the_database_defaults(self, document):
        assert document.id is not None
        assert document.created_at

    def test_found_by_content_hash(self, conn, document):
        found = DocumentRepository(conn).read_by_hash("hash-acme")
        assert found is not None and found.id == document.id

    def test_an_unknown_hash_is_not_an_error(self, conn):
        assert DocumentRepository(conn).read_by_hash("nothing") is None

    def test_the_same_file_cannot_be_registered_twice(self, conn, document):
        import sqlite3

        with pytest.raises(sqlite3.IntegrityError), transaction(conn):
            DocumentRepository(conn).create(
                Document(
                    filename="other.pdf",
                    file_hash="hash-acme",
                    company="Acme",
                    year=2025,
                )
            )


class TestStageRuns:
    def test_a_running_stage_is_not_done(self, conn, document):
        runs = StageRunRepository(conn)
        with transaction(conn):
            runs.start(document.id, Stage.PARSE)
        assert runs.is_done(document.id, Stage.PARSE) is False

    def test_a_finished_stage_is_done_and_keeps_its_start_time(self, conn, document):
        runs = StageRunRepository(conn)
        with transaction(conn):
            runs.start(document.id, Stage.PARSE)
            stored = runs.finish(document.id, Stage.PARSE)
        assert runs.is_done(document.id, Stage.PARSE)
        assert stored.started_at and stored.finished_at

    def test_a_failed_stage_is_not_done_and_keeps_the_reason(self, conn, document):
        runs = StageRunRepository(conn)
        with transaction(conn):
            runs.start(document.id, Stage.EMBED)
            stored = runs.fail(document.id, Stage.EMBED, "RateLimitError")
        assert runs.is_done(document.id, Stage.EMBED) is False
        assert stored.status is StageStatus.FAILED
        assert stored.error == "RateLimitError"

    def test_restarting_replaces_rather_than_duplicates(self, conn, document):
        runs = StageRunRepository(conn)
        with transaction(conn):
            runs.start(document.id, Stage.PARSE)
            runs.finish(document.id, Stage.PARSE)
            runs.start(document.id, Stage.PARSE)
        assert len(runs.read_for_document(document.id)) == 1


class TestBlocks:
    def test_bulk_insert_and_read_back_in_order(self, conn, seeded):
        stored = BlockRepository(conn).read_for_document(seeded.id)
        assert [b.seq for b in stored] == [0, 1, 2, 3]

    def test_the_resume_point_is_the_highest_page(self, conn, seeded):
        assert BlockRepository(conn).last_parsed_page(seeded.id) == 5

    def test_numbering_continues_from_where_it_stopped(self, conn, seeded):
        assert BlockRepository(conn).next_seq(seeded.id) == 4

    def test_an_unparsed_document_has_no_resume_point(self, conn, document):
        assert BlockRepository(conn).last_parsed_page(document.id) is None
        assert BlockRepository(conn).next_seq(document.id) == 0

    def test_a_bounding_box_survives_the_round_trip(self, conn, document):
        with transaction(conn):
            BlockRepository(conn).create(
                Block(
                    document_id=document.id,
                    seq=99,
                    page_no=1,
                    label="text",
                    text="x",
                    bbox=(10.0, 20.0, 30.0, 40.0),
                )
            )
        stored = BlockRepository(conn).read_for_document(document.id)[-1]
        assert stored.bbox == (10.0, 20.0, 30.0, 40.0)

    def test_a_generator_is_not_consumed_before_it_is_stored(self, conn, document):
        """create_all accepts any iterable. Walking it twice would silently
        store the rows and return nothing."""
        rows = (
            Block(document_id=document.id, seq=i, page_no=1, label="text", text=f"b{i}")
            for i in range(3)
        )
        assert len(BlockRepository(conn).create_all(rows)) == 3


class TestChunksAndSearch:
    def test_keyword_search_finds_a_chunk(self, conn, seeded):
        hits = ChunkRepository(conn).read_by_keywords("workforce")
        assert hits and "workforce" in hits[0].chunk.text

    def test_keyword_scores_are_higher_is_better(self, conn, seeded):
        """BM25 comes out of SQLite negative, with more negative meaning
        better. The sign is flipped so every search agrees."""
        hits = ChunkRepository(conn).read_by_keywords("workforce")
        assert hits[0].score > 0

    def test_question_words_do_not_decide_the_ranking(self, conn, seeded):
        """Joined by OR, stopwords let an irrelevant chunk win on 'how' and
        'much' alone."""
        from app.db.repositories.chunks import to_match_expression

        assert to_match_expression("How many employees") == '"employees"'

    def test_a_query_of_only_stopwords_still_searches(self, conn, seeded):
        from app.db.repositories.chunks import to_match_expression

        assert to_match_expression("how much") == '"how" OR "much"'

    @pytest.mark.parametrize(
        "junk", ['"" AND NEAR(', "a OR b) NOT", "***", 'NEAR/2 "x', "^ * ()"]
    )
    def test_malformed_input_never_raises(self, conn, seeded, junk):
        """FTS5 has its own query language, so raw user text is a syntax error
        waiting to happen."""
        ChunkRepository(conn).read_by_keywords(junk)

    def test_vector_search_returns_the_nearest_chunk(self, conn, seeded, embeddings):
        query = embeddings.embed_query("average number of employees")
        hits = ChunkRepository(conn).read_by_similarity(query, limit=1)
        assert hits and "12,345" in hits[0].chunk.text

    def test_read_by_ids_preserves_the_order_asked_for(self, conn, seeded):
        chunks = ChunkRepository(conn)
        ids = [ident(c) for c in chunks.read_for_document(seeded.id)]
        assert [c.id for c in chunks.read_by_ids(list(reversed(ids)))] == list(
            reversed(ids)
        )

    def test_embedded_chunks_are_not_offered_for_embedding_again(self, conn, seeded):
        assert ChunkRepository(conn).read_without_embeddings(seeded.id) == []


class TestCascades:
    def test_deleting_a_document_removes_everything_derived_from_it(self, conn, seeded):
        with transaction(conn):
            DocumentRepository(conn).delete(seeded)
        for table in ("blocks", "chunks", "extracted_facts", "stage_runs"):
            assert conn.execute(f"SELECT count(*) n FROM {table}").fetchone()["n"] == 0

    def test_deleting_a_document_removes_its_embeddings(self, conn, seeded):
        """The one the database cannot do for you. A vec0 table cannot declare
        a foreign key, so nothing cascades into it and orphaned vectors would
        keep being returned by searches."""
        with transaction(conn):
            DocumentRepository(conn).delete(seeded)
        assert conn.execute("SELECT count(*) n FROM chunk_vectors").fetchone()["n"] == 0

    def test_the_keyword_index_cleans_itself_up(self, conn, seeded):
        with transaction(conn):
            DocumentRepository(conn).delete(seeded)
        assert ChunkRepository(conn).read_by_keywords("workforce") == []


class TestTransactions:
    def test_a_failed_transaction_leaves_nothing_behind(self, conn):
        with pytest.raises(RuntimeError), transaction(conn):
            DocumentRepository(conn).create(
                Document(filename="x.pdf", file_hash="rollback", company="X", year=2025)
            )
            raise RuntimeError("boom")
        assert DocumentRepository(conn).read_by_hash("rollback") is None


class TestChat:
    def test_messages_come_back_in_the_order_they_were_said(self, conn):
        conversations, messages = ConversationRepository(conn), MessageRepository(conn)
        with transaction(conn):
            thread = conversations.create(Conversation(title="t"))
            first = messages.create(
                Message(conversation_id=ident(thread), role=Role.USER, content="q")
            )
            second = messages.create(
                Message(conversation_id=ident(thread), role=Role.ASSISTANT, content="a")
            )
        assert [m.id for m in messages.read_for_conversation(ident(thread))] == [
            first.id,
            second.id,
        ]

    def test_the_verified_flag_survives_as_a_boolean(self, conn, seeded):
        """SQLite has no boolean type, so it is stored as 0 or 1."""
        conversations, messages = ConversationRepository(conn), MessageRepository(conn)
        chunk = ChunkRepository(conn).read_for_document(seeded.id)[0]
        with transaction(conn):
            thread = conversations.create(Conversation())
            message = messages.create(
                Message(conversation_id=ident(thread), role=Role.ASSISTANT, content="a")
            )
            CitationRepository(conn).create(
                Citation(
                    message_id=ident(message),
                    chunk_id=ident(chunk),
                    quote="q",
                    page_no=1,
                    verified=True,
                )
            )
        stored = CitationRepository(conn).read_for_message(ident(message))[0]
        assert stored.verified is True


class TestFacts:
    def test_a_field_can_be_compared_across_reports(self, conn, seeded):
        """The reason extracted_facts stores a key rather than a column per
        field: five companies side by side is one query."""
        assert [f.value_raw for f in FactRepository(conn).read_by_field("fte")] == [
            "12,345"
        ]

    def test_several_values_for_one_field_are_allowed(self, conn, seeded):
        """A company has one headcount and many sustainability goals."""
        facts = FactRepository(conn)
        with transaction(conn):
            for i in range(3):
                facts.create(
                    Fact(
                        document_id=seeded.id,
                        field_key="sustainability_goal",
                        verbatim_quote=f"goal {i}",
                        page_no=1,
                        extractor_version="test",
                    )
                )
        assert len(facts.read_by_field("sustainability_goal")) == 3
