"""The pipeline and the embed stage."""

from __future__ import annotations

import pytest

from app.db.connection import transaction
from app.db.models import Chunk, ChunkType, Stage, StageStatus
from app.db.repositories import ChunkRepository, StageRunRepository
from app.ingest.embed import embed_document
from app.ingest.pipeline import DEFAULT_STAGES, ingest, register_document
from tests.conftest import ident


class TestEmbedDocument:
    def test_embeds_everything_that_has_no_vector(self, conn, document, embeddings):
        chunks = ChunkRepository(conn)
        with transaction(conn):
            chunks.create_all(
                [
                    Chunk(
                        document_id=ident(document),
                        seq=i,
                        page_start=1,
                        page_end=1,
                        chunk_type=ChunkType.PROSE,
                        context_header="Acme | 2025",
                        text=f"chunk number {i}",
                    )
                    for i in range(5)
                ]
            )
        assert embed_document(conn, document, embeddings) == 5
        assert chunks.read_without_embeddings(ident(document)) == []

    def test_running_again_does_nothing(self, conn, seeded, embeddings):
        assert embed_document(conn, seeded, embeddings) == 0

    def test_only_the_missing_chunks_are_embedded(self, conn, seeded, embeddings):
        chunks = ChunkRepository(conn)
        with transaction(conn):
            chunks.create(
                Chunk(
                    document_id=ident(seeded),
                    seq=9,
                    page_start=1,
                    page_end=1,
                    chunk_type=ChunkType.PROSE,
                    context_header="Acme | 2025",
                    text="a new chunk",
                )
            )
        assert embed_document(conn, seeded, embeddings) == 1

    def test_a_document_with_no_chunks_is_not_an_error(
        self, conn, document, embeddings
    ):
        assert embed_document(conn, document, embeddings) == 0

    def test_commits_in_batches_so_an_interruption_keeps_what_it_did(
        self, conn, document, embeddings
    ):
        chunks = ChunkRepository(conn)
        with transaction(conn):
            chunks.create_all(
                [
                    Chunk(
                        document_id=ident(document),
                        seq=i,
                        page_start=1,
                        page_end=1,
                        chunk_type=ChunkType.PROSE,
                        context_header="Acme | 2025",
                        text=f"chunk {i}",
                    )
                    for i in range(7)
                ]
            )

        class FailsHalfway:
            def __init__(self):
                self.calls = 0

            dimensions = embeddings.dimensions

            def embed_documents(self, texts):
                self.calls += 1
                if self.calls > 1:
                    raise RuntimeError("provider fell over")
                return embeddings.embed_documents(texts)

            def embed_query(self, text):
                return embeddings.embed_query(text)

        with pytest.raises(RuntimeError):
            embed_document(conn, document, FailsHalfway(), batch_size=3)
        assert len(chunks.read_without_embeddings(ident(document))) == 4

    def test_an_unsaved_document_is_refused(self, embeddings, conn):
        from app.db.models import Document

        with pytest.raises(ValueError):
            embed_document(
                conn,
                Document(filename="x.pdf", file_hash="h", company="X", year=2025),
                embeddings,
            )


class TestRegisterDocument:
    def test_a_new_file_is_created_with_its_page_count(self, conn, make_pdf):
        document, created = register_document(conn, make_pdf(pages=3), "Acme", 2025)
        assert created is True
        assert document.id is not None and document.page_count == 3

    def test_the_same_contents_are_recognised_under_a_different_name(
        self, conn, make_pdf, tmp_path
    ):
        first = make_pdf("acme-2025.pdf")
        renamed = tmp_path / "acme annual report (1).pdf"
        renamed.write_bytes(first.read_bytes())

        original, _ = register_document(conn, first, "Acme", 2025)
        again, created = register_document(conn, renamed, "Acme", 2025)
        assert created is False and again.id == original.id

    def test_edited_contents_under_the_same_name_are_a_new_document(
        self, conn, make_pdf
    ):
        original, _ = register_document(conn, make_pdf(pages=3), "Acme", 2025)
        edited, created = register_document(conn, make_pdf(pages=9), "Acme", 2025)
        assert created is True and edited.id != original.id


class TestIngest:
    @staticmethod
    def pdf(tmp_path):
        import pymupdf

        path = tmp_path / "acme-2025.pdf"
        document = pymupdf.open()
        for _ in range(3):
            document.new_page()
        document.save(path)
        document.close()
        return path

    def test_finished_stages_are_skipped(self, conn, make_pdf, embeddings, model):
        pdf = make_pdf()
        document, _ = register_document(conn, pdf, "Acme", 2025)
        runs = StageRunRepository(conn)
        with transaction(conn):
            for stage in DEFAULT_STAGES:
                runs.start(ident(document), stage)
                runs.finish(ident(document), stage)

        seen: list[str] = []
        ingest(
            conn,
            pdf,
            company="Acme",
            year=2025,
            embeddings=embeddings,
            model=model,
            on_progress=lambda stage, message: seen.append(message),
        )
        assert all("skipping" in message or "registered" in message for message in seen)

    def test_embedding_is_skipped_rather_than_failing_without_a_provider(
        self, conn, make_pdf, model
    ):
        pdf = make_pdf()
        seen: list[tuple[str, str]] = []
        ingest(
            conn,
            pdf,
            company="Acme",
            year=2025,
            embeddings=None,
            model=model,
            stages=[Stage.EMBED],
            on_progress=lambda stage, message: seen.append((stage.value, message)),
        )
        assert ("embed", "no embedding provider, skipping") in seen

    def test_extraction_is_skipped_without_a_model(self, conn, make_pdf, embeddings):
        pdf = make_pdf()
        seen: list[tuple[str, str]] = []
        ingest(
            conn,
            pdf,
            company="Acme",
            year=2025,
            embeddings=embeddings,
            model=None,
            stages=[Stage.EXTRACT],
            on_progress=lambda stage, message: seen.append((stage.value, message)),
        )
        assert seen[-1][0] == "extract" and "skipping" in seen[-1][1]

    def test_a_failing_stage_is_recorded_before_the_error_propagates(
        self, conn, make_pdf, embeddings, model, monkeypatch
    ):
        pdf = make_pdf()

        def explode(*args, **kwargs):
            raise RuntimeError("layout model fell over")

        monkeypatch.setattr("app.ingest.pipeline.parse_document", explode)
        with pytest.raises(RuntimeError):
            ingest(
                conn,
                pdf,
                company="Acme",
                year=2025,
                embeddings=embeddings,
                model=model,
                stages=[Stage.PARSE],
            )

        document = register_document(conn, pdf, "Acme", 2025)[0]
        run = StageRunRepository(conn).read_by_id((ident(document), Stage.PARSE))
        assert run is not None
        assert run.status is StageStatus.FAILED
        assert "layout model fell over" in (run.error or "")

    def test_a_later_stage_is_not_attempted_after_a_failure(
        self, conn, make_pdf, embeddings, model, monkeypatch
    ):
        pdf = make_pdf()
        monkeypatch.setattr(
            "app.ingest.pipeline.parse_document",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        with pytest.raises(RuntimeError):
            ingest(
                conn, pdf, company="Acme", year=2025, embeddings=embeddings, model=model
            )

        document = register_document(conn, pdf, "Acme", 2025)[0]
        runs = StageRunRepository(conn)
        assert runs.read_by_id((ident(document), Stage.CHUNK)) is None

    def test_a_missing_file_is_refused_before_any_row_is_written(
        self, conn, tmp_path, embeddings, model
    ):
        with pytest.raises(FileNotFoundError):
            ingest(
                conn,
                tmp_path / "nope.pdf",
                company="Acme",
                year=2025,
                embeddings=embeddings,
                model=model,
            )
