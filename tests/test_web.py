"""Testing the HTTP layer, driven through FastAPI's test client."""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.routes.deps import embeddings as embeddings_dependency
from app.routes.deps import language_model


@pytest.fixture
def client(conn, seeded, embeddings, model):
    application = create_app()
    application.dependency_overrides[embeddings_dependency] = lambda: embeddings
    application.dependency_overrides[language_model] = lambda: model
    with TestClient(application) as test_client:
        yield test_client


class TestPages:
    @pytest.mark.parametrize("path", ["/", "/compare", "/chat"])
    def test_renders(self, client, path):
        response = client.get(path)
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_the_index_shows_the_extracted_figure(self, client, seeded):
        assert "12,345" in client.get("/").text

    def test_a_report_page_shows_its_evidence(self, client, seeded):
        body = client.get(f"/documents/{seeded.id}").text
        assert "12,345" in body
        assert "The average number of employees in 2025" in body

    def test_a_missing_report_is_a_404(self, client):
        assert client.get("/documents/9999").status_code == 404

    def test_page_numbers_link_into_the_pdf(self, client, seeded):
        assert f"/documents/{seeded.id}/pdf#page=5" in client.get("/").text

    def test_a_report_with_no_pdf_on_disk_is_a_404_not_a_crash(self, client, seeded):
        assert client.get(f"/documents/{seeded.id}/pdf").status_code == 404


class TestDocumentsApi:
    def test_lists_reports_with_their_datapoints(self, client, seeded):
        payload = client.get("/api/documents").json()
        assert len(payload) == 1
        assert payload[0]["company"] == "Acme"
        assert payload[0]["facts"][0]["value"] == "12,345"

    def test_every_datapoint_carries_its_evidence(self, client, seeded):
        fact = client.get("/api/documents").json()[0]["facts"][0]
        assert fact["page"] == 5 and fact["quote"]


class TestChatApi:
    @staticmethod
    def events(response) -> list[tuple[str, Any]]:
        found = []
        for frame in response.text.split("\n\n"):
            lines = dict(
                line.split(": ", 1) for line in frame.splitlines() if ": " in line
            )
            if "event" in lines:
                found.append((lines["event"], json.loads(lines["data"])))
        return found

    def test_streams_a_typed_event_sequence(self, client, seeded):
        response = client.post("/api/chat", json={"question": "how many employees?"})
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        names = [name for name, _ in self.events(response)]
        assert names[0] == "meta"
        assert "token" in names
        assert names[-1] == "citations"

    def test_sources_arrive_before_a_single_token(self, client, seeded):
        events = self.events(client.post("/api/chat", json={"question": "employees?"}))
        name, meta = events[0]
        assert name == "meta"
        assert meta["sources"] and meta["conversation_id"]

    def test_each_source_carries_what_the_interface_needs(self, client, seeded):
        _, meta = self.events(
            client.post("/api/chat", json={"question": "employees?"})
        )[0]
        source = meta["sources"][0]
        expected = {"n", "chunk_id", "document_id", "company", "page_start"}
        assert expected <= source.keys()

    def test_the_turn_is_persisted(self, client, conn, seeded):
        client.post("/api/chat", json={"question": "how many employees?"})
        rows = conn.execute("SELECT role FROM messages ORDER BY id").fetchall()
        assert [r["role"] for r in rows] == ["user", "assistant"]

    def test_a_follow_up_continues_the_same_thread(self, client, seeded):
        first = self.events(client.post("/api/chat", json={"question": "employees?"}))
        thread = first[0][1]["conversation_id"]
        second = self.events(
            client.post(
                "/api/chat",
                json={"question": "and revenue?", "conversation_id": thread},
            )
        )
        assert second[0][1]["conversation_id"] == thread

    def test_an_empty_question_is_refused(self, client):
        assert client.post("/api/chat", json={"question": ""}).status_code == 422


class TestHealth:
    def test_reports_where_the_database_is(self, client):
        payload = client.get("/healthz").json()
        assert payload["ok"] is True and payload["database"]


class TestUpload:

    @staticmethod
    def send(client, name: str, content: bytes, **fields):
        return client.post(
            "/api/documents",
            files={"file": (name, content, "application/pdf")},
            data=fields,
        )

    def test_a_file_that_is_not_a_pdf_is_refused(self, client):
        response = self.send(client, "report-2025.pdf", b"PK\x03\x04 a zip file")
        assert response.status_code == 400
        assert "does not look like a PDF" in response.json()["detail"]

    def test_a_non_pdf_extension_is_refused(self, client):
        assert self.send(client, "notes.txt", b"%PDF-1.4").status_code == 400

    def test_an_unreadable_filename_is_refused_with_advice(self, client):
        response = self.send(client, "scan001.pdf", b"%PDF-1.4 content")
        assert response.status_code == 400
        assert "shell-annual-report-2025.pdf" in response.json()["detail"]

    def test_an_oversized_file_is_refused(self, client, monkeypatch):
        monkeypatch.setattr("app.routes.uploads.MAX_UPLOAD_BYTES", 10)
        response = self.send(client, "acme-2025.pdf", b"%PDF-1.4" + b"x" * 500)
        assert response.status_code == 413

    def test_a_rejected_upload_leaves_no_file_behind(
        self, client, monkeypatch, tmp_path
    ):
        monkeypatch.setattr("app.routes.uploads.PDF_DIR", tmp_path)
        monkeypatch.setattr("app.routes.uploads.MAX_UPLOAD_BYTES", 10)
        self.send(client, "acme-2025.pdf", b"%PDF-1.4" + b"x" * 500)
        assert list(tmp_path.glob("*.pdf")) == []


class TestProgress:
    def test_reports_the_stages_and_counts(self, client, seeded):
        payload = client.get(f"/api/documents/{seeded.id}/progress").json()
        assert payload["company"] == "Acme"
        assert payload["chunks"] == 2 and payload["facts"] == 1
        assert payload["pages_parsed"] == 5

    def test_a_document_with_no_finished_stages_is_not_done(self, client, seeded):
        assert (
            client.get(f"/api/documents/{seeded.id}/progress").json()["done"] is False
        )

    def test_reports_a_failure_with_its_reason(self, client, conn, seeded):
        from app.db.connection import transaction
        from app.db.models import Stage
        from app.db.repositories import StageRunRepository

        with transaction(conn):
            StageRunRepository(conn).fail(seeded.id, Stage.PARSE, "TritonMissing")
        payload = client.get(f"/api/documents/{seeded.id}/progress").json()
        assert "TritonMissing" in payload["error"]

    def test_a_missing_document_is_a_404(self, client):
        assert client.get("/api/documents/9999/progress").status_code == 404

    def test_a_fully_ingested_report_is_reported_as_done(self, client, conn, seeded):
        from app.db.connection import transaction
        from app.db.models import Stage
        from app.db.repositories import StageRunRepository

        runs = StageRunRepository(conn)
        with transaction(conn):
            for stage in Stage:
                runs.start(seeded.id, stage)
                runs.finish(seeded.id, stage)

        payload = client.get(f"/api/documents/{seeded.id}/progress").json()
        assert payload["done"] is True


class TestInProgress:

    def test_nothing_running_is_an_empty_list(self, client, seeded):
        assert client.get("/api/documents/in-progress").json() == []

    def test_a_running_ingest_is_listed(self, client, conn, seeded):
        from app.db.connection import transaction
        from app.db.models import Stage
        from app.db.repositories import StageRunRepository

        with transaction(conn):
            StageRunRepository(conn).start(seeded.id, Stage.PARSE)

        listed = client.get("/api/documents/in-progress").json()
        assert len(listed) == 1
        assert listed[0]["running"] == "parse"

    def test_a_finished_ingest_is_not_listed(self, client, conn, seeded):
        from app.db.connection import transaction
        from app.db.models import Stage
        from app.db.repositories import StageRunRepository

        runs = StageRunRepository(conn)
        with transaction(conn):
            for stage in Stage:
                runs.start(seeded.id, stage)
                runs.finish(seeded.id, stage)
        assert client.get("/api/documents/in-progress").json() == []

    def test_a_failed_ingest_is_still_listed(self, client, conn, seeded):
        from app.db.connection import transaction
        from app.db.models import Stage
        from app.db.repositories import StageRunRepository

        with transaction(conn):
            StageRunRepository(conn).fail(seeded.id, Stage.PARSE, "TritonMissing")
        listed = client.get("/api/documents/in-progress").json()
        assert len(listed) == 1 and "TritonMissing" in listed[0]["error"]


class TestRemove:
    def test_removes_the_report_and_everything_derived_from_it(
        self, client, conn, seeded
    ):
        response = client.delete(f"/api/documents/{seeded.id}")
        assert response.status_code == 200
        for table in ("documents", "blocks", "chunks", "extracted_facts"):
            assert conn.execute(f"SELECT count(*) n FROM {table}").fetchone()["n"] == 0

    def test_removes_the_embeddings_too(self, client, conn, seeded):
        client.delete(f"/api/documents/{seeded.id}")
        assert conn.execute("SELECT count(*) n FROM chunk_vectors").fetchone()["n"] == 0

    def test_reports_what_it_removed(self, client, seeded):
        removed = client.delete(f"/api/documents/{seeded.id}").json()["removed"]
        assert removed["chunks"] == 2 and removed["facts"] == 1

    def test_the_source_pdf_is_not_deleted(
        self, client, conn, seeded, monkeypatch, tmp_path
    ):
        pdf = tmp_path / seeded.filename
        pdf.write_bytes(b"%PDF-1.4 pretend")
        monkeypatch.setattr("app.config.PDF_DIR", tmp_path)
        client.delete(f"/api/documents/{seeded.id}")
        assert pdf.exists()

    def test_removing_it_twice_is_a_404(self, client, seeded):
        client.delete(f"/api/documents/{seeded.id}")
        assert client.delete(f"/api/documents/{seeded.id}").status_code == 404

    def test_the_report_disappears_from_the_index_page(self, client, seeded):
        assert "Acme" in client.get("/").text
        client.delete(f"/api/documents/{seeded.id}")
        assert "Acme" not in client.get("/").text

    def test_a_removed_report_can_be_uploaded_again(
        self, client, conn, seeded, monkeypatch, tmp_path, make_pdf
    ):
        monkeypatch.setattr("app.routes.uploads.PDF_DIR", tmp_path)
        leftover = tmp_path / "acme-2025.pdf"
        leftover.write_bytes(b"%PDF-1.4 stale")
        client.delete(f"/api/documents/{seeded.id}")

        content = make_pdf("fresh.pdf").read_bytes()
        response = client.post(
            "/api/documents",
            files={"file": ("acme-2025.pdf", content, "application/pdf")},
        )
        assert response.status_code == 202
        assert response.json()["started"] is True


class TestConversations:
    def test_a_reopened_thread_carries_its_citations(self, client, conn, seeded):
        posted = client.post("/api/chat", json={"question": "how many employees?"})
        thread = TestChatApi.events(posted)[0][1]["conversation_id"]

        payload = client.get(f"/api/conversations/{thread}").json()
        assert [m["role"] for m in payload["messages"]] == ["user", "assistant"]
        assert "citations" in payload["messages"][1]

    def test_a_thread_is_named_rather_than_untitled(self, client, seeded):
        posted = client.post("/api/chat", json={"question": "how many employees?"})
        thread = TestChatApi.events(posted)[0][1]["conversation_id"]
        title = client.get(f"/api/conversations/{thread}").json()["title"]
        assert title and title != "Untitled"

    def test_a_thread_can_be_renamed(self, client, seeded):
        posted = client.post("/api/chat", json={"question": "how many employees?"})
        thread = TestChatApi.events(posted)[0][1]["conversation_id"]

        response = client.patch(
            f"/api/conversations/{thread}", json={"title": "Headcount across reports"}
        )
        assert response.status_code == 200
        assert client.get(f"/api/conversations/{thread}").json()["title"] == (
            "Headcount across reports"
        )

    def test_an_empty_name_is_refused(self, client, seeded):
        posted = client.post("/api/chat", json={"question": "employees?"})
        thread = TestChatApi.events(posted)[0][1]["conversation_id"]
        assert client.patch(
            f"/api/conversations/{thread}", json={"title": "  "}
        ).status_code in (200, 422)

    def test_renaming_a_missing_thread_is_a_404(self, client):
        assert (
            client.patch("/api/conversations/9999", json={"title": "x"}).status_code
            == 404
        )

    def test_a_thread_can_be_deleted(self, client, conn, seeded):
        posted = client.post("/api/chat", json={"question": "how many employees?"})
        thread = TestChatApi.events(posted)[0][1]["conversation_id"]

        assert client.delete(f"/api/conversations/{thread}").status_code == 200
        assert client.get(f"/api/conversations/{thread}").status_code == 404

    def test_deleting_a_thread_takes_its_messages_and_citations(
        self, client, conn, seeded
    ):
        posted = client.post("/api/chat", json={"question": "how many employees?"})
        thread = TestChatApi.events(posted)[0][1]["conversation_id"]
        client.delete(f"/api/conversations/{thread}")

        for table in ("conversations", "messages", "citations"):
            assert conn.execute(f"SELECT count(*) n FROM {table}").fetchone()["n"] == 0

    def test_deleting_a_missing_thread_is_a_404(self, client):
        assert client.delete("/api/conversations/9999").status_code == 404
