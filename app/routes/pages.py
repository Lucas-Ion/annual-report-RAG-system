"""The HTML pages that are served"""

from __future__ import annotations

from collections import defaultdict

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse

from app.branding import logo_for
from app.config import resolve_pdf
from app.db.models import Fact
from app.db.repositories import (
    ChunkRepository,
    ConversationRepository,
    DocumentRepository,
    FactRepository,
    MessageRepository,
)
from app.ingest.fields import FIELDS
from app.routes.deps import Connection, templates

router = APIRouter()


def _by_field(facts: list[Fact]) -> dict[str, list[Fact]]:
    grouped: dict[str, list[Fact]] = defaultdict(list)
    for fact in facts:
        grouped[fact.field_key].append(fact)
    return {field.key: grouped[field.key] for field in FIELDS if grouped.get(field.key)}


@router.get("/", response_class=HTMLResponse)
def index(request: Request, conn: Connection) -> HTMLResponse:
    documents = DocumentRepository(conn).read()
    facts = FactRepository(conn)
    chunks = ChunkRepository(conn)

    rows = [
        {
            "document": document,
            "facts": _by_field(facts.read_for_document(document.id)),
            "chunks": len(chunks.read_for_document(document.id)),
            "pdf": resolve_pdf(document.filename) is not None,
            "logo": logo_for(document.company),
        }
        for document in documents
        if document.id is not None
    ]
    return templates(request).TemplateResponse(
        request,
        "documents.html",
        {"rows": rows, "fields": FIELDS},
    )


@router.get("/documents/{document_id}", response_class=HTMLResponse)
def document(request: Request, document_id: int, conn: Connection) -> HTMLResponse:
    found = DocumentRepository(conn).read_by_id(document_id)
    if found is None:
        raise HTTPException(status_code=404, detail="no such document")

    return templates(request).TemplateResponse(
        request,
        "document.html",
        {
            "document": found,
            "grouped": _by_field(FactRepository(conn).read_for_document(document_id)),
            "fields": {field.key: field for field in FIELDS},
            "chunks": len(ChunkRepository(conn).read_for_document(document_id)),
            "pdf": resolve_pdf(found.filename) is not None,
            "logo": logo_for(found.company),
        },
    )


@router.get("/documents/{document_id}/pdf")
def document_pdf(document_id: int, conn: Connection) -> FileResponse:
    found = DocumentRepository(conn).read_by_id(document_id)
    if found is None:
        raise HTTPException(status_code=404, detail="no such document")

    path = resolve_pdf(found.filename)
    if path is None:
        raise HTTPException(
            status_code=404,
            detail=f"{found.filename} is not in data/pdfs on this machine",
        )

    return FileResponse(
        path,
        media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="{found.filename}"'},
    )


@router.get("/compare", response_class=HTMLResponse)
def compare(request: Request, conn: Connection) -> HTMLResponse:
    documents = {
        document.id: document
        for document in DocumentRepository(conn).read()
        if document.id is not None
    }
    facts = FactRepository(conn)
    table = [
        {
            "field": field,
            "rows": [
                (
                    documents[fact.document_id],
                    fact,
                    logo_for(documents[fact.document_id].company),
                )
                for fact in facts.read_by_field(field.key)
                if fact.document_id in documents
            ],
        }
        for field in FIELDS
    ]
    return templates(request).TemplateResponse(
        request, "compare.html", {"table": table}
    )


@router.get("/chat", response_class=HTMLResponse)
def chat(
    request: Request, conn: Connection, conversation: int | None = None
) -> HTMLResponse:
    conversations = ConversationRepository(conn)
    history = []
    current = conversations.read_by_id(conversation) if conversation else None
    if current is not None and current.id is not None:
        history = MessageRepository(conn).read_for_conversation(current.id)

    return templates(request).TemplateResponse(
        request,
        "chat.html",
        {
            "documents": DocumentRepository(conn).read(),
            "conversations": conversations.read()[:20],
            "conversation": current,
            "history": history,
        },
    )
