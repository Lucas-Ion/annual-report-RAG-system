"""The HTML pages.

Three of them. A list of the indexed reports with everything extracted from
each, one page per report showing those datapoints with their sources, and the
chat.

Handlers here do nothing but read and render. Anything that thinks is a call
into app/db, app/retrieve or app/chat, which is what keeps this file short
enough to be obviously correct.
"""

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
    """Group facts by field, keeping the registry's order.

    Args:
        facts: A document's extracted facts.

    Returns:
        Field key to facts, with fields that produced nothing left out.
    """
    grouped: dict[str, list[Fact]] = defaultdict(list)
    for fact in facts:
        grouped[fact.field_key].append(fact)
    return {field.key: grouped[field.key] for field in FIELDS if grouped.get(field.key)}


@router.get("/", response_class=HTMLResponse)
def index(request: Request, conn: Connection) -> HTMLResponse:
    """List the indexed reports and what was extracted from each.

    This is the page the brief means by pre-extracted data being visible.
    Everything on it was computed at ingest and is read straight from the
    database, so it needs no API key and renders in milliseconds.

    Args:
        request: The incoming request.
        conn: Database connection.

    Returns:
        The rendered page.
    """
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
    """Show one report's extracted datapoints, with their evidence.

    Args:
        request: The incoming request.
        document_id: Which report.
        conn: Database connection.

    Returns:
        The rendered page.

    Raises:
        HTTPException: 404 if there is no such report.
    """
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
    """Serve a report's source PDF, for opening in a browser tab.

    Sent inline rather than as a download, which is what lets a link carry a
    #page=N fragment and land the reader on the page a figure was taken from.
    Every page badge in this application points here, so a quoted number is one
    click from the paragraph it was quoted out of.

    Args:
        document_id: Which report.
        conn: Database connection.

    Returns:
        The PDF.

    Raises:
        HTTPException: 404 if there is no such report, or if its PDF is not on
            disk. The second is not a bug: the database ships seeded and the
            source files can legitimately be absent.
    """
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
    """Put one field side by side across every report.

    The reason extracted_facts stores a field key rather than a column per
    field: comparing five companies is one query.

    Args:
        request: The incoming request.
        conn: Database connection.

    Returns:
        The rendered page.
    """
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
    """The chat interface.

    Args:
        request: The incoming request.
        conn: Database connection.
        conversation: A thread to reopen, or None to start fresh.

    Returns:
        The rendered page.
    """
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
