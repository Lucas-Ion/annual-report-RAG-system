"""The JSON and streaming endpoints.

The chat endpoint is the interesting one. It streams, because an answer over
eight excerpts takes ten to twenty seconds to write and an interface that sits
blank for that long reads as broken.

Server-sent events rather than a raw text stream, because the response is not
only text: the citations exist once the last token has arrived and been checked
against the sources, and they have to reach the browser too. One event type per
kind of thing solves that without a second request.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.branding import logo_for
from app.chat.answer import (
    HISTORY_TURNS,
    finish,
    prepare,
    start_conversation,
)
from app.chat.prompts import SYSTEM_PROMPT
from app.db.connection import transaction
from app.db.repositories import (
    ChunkRepository,
    CitationRepository,
    ConversationRepository,
    DocumentRepository,
    FactRepository,
    MessageRepository,
)
from app.providers import MissingApiKey
from app.routes.deps import Connection, Embeddings, Model

router = APIRouter(prefix="/api")


class Rename(BaseModel):
    """A new name for a conversation."""

    title: str = Field(min_length=1, max_length=120)


class Question(BaseModel):
    """A question, and where to put the answer."""

    question: str = Field(min_length=1, max_length=2000)
    conversation_id: int | None = Field(
        default=None, description="Existing thread, or null to start a new one."
    )
    document_id: int | None = Field(
        default=None,
        description=(
            "Restrict to one report. Left null, the question is read for a "
            "company name and narrowed automatically if it names exactly one."
        ),
    )


def _event(name: str, payload: object) -> str:
    """Format one server-sent event.

    Args:
        name: Event type, which the browser listens for by name.
        payload: JSON serialisable body.

    Returns:
        The wire format, terminated by the blank line SSE requires.
    """
    return f"event: {name}\ndata: {json.dumps(payload)}\n\n"


@router.get("/documents")
def documents(conn: Connection) -> list[dict[str, object]]:
    """List the indexed reports and their extracted datapoints.

    Args:
        conn: Database connection.

    Returns:
        One entry per report.
    """
    facts = FactRepository(conn)
    return [
        {
            "id": document.id,
            "company": document.company,
            "year": document.year,
            "filename": document.filename,
            "pages": document.page_count,
            "facts": [
                {
                    "field": fact.field_key,
                    "value": fact.value_raw,
                    "numeric": fact.value_numeric,
                    "unit": fact.unit,
                    "page": fact.page_no,
                    "quote": fact.verbatim_quote,
                    "confidence": fact.confidence,
                }
                for fact in facts.read_for_document(document.id)
            ],
        }
        for document in DocumentRepository(conn).read()
        if document.id is not None
    ]


@router.post("/chat")
def chat(
    payload: Question, conn: Connection, embeddings: Embeddings, model: Model
) -> StreamingResponse:
    """Answer a question, streaming the reply.

    Events, in order:

      * `meta`      the conversation id and the sources retrieved
      * `token`     one fragment of the answer, many of these
      * `citations` the verified citations, once the answer is complete
      * `error`     something went wrong, and the stream ends

    Sources are sent first, before a single token, so the interface can show
    what the answer is being built from while it is still being written.

    Args:
        payload: The question.
        conn: Database connection.
        embeddings: The embedding provider.
        model: The language model.

    Returns:
        An event stream.

    Raises:
        HTTPException: 503 if no API key is configured, which is a setup
            problem rather than a failure, so it says what to do about it.
    """
    conversation_id = payload.conversation_id
    if conversation_id is None:
        try:
            conversation_id = start_conversation(conn, payload.question, model).id
        except MissingApiKey as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
    assert conversation_id is not None

    def stream() -> Iterator[str]:
        """Run the turn, emitting events as it goes."""
        try:
            prompt, sources, history = prepare(
                conn,
                payload.question,
                conversation_id=conversation_id,
                embeddings=embeddings,
                document_id=payload.document_id,
            )
            yield _event(
                "meta",
                {
                    "conversation_id": conversation_id,
                    "sources": [
                        {
                            "n": number,
                            "chunk_id": chunk.id,
                            "document_id": chunk.document_id,
                            "company": chunk.context_header.split(" | ")[0],
                            "logo": logo_for(chunk.context_header.split(" | ")[0]),
                            "section": chunk.section,
                            "page_start": chunk.page_start,
                            "page_end": chunk.page_end,
                            "type": chunk.chunk_type.value,
                            "text": chunk.text[:600],
                        }
                        for number, chunk in enumerate(sources, start=1)
                    ],
                },
            )

            parts: list[str] = []
            for piece in model.stream(system=SYSTEM_PROMPT, prompt=prompt):
                parts.append(piece)
                yield _event("token", piece)

            answer = finish(
                conn,
                conversation_id=conversation_id,
                question=payload.question,
                text="".join(parts),
                sources=sources,
            )
            yield _event(
                "citations",
                [
                    {
                        "chunk_id": citation.chunk_id,
                        "page": citation.page_no,
                        "quote": citation.quote,
                        "verified": citation.verified,
                    }
                    for citation in answer.citations
                ],
            )
        except MissingApiKey as exc:
            yield _event("error", {"message": str(exc), "kind": "no_api_key"})
        except Exception as exc:  # the stream is already open, so report in band
            yield _event("error", {"message": repr(exc), "kind": "failed"})

    # Buffering is what breaks streaming behind a proxy: the whole response
    # arrives at once, several seconds late, and looks exactly like a hang.
    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.patch("/conversations/{conversation_id}")
def rename(
    conversation_id: int, payload: Rename, conn: Connection
) -> dict[str, object]:
    """Rename a conversation.

    Args:
        conversation_id: Which thread.
        payload: The new title.
        conn: Database connection.

    Returns:
        The thread's id and its new title.

    Raises:
        HTTPException: 404 if there is no such thread.
    """
    conversations = ConversationRepository(conn)
    found = conversations.read_by_id(conversation_id)
    if found is None:
        raise HTTPException(status_code=404, detail="no such conversation")

    found.title = payload.title.strip()
    with transaction(conn):
        conversations.update(found)
    return {"id": conversation_id, "title": found.title}


@router.get("/conversations/{conversation_id}")
def conversation(conversation_id: int, conn: Connection) -> dict[str, object]:
    """Return one thread's messages.

    Args:
        conversation_id: Which thread.
        conn: Database connection.

    Returns:
        The thread and its messages.

    Raises:
        HTTPException: 404 if there is no such thread.
    """
    found = ConversationRepository(conn).read_by_id(conversation_id)
    if found is None:
        raise HTTPException(status_code=404, detail="no such conversation")

    citations = CitationRepository(conn)
    chunks = ChunkRepository(conn)
    messages = []
    for message in MessageRepository(conn).read_for_conversation(conversation_id)[
        -HISTORY_TURNS * 4 :
    ]:
        assert message.id is not None
        # Citations are stored; the excerpts that were retrieved are not. So a
        # reopened conversation shows what the answer actually cited rather
        # than everything it was given, which is the more useful list anyway.
        cited = citations.read_for_message(message.id)
        sources = {
            chunk.id: chunk for chunk in chunks.read_by_ids([c.chunk_id for c in cited])
        }
        messages.append(
            {
                "role": message.role.value,
                "content": message.content,
                "citations": [
                    {
                        "chunk_id": citation.chunk_id,
                        "page": citation.page_no,
                        "quote": citation.quote,
                        "verified": citation.verified,
                        "document_id": (
                            source.document_id
                            if (source := sources.get(citation.chunk_id))
                            else None
                        ),
                        "company": (
                            source.context_header.split(" | ")[0] if source else None
                        ),
                        "section": source.section if source else None,
                    }
                    for citation in cited
                ],
            }
        )
    return {"id": found.id, "title": found.title, "messages": messages}
