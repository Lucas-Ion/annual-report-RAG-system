"""JSON and streaming endpoints"""

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
    title: str = Field(min_length=1, max_length=120)


class Question(BaseModel):
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
    return f"event: {name}\ndata: {json.dumps(payload)}\n\n"


@router.get("/documents")
def documents(conn: Connection) -> list[dict[str, object]]:
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

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.patch("/conversations/{conversation_id}")
def rename(
    conversation_id: int, payload: Rename, conn: Connection
) -> dict[str, object]:
    conversations = ConversationRepository(conn)
    found = conversations.read_by_id(conversation_id)
    if found is None:
        raise HTTPException(status_code=404, detail="no such conversation")

    found.title = payload.title.strip()
    with transaction(conn):
        conversations.update(found)
    return {"id": conversation_id, "title": found.title}


@router.delete("/conversations/{conversation_id}")
def remove_conversation(conversation_id: int, conn: Connection) -> dict[str, object]:
    conversations = ConversationRepository(conn)
    found = conversations.read_by_id(conversation_id)
    if found is None:
        raise HTTPException(status_code=404, detail="no such conversation")

    with transaction(conn):
        conversations.delete(found)
    return {"id": conversation_id, "title": found.title}


@router.get("/conversations/{conversation_id}")
def conversation(conversation_id: int, conn: Connection) -> dict[str, object]:
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
