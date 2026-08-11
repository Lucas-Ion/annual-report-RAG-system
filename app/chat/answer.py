"""Answering a question with retrieve, ask, verify, store."""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

from app.chat.prompts import SYSTEM_PROMPT, build_prompt, build_title_prompt
from app.db.connection import transaction
from app.db.models import Block, Chunk, Citation, Conversation, Message, Role
from app.db.repositories import (
    BlockRepository,
    ChunkRepository,
    CitationRepository,
    ConversationRepository,
    DocumentRepository,
    FactRepository,
    MessageRepository,
)
from app.providers.base import EmbeddingProvider, TextGenerator
from app.retrieve import detect_document, search
from app.verify import find_source, locate_page

DEFAULT_SOURCES = 8

HISTORY_TURNS = 6

TITLE_TOKENS = 256

TITLE_FALLBACK_WORDS = 6

FACT_SOURCES = 4

_CITATION = re.compile(r"\[(\d+)(?:\s*:\s*[\"“]([^\"”]*)[\"”])?\]")


@dataclass(slots=True, kw_only=True)
class Answer:
    text: str
    message: Message
    citations: list[Citation] = field(default_factory=list)
    sources: list[Chunk] = field(default_factory=list)


def parse_citations(
    text: str, sources: Sequence[Chunk], blocks: Sequence[Block] = ()
) -> list[Citation]:
    citations: list[Citation] = []
    for match in _CITATION.finditer(text):
        index = int(match.group(1)) - 1
        quote = (match.group(2) or "").strip()

        cited = sources[index] if 0 <= index < len(sources) else None
        source = find_source(quote, sources) if quote else None

        chunk = source or cited
        if chunk is None or chunk.id is None:
            continue

        citations.append(
            Citation(
                message_id=0,
                chunk_id=chunk.id,
                quote=quote,
                page_no=(
                    locate_page(quote, chunk, blocks) if source else chunk.page_start
                ),
                verified=source is not None,
            )
        )
    return citations


def spanning_blocks(conn: sqlite3.Connection, chunks: Sequence[Chunk]) -> list[Block]:
    repository = BlockRepository(conn)
    found: list[Block] = []
    for chunk in chunks:
        if chunk.page_end != chunk.page_start:
            found.extend(
                repository.read_page_range(
                    chunk.document_id, chunk.page_start, chunk.page_end
                )
            )
    return found


def fact_backed_chunks(
    conn: sqlite3.Connection, document_id: int, limit: int = FACT_SOURCES
) -> list[Chunk]:
    facts = FactRepository(conn).read_for_document(document_id)
    chunk_ids = list(
        dict.fromkeys(fact.chunk_id for fact in facts if fact.chunk_id is not None)
    )
    return ChunkRepository(conn).read_by_ids(chunk_ids[:limit])


def retrieval_query(
    question: str, history: Sequence[Message], *, scoped: bool = False
) -> str:
    if scoped:
        return question
    previous = [message for message in history if message.role is Role.USER]
    if not previous or len(question.split()) > 6:
        return question
    return f"{previous[-1].content} {question}"


def prepare(
    conn: sqlite3.Connection,
    question: str,
    *,
    conversation_id: int,
    embeddings: EmbeddingProvider,
    sources: int = DEFAULT_SOURCES,
    document_id: int | None = None,
) -> tuple[str, list[Chunk], list[Message]]:
    history = MessageRepository(conn).read_for_conversation(conversation_id)

    if document_id is None:
        named = detect_document(question, DocumentRepository(conn).read())
        document_id = named.id if named else None

    chunks = search(
        ChunkRepository(conn),
        embeddings,
        retrieval_query(question, history, scoped=document_id is not None),
        limit=sources,
        document_id=document_id,
    )

    if document_id is not None:
        seen = {chunk.id for chunk in chunks}
        chunks += [
            chunk
            for chunk in fact_backed_chunks(conn, document_id)
            if chunk.id not in seen
        ]

    return build_prompt(question, chunks, history[-HISTORY_TURNS:]), chunks, history


def finish(
    conn: sqlite3.Connection,
    *,
    conversation_id: int,
    question: str,
    text: str,
    sources: Sequence[Chunk],
) -> Answer:
    messages = MessageRepository(conn)
    citation_repository = CitationRepository(conn)
    citations = parse_citations(text, sources, spanning_blocks(conn, sources))

    with transaction(conn):
        messages.create(
            Message(conversation_id=conversation_id, role=Role.USER, content=question)
        )
        stored = messages.create(
            Message(conversation_id=conversation_id, role=Role.ASSISTANT, content=text)
        )
        assert stored.id is not None
        saved = [
            citation_repository.create(
                Citation(
                    message_id=stored.id,
                    chunk_id=citation.chunk_id,
                    quote=citation.quote,
                    page_no=citation.page_no,
                    verified=citation.verified,
                )
            )
            for citation in citations
        ]

    return Answer(text=text, message=stored, citations=saved, sources=list(sources))


def ask(
    conn: sqlite3.Connection,
    question: str,
    *,
    conversation_id: int,
    model: TextGenerator,
    embeddings: EmbeddingProvider,
    sources: int = DEFAULT_SOURCES,
    document_id: int | None = None,
) -> Answer:
    prompt, chunks, _history = prepare(
        conn,
        question,
        conversation_id=conversation_id,
        embeddings=embeddings,
        sources=sources,
        document_id=document_id,
    )
    text = model.complete(system=SYSTEM_PROMPT, prompt=prompt)
    return finish(
        conn,
        conversation_id=conversation_id,
        question=question,
        text=text,
        sources=chunks,
    )


def ask_streaming(
    conn: sqlite3.Connection,
    question: str,
    *,
    conversation_id: int,
    model: TextGenerator,
    embeddings: EmbeddingProvider,
    sources: int = DEFAULT_SOURCES,
    document_id: int | None = None,
) -> Iterator[str]:
    prompt, chunks, _history = prepare(
        conn,
        question,
        conversation_id=conversation_id,
        embeddings=embeddings,
        sources=sources,
        document_id=document_id,
    )

    parts: list[str] = []
    for piece in model.stream(system=SYSTEM_PROMPT, prompt=prompt):
        parts.append(piece)
        yield piece

    finish(
        conn,
        conversation_id=conversation_id,
        question=question,
        text="".join(parts),
        sources=chunks,
    )


def fallback_title(question: str) -> str:
    words = question.strip().split()
    if not words:
        return "New conversation"
    short = " ".join(words[:TITLE_FALLBACK_WORDS]).rstrip("?.,:;")
    return short + ("…" if len(words) > TITLE_FALLBACK_WORDS else "")


def start_conversation(
    conn: sqlite3.Connection, question: str, model: TextGenerator | None = None
) -> Conversation:
    title = fallback_title(question)
    if model is not None:
        try:
            suggested = (
                model.complete(
                    system="You write short, plain titles.",
                    prompt=build_title_prompt(question),
                    max_tokens=TITLE_TOKENS,
                )
                .strip()
                .strip('"')
                .strip()
            )
            if suggested:
                title = suggested
        except Exception:
            pass

    conversations = ConversationRepository(conn)
    with transaction(conn):
        return conversations.create(Conversation(title=title))
