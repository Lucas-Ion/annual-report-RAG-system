"""Answering a question: retrieve, ask, verify, store.

The same guarantee as extraction, applied to prose. A quotation in an answer is
checked against the excerpt it claims to come from before it is treated as a
citation, so the interface can show sources that were confirmed rather than
sources that were asserted.

Streaming is the reason this is shaped the way it is. An answer over eight
excerpts takes several seconds to write and a blank interface for that long
reads as broken, so the model's output has to be yielded as it arrives.
Verification and storage therefore happen after the last token rather than
before the first, which is why prepare() and finish() are separate: the
streaming and non-streaming paths share both ends and differ only in the middle.
"""

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

# Excerpts sent with a question. Eight is roughly 4,000 tokens of context,
# enough to hold a figure stated in one place and tabulated in another without
# burying the answer in near duplicates.
DEFAULT_SOURCES = 8

# How many earlier turns are shown to the model.
HISTORY_TURNS = 6

# Budget for naming a conversation. Generous for three or four words, because
# this model thinks before it answers and thinking spends the same allowance.
# At 32 the whole budget went to a thinking block, the reply came back empty,
# and every conversation was called Untitled.
TITLE_TOKENS = 256

# Longest fallback title, when the model cannot be reached.
TITLE_FALLBACK_WORDS = 6

# Extra excerpts drawn from the pre-extracted facts, when a question is about
# one report. Appended rather than substituted, so they never displace a search
# result, and capped because a document with twenty six sustainability goals
# would otherwise fill the prompt on its own.
FACT_SOURCES = 4

# Matches "[3]" and '[3: "quoted text"]'. Straight and curly quotes both,
# because a model writing prose will often use typographic ones and losing a
# citation to a punctuation preference would be a silly way to fail.
_CITATION = re.compile(r"\[(\d+)(?:\s*:\s*[\"“]([^\"”]*)[\"”])?\]")


@dataclass(slots=True, kw_only=True)
class Answer:
    """One completed turn, with everything needed to display it.

    Attributes:
        text: The answer as written, citation markers included. Rendering
            decides what to do with the markers.
        message: The stored assistant message.
        citations: Citations parsed out of the answer. Includes unverified
            ones, which the interface presents differently, because counting
            them is the only way to know whether the prompt is holding up.
        sources: Every excerpt retrieved, whether or not the answer cited it.
            Shown as a sources panel, so an answer with no usable citations
            still shows what it was working from.
    """

    text: str
    message: Message
    citations: list[Citation] = field(default_factory=list)
    sources: list[Chunk] = field(default_factory=list)


def parse_citations(
    text: str, sources: Sequence[Chunk], blocks: Sequence[Block] = ()
) -> list[Citation]:
    """Pull citation markers out of an answer and check the quotations.

    A quotation is looked for in every excerpt rather than only the one the
    marker names. Models misnumber references while quoting accurately, and
    throwing away a good quote over a bookkeeping slip helps nobody. What is
    never relaxed is that the text has to exist in something the model was
    actually shown.

    A marker with no quotation is kept as an unverified citation. It still says
    which excerpt a claim came from, which is worth recording even though the
    interface will not present it as a confirmed quote.

    Args:
        text: The answer as written.
        sources: The excerpts given to the model, in the order they were
            numbered.
        blocks: Blocks covering the page ranges of any excerpt that spans a
            page break, so a quotation is attributed to the page it is actually
            printed on rather than to the excerpt's first page.

    Returns:
        One citation per marker, in the order they appear. message_id is left
        unset, since the message does not exist yet.
    """
    citations: list[Citation] = []
    for match in _CITATION.finditer(text):
        index = int(match.group(1)) - 1
        quote = (match.group(2) or "").strip()

        cited = sources[index] if 0 <= index < len(sources) else None
        source = find_source(quote, sources) if quote else None

        # Prefer wherever the quote actually is. Fall back to the numbered
        # excerpt when there is no quote to locate.
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
    """Fetch the blocks needed to pin quotations to a page.

    Only for excerpts that cross a page break. A chunk sitting on one page
    already knows its page, so reading its blocks would be work for nothing.

    Args:
        conn: An open connection.
        chunks: The excerpts that were given to the model.

    Returns:
        Blocks covering the page ranges of the chunks that span pages.
    """
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
    """Return the excerpts that the pre-extracted facts came from.

    Ingest already did the hard version of this search. Finding Heineken's
    headcount took four differently worded queries, a round robin merge and a
    verified quotation, and the result is sitting in extracted_facts pointing
    at the exact chunk. Retrieval on its own does much worse: asked "Heineken
    employee count", the chunk holding 87,870 ranks 29th, because the report's
    methodology sections talk about "employee headcount" far more than the page
    that actually states the number does.

    So the answer is not to keep tuning retrieval. It is to reuse the work.
    These chunks are known good evidence for the questions people actually ask
    about a report, and adding them costs nothing at query time.

    Facts come back grouped by field, so the headcount precedes the
    sustainability goals and survives the cap.

    Args:
        conn: An open connection.
        document_id: The report in question.
        limit: How many distinct chunks to return at most.

    Returns:
        The chunks backing this document's facts, deduplicated.
    """
    facts = FactRepository(conn).read_for_document(document_id)
    chunk_ids = list(
        dict.fromkeys(fact.chunk_id for fact in facts if fact.chunk_id is not None)
    )
    return ChunkRepository(conn).read_by_ids(chunk_ids[:limit])


def retrieval_query(
    question: str, history: Sequence[Message], *, scoped: bool = False
) -> str:
    """Work out what to actually search for.

    A follow-up like "what about 2024?" is useless as a search on its own, so
    the previous question is folded in to give it a subject.

    Not when the question already names a company, though, and that exception
    is the whole reason this takes a `scoped` argument. Folding turned "and
    Heineken's employee count?" into a query still dominated by the previous
    question about Shell and climate spending, and retrieval came back with one
    Heineken excerpt, about climate risk. The follow-up handling made the
    answer worse than no handling at all. A question that names its own subject
    does not need the previous one.

    Args:
        question: What was just asked.
        history: The conversation so far, oldest first.
        scoped: Whether the question already names a company.

    Returns:
        The text to search with.
    """
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
    """Retrieve context and build the prompt for a question.

    Args:
        conn: An open connection.
        question: What was asked.
        conversation_id: The thread this belongs to.
        embeddings: The embedding provider.
        sources: How many excerpts to retrieve.
        document_id: Restrict to one report. When None, the question is read
            for a company name and narrowed automatically if it names exactly
            one.

    Returns:
        The prompt, the excerpts it was built from, and the conversation
        history that preceded it.
    """
    history = MessageRepository(conn).read_for_conversation(conversation_id)

    # A question naming one company is confined to that report. Without this,
    # "how much did Shell spend on climate adaptation" retrieves ASML excerpts
    # about climate adaptation, which are an excellent match for the words and
    # no use at all for the question.
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
    """Store a completed turn and its citations.

    The question and the answer are written in one transaction, so an
    interruption cannot leave a thread holding a question with no reply or a
    reply with no question.

    Args:
        conn: An open connection.
        conversation_id: The thread.
        question: What was asked.
        text: The answer as written.
        sources: The excerpts it was given.

    Returns:
        The completed turn.
    """
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
        assert stored.id is not None  # just written, same transaction
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
    """Answer a question and store the turn.

    The non-streaming path, for tests and for anything that wants the finished
    result rather than a live feed.

    Args:
        conn: An open connection.
        question: What was asked.
        conversation_id: The thread this belongs to.
        model: The language model.
        embeddings: The embedding provider.
        sources: How many excerpts to retrieve.
        document_id: Restrict to one report, or None for all of them.

    Returns:
        The completed turn.
    """
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
    """Answer a question, yielding text as it is written.

    The turn is stored once the last fragment has arrived, so an abandoned
    request leaves nothing behind. That is deliberate: a half written answer in
    the history is worse than no answer, because it looks like a complete one.

    Args:
        conn: An open connection.
        question: What was asked.
        conversation_id: The thread this belongs to.
        model: The language model.
        embeddings: The embedding provider.
        sources: How many excerpts to retrieve.
        document_id: Restrict to one report, or None for all of them.

    Yields:
        Fragments of the answer, in order.
    """
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
    """Name a thread from its question, without asking a model.

    Used when there is no model or the call fails. A few words of the actual
    question is always more useful than "Untitled", and it costs nothing.

    Args:
        question: The first question in the thread.

    Returns:
        A short title.
    """
    words = question.strip().split()
    if not words:
        return "New conversation"
    short = " ".join(words[:TITLE_FALLBACK_WORDS]).rstrip("?.,:;")
    return short + ("…" if len(words) > TITLE_FALLBACK_WORDS else "")


def start_conversation(
    conn: sqlite3.Connection, question: str, model: TextGenerator | None = None
) -> Conversation:
    """Open a thread, titled from its first question.

    A model failure here falls back to the question's own opening words rather
    than to nothing. Naming is a convenience and must never cost a question,
    but "Untitled" in a sidebar of twenty threads is useless, so the fallback
    is a real title rather than an absence.

    Args:
        conn: An open connection.
        question: The first question, used to name the thread.
        model: The language model. Without one the fallback is used.

    Returns:
        The new conversation.
    """
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
