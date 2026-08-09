"""Domain objects for the annual report RAG system.

These are what the repositories hand back. Nothing outside app/db ever sees a
sqlite3.Row, which is the entire point of the repository layer: the ingest
pipeline and the web routes work with Document and Chunk objects and stay
ignorant of the fact that any of this is stored in SQLite.

Every stored object carries an optional id. An object built in memory has none
yet, and the repository fills it in when the row is written.

The enums mirror the CHECK constraints in schema.sql. Keeping them here means
an invalid stage or role is caught by the type checker while you are writing
the code, rather than by SQLite at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Stage(StrEnum):
    PARSE = "parse"
    CHUNK = "chunk"
    EMBED = "embed"
    EXTRACT = "extract"


class StageStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class ChunkType(StrEnum):
    PROSE = "prose"
    TABLE = "table"


class Role(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(slots=True, kw_only=True)
class Document:
    """One ingested annual report.

    Attributes:
        id: Row id. None until the document has been created.
        filename: Original PDF filename
        file_hash: sha256 of the file bytes. Used to maintain idempotency, so a
            second upload of the same PDF is recognised and skipped rather
            than parsed again.
        company: Issuing company, used to scope a question to one report.
        year: Reporting year.
        page_count: Pages in the PDF. Unknown until the parse stage has run.
        created_at: Set by the database default.
    """

    id: int | None = None
    filename: str
    file_hash: str
    company: str
    year: int
    page_count: int | None = None
    created_at: str | None = None


@dataclass(slots=True, kw_only=True)
class StageRun:
    """Progress of one ingest stage for one document.

     Its identity is the pair (document_id, stage), because a document has
     exactly one run of each stage and a re-run overwrites the previous
     result rather than adding to it.

    Attributes:
        document_id: The document being processed.
        stage: Which of the four stages this row tracks.
        status: Where that stage has got to.
        started_at: Timestamp the stage was last started.
        finished_at: Timestamp it finished, successfully or not.
        error: Failure message, populated only when status is FAILED. Kept so
            a crashed ingest can be diagnosed after the fact instead of being
            reproduced.
    """

    document_id: int
    stage: Stage
    status: StageStatus
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None


@dataclass(slots=True, kw_only=True)
class Block:
    """One item as the parser found it, in reading order.

    A block is a heading, a paragraph, a list item, or a whole table. It is
    the raw expensive output of Docling, kept exactly as produced so that
    chunking can be rewritten and re-run without touching a PDF again.

    Attributes:
        id: Row id. None until created.
        document_id: Owning document.
        seq: Position in reading order within the document, starting at 0.
        page_no: Page this block appears on, counting from 1.
        label: Docling's structural classification, for example
            section_header, text, table, or list_item. Chunking splits on this
            rather than trying to recognise headings by pattern matching.
        level: Heading depth as the parser reported it, or None for anything
            that is not a heading. Measured on these reports, Docling calls
            every heading level 1, so do not build anything on the assumption
            that this expresses a hierarchy. See the note in schema.sql.
        text: Markdown for tables, plain text for everything else.
        bbox: Position on the page as (left, top, right, bottom), or None when
            the parser did not report one. Stored so a citation could one day
            be highlighted on a page image.
    """

    id: int | None = None
    document_id: int
    seq: int
    page_no: int
    label: str
    text: str
    level: int | None = None
    bbox: tuple[float, float, float, float] | None = None


@dataclass(slots=True, kw_only=True)
class Chunk:
    """A retrieval sized piece of a document, built from one or more blocks.

    Attributes:
        id: Row id. None until created.
        document_id: Owning document.
        seq: Position among this document's chunks.
        page_start: First page the chunk's content came from.
        page_end: Last page. Equal to page_start unless a table spans a break.
        chunk_type: Prose or table.
        context_header: Synthetic breadcrumb such as
            "ABN AMRO | Annual Report 2025 | Sustainability".
        text: The verbatim source text.
        section: Nearest preceding heading, or None above the first one.
        token_count: Size estimate.
    """

    id: int | None = None
    document_id: int
    seq: int
    page_start: int
    page_end: int
    chunk_type: ChunkType
    context_header: str
    text: str
    section: str | None = None
    token_count: int | None = None

    @property
    def embedding_text(self) -> str:
        """The chunk as the embedding model should see it.

        A chunk pulled from the middle of a 400 page report often reads as
        orphaned: "increased by 12% to 4,208" says nothing on its own. Gluing
        the breadcrumb on front gives the model the company, the year and the
        section, which is usually the difference between a chunk being
        findable and not.

        This lives here, as one property, so that no caller can accidentally
        embed the bare text or store the prefixed version. The header is for
        the embedding model only and must never reach the verbatim check.

        Returns:
            The context header and the source text, separated by a blank line.
        """
        return f"{self.context_header}\n\n{self.text}"


@dataclass(slots=True, kw_only=True)
class Fact:
    """A datapoint pulled out of a report at ingest time.

    These back the "pre extracted data visible in the application" part of the
    brief, and they are computed up front so the overview page never waits on
    a model call.

    Attributes:
        id: Row id. None until created.
        document_id: Owning document.
        field_key: Which field this is, matching an entry in the extraction
            field registry, for example "fte" or "sustainability_goal".
        verbatim_quote: The exact span this was read from. Must appear byte for
            byte in the referenced chunk's text, and is checked before the row
            is ever written.
        page_no: Page the quote sits on, so the answer can cite it.
        value_raw: The figure as printed, for example "20,417".
        value_numeric: The same figure parsed, for sorting and comparison.
        unit: FTE, EUR m, tCO2e, and so on.
        chunk_id: Chunk the quote came from. Nullable because rechunking may
            legitimately dissolve the chunk a fact was found in.
        confidence: The extractor's own reported confidence, when it gives one.
        extractor_version: Lets a later run supersede earlier rows instead of
            silently duplicating them.
        created_at: Set by the database default.
    """

    id: int | None = None
    document_id: int
    field_key: str
    verbatim_quote: str
    page_no: int
    extractor_version: str
    value_raw: str | None = None
    value_numeric: float | None = None
    unit: str | None = None
    chunk_id: int | None = None
    confidence: float | None = None
    created_at: str | None = None


@dataclass(slots=True, kw_only=True)
class Conversation:
    """One chat thread.

    Attributes:
        id: Row id. None until created.
        title: Short label for the thread, usually derived from the first
            question. None until one has been generated.
        created_at: Set by the database default.
    """

    id: int | None = None
    title: str | None = None
    created_at: str | None = None


@dataclass(slots=True, kw_only=True)
class Message:
    """One turn in a conversation.

    Attributes:
        id: Row id. None until created.
        conversation_id: Owning thread.
        role: Who said it.
        content: The text of the turn.
        created_at: Set by the database default.
    """

    id: int | None = None
    conversation_id: int
    role: Role
    content: str
    created_at: str | None = None


@dataclass(slots=True, kw_only=True)
class Citation:
    """A source attached to an assistant message.

    Attributes:
        id: Row id. None until created.
        message_id: The answer this supports.
        chunk_id: Where the quote was taken from.
        quote: The quoted span itself.
        page_no: Page to show the user.
        verified: Whether the quote was found byte for byte in the cited chunk.
            Unverified citations are dropped before display rather than shown
            with a caveat, because a citation nobody can trust is worse than no
            citation at all.
    """

    id: int | None = None
    message_id: int
    chunk_id: int
    quote: str
    page_no: int
    verified: bool = False


@dataclass(slots=True, kw_only=True)
class SearchHit:
    """One chunk returned by a search, with how well it scored.

    Attributes:
        chunk: The matched chunk, fully populated.
        score: Relevance, where higher is always better. The two searches use
            completely different underlying metrics, one of which is naturally
            backwards, so both are normalised to that rule before they leave
            the repository. Do not compare scores across the two searches;
            only the ordering within one result list is meaningful.
    """

    chunk: Chunk
    score: float
