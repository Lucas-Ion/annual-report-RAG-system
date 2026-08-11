"""Domain objects for the RAG system."""

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
    id: int | None = None
    filename: str
    file_hash: str
    company: str
    year: int
    page_count: int | None = None
    created_at: str | None = None


@dataclass(slots=True, kw_only=True)
class StageRun:
    document_id: int
    stage: Stage
    status: StageStatus
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None


@dataclass(slots=True, kw_only=True)
class Block:
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
        return f"{self.context_header}\n\n{self.text}"


@dataclass(slots=True, kw_only=True)
class Fact:
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
    id: int | None = None
    title: str | None = None
    created_at: str | None = None


@dataclass(slots=True, kw_only=True)
class Message:
    id: int | None = None
    conversation_id: int
    role: Role
    content: str
    created_at: str | None = None


@dataclass(slots=True, kw_only=True)
class Citation:
    id: int | None = None
    message_id: int
    chunk_id: int
    quote: str
    page_no: int
    verified: bool = False


@dataclass(slots=True, kw_only=True)
class SearchHit:
    chunk: Chunk
    score: float
