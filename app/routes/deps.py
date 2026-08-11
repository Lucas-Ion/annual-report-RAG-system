"""Shared resources for request handlers."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from functools import lru_cache
from typing import Annotated

from fastapi import Depends, Request

from app.db.connection import connect
from app.providers import BGEEmbeddings, ClaudeProvider


@lru_cache(maxsize=1)
def embeddings() -> BGEEmbeddings:
    return BGEEmbeddings()


@lru_cache(maxsize=1)
def language_model() -> ClaudeProvider:
    return ClaudeProvider()


def connection() -> Iterator[sqlite3.Connection]:
    conn = connect()
    try:
        yield conn
    finally:
        conn.close()


def templates(request: Request):
    return request.app.state.templates


Connection = Annotated[sqlite3.Connection, Depends(connection)]
Embeddings = Annotated[BGEEmbeddings, Depends(embeddings)]
Model = Annotated[ClaudeProvider, Depends(language_model)]
