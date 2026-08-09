"""Shared resources for request handlers.

Two very different lifetimes here, and getting them the wrong way round is the
usual way a SQLite web application goes wrong.

The providers are built once for the process. The embedding model is 2GB and
takes the better part of a minute to load, so loading it per request would make
every question unusable.

The connection is built per request and closed with it. A sqlite3 connection
belongs to the thread that opened it, and FastAPI runs synchronous handlers in
a threadpool, so a shared connection would eventually be used from the wrong
thread and fail. Opening one costs microseconds.
"""

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
    """The embedding provider, built once per process.

    Constructing it is free; the model loads on first use, which means startup
    stays fast and the cost lands on whoever asks the first question.

    Returns:
        The shared provider.
    """
    return BGEEmbeddings()


@lru_cache(maxsize=1)
def language_model() -> ClaudeProvider:
    """The language model, built once per process.

    Does not check for an API key. A missing key raises MissingApiKey at the
    point of use, so the document browser keeps working for someone who has not
    set one up.

    Returns:
        The shared provider.
    """
    return ClaudeProvider()


def connection() -> Iterator[sqlite3.Connection]:
    """Open a connection for one request and close it afterwards.

    Yields:
        A configured connection.
    """
    conn = connect()
    try:
        yield conn
    finally:
        conn.close()


def templates(request: Request):
    """Reach the Jinja environment set up by the application factory.

    Args:
        request: The incoming request.

    Returns:
        The shared Jinja2Templates instance.
    """
    return request.app.state.templates


Connection = Annotated[sqlite3.Connection, Depends(connection)]
Embeddings = Annotated[BGEEmbeddings, Depends(embeddings)]
Model = Annotated[ClaudeProvider, Depends(language_model)]
