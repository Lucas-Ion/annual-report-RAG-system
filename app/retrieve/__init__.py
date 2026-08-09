"""Turning a question into the handful of chunks most likely to answer it."""

from app.retrieve.hybrid import RRF_K, fuse, interleave, search, search_many
from app.retrieve.scope import aliases, detect_document

__all__ = [
    "RRF_K",
    "aliases",
    "detect_document",
    "fuse",
    "interleave",
    "search",
    "search_many",
]
