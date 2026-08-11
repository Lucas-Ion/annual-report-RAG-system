"""Working out which report a question is about"""

from __future__ import annotations

import re
from collections.abc import Sequence

from app.db.models import Document

_SUFFIXES = frozenset(
    {"nv", "n.v.", "bv", "b.v.", "plc", "ltd", "limited", "inc", "sa", "ag", "group"}
)

_SHORT_ALIAS = 3


def aliases(company: str) -> set[str]:
    found = {company.strip()}
    words = company.split()
    trimmed = [
        word for word in words if word.replace(".", "").casefold() not in _SUFFIXES
    ]
    if trimmed and len(trimmed) != len(words):
        found.add(" ".join(trimmed))
    return {name.casefold() for name in found if name}


def _mentions(question: str, alias: str, original: str) -> bool:
    pattern = re.compile(rf"(?<!\w){re.escape(alias)}(?!\w)", re.IGNORECASE)
    if len(alias) >= _SHORT_ALIAS:
        return pattern.search(question) is not None

    exact = re.compile(rf"(?<!\w){re.escape(original.strip())}(?!\w)")
    return exact.search(question) is not None


def detect_document(question: str, documents: Sequence[Document]) -> Document | None:
    matched = [
        document
        for document in documents
        if any(
            _mentions(question, alias, document.company)
            for alias in aliases(document.company)
        )
    ]
    return matched[0] if len(matched) == 1 else None
