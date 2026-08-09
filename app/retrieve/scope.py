"""Working out which report a question is about.

Retrieval searches every indexed report at once, which is right for "which
company has the most ambitious climate target" and wrong for almost everything
else. Asked how much Shell spent on climate adaptation, the index cheerfully
returned an ASML excerpt about climate adaptation activities, because to a
search engine that is an excellent match.

Naming a company is the strongest signal a question carries, and it is free to
read. When exactly one indexed company is named, retrieval is confined to that
report. When several are named the question is a comparison and the whole index
is searched, which is equally deliberate.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from app.db.models import Document

# Stripped from a company name to produce a shorter alias. "Heineken N.V."
# should be found by a question that just says Heineken, and nobody types the
# suffix.
_SUFFIXES = frozenset(
    {"nv", "n.v.", "bv", "b.v.", "plc", "ltd", "limited", "inc", "sa", "ag", "group"}
)

# Aliases shorter than this are only matched with exact capitalisation. "CM" is
# a real company here and also two letters that turn up inside other words and
# as a unit of length, so it needs the stricter test.
_SHORT_ALIAS = 3


def aliases(company: str) -> set[str]:
    """Build the names a question might use for a company.

    Args:
        company: The company as stored on the document.

    Returns:
        Lowercased forms to look for, including the full name and the name with
        corporate suffixes removed.
    """
    found = {company.strip()}
    words = company.split()

    # Every dot removed, not just the ones at the ends. "N.V." has one in the
    # middle, so stripping only the outer dots leaves "n.v" and never matches
    # the suffix list.
    trimmed = [
        word for word in words if word.replace(".", "").casefold() not in _SUFFIXES
    ]
    if trimmed and len(trimmed) != len(words):
        found.add(" ".join(trimmed))
    return {name.casefold() for name in found if name}


def _mentions(question: str, alias: str, original: str) -> bool:
    """Test whether a question names a company.

    Args:
        question: The question as typed.
        alias: A lowercased alias.
        original: The company name as stored, for the case sensitive test.

    Returns:
        True if the alias appears as a whole word.
    """
    pattern = re.compile(rf"(?<!\w){re.escape(alias)}(?!\w)", re.IGNORECASE)
    if len(alias) >= _SHORT_ALIAS:
        return pattern.search(question) is not None

    # Short aliases only count when the capitalisation matches too, so "CM"
    # matches and "cm" in "50cm" or a stray lowercase word does not.
    exact = re.compile(rf"(?<!\w){re.escape(original.strip())}(?!\w)")
    return exact.search(question) is not None


def detect_document(question: str, documents: Sequence[Document]) -> Document | None:
    """Find the single report a question is about, if it names one.

    Args:
        question: The question as typed.
        documents: The indexed reports.

    Returns:
        The document, or None when the question names no company or names
        more than one. Both of those mean the whole index should be searched:
        the first because there is nothing to narrow to, the second because a
        comparison needs both sides.
    """
    matched = [
        document
        for document in documents
        if any(
            _mentions(question, alias, document.company)
            for alias in aliases(document.company)
        )
    ]
    return matched[0] if len(matched) == 1 else None
