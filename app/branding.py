"""Finding a company's logo.

Presentation only. Nothing here affects retrieval or extraction, and a missing
logo is an ordinary outcome: the interface simply falls back to the company
name, which it shows anyway.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from app.retrieve.scope import aliases

LOGO_DIR = Path(__file__).resolve().parent / "static" / "images"

_NOT_SLUG = re.compile(r"[^a-z0-9]+")


def slug(name: str) -> str:
    """Reduce a company name to a filename-safe form.

    Args:
        name: A company name or alias.

    Returns:
        Lowercase, with runs of anything else collapsed to a single hyphen.
        "ABN AMRO" becomes "abn-amro", "Heineken N.V." becomes "heineken-n-v".
    """
    return _NOT_SLUG.sub("-", name.casefold()).strip("-")


@lru_cache(maxsize=64)
def logo_for(company: str) -> str | None:
    """Find the logo file for a company, if there is one.

    Reuses the alias logic from retrieve.scope, which already knows that
    "Heineken N.V." and "Heineken" are the same company. Shorter aliases are
    tried first, since a logo file is named after the brand rather than the
    legal entity.

    Args:
        company: The company as stored on the document.

    Returns:
        A URL under /static, or None when no matching file exists.
    """
    for alias in sorted(aliases(company), key=len):
        name = slug(alias)
        if name and (LOGO_DIR / f"{name}.svg").is_file():
            return f"/static/images/{name}.svg"
    return None
