"""Finding a company's logo."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from app.retrieve.scope import aliases

LOGO_DIR = Path(__file__).resolve().parent / "static" / "images"

_NOT_SLUG = re.compile(r"[^a-z0-9]+")


def slug(name: str) -> str:
    return _NOT_SLUG.sub("-", name.casefold()).strip("-")


@lru_cache(maxsize=64)
def logo_for(company: str) -> str | None:
    for alias in sorted(aliases(company), key=len):
        name = slug(alias)
        if name and (LOGO_DIR / f"{name}.svg").is_file():
            return f"/static/images/{name}.svg"
    return None
