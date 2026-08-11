"""Reading a company and a year out of a report's filename."""

from __future__ import annotations

import re

_NOISE = re.compile(
    r"\b(annual|integrated|financial|statements?|report|jaarverslag|final|en|nl)\b",
    re.IGNORECASE,
)
_YEAR = re.compile(r"(?:19|20)\d{2}")

CANONICAL = {
    "abn amro": "ABN AMRO",
    "asml": "ASML",
    "cm": "CM",
    "heineken": "Heineken N.V.",
    "heineken nv": "Heineken N.V.",
    "heineken holding nv": "Heineken Holding N.V.",
    "shell": "Shell",
}


def infer_year(stem: str) -> int | None:
    found = _YEAR.findall(stem)
    return int(found[-1]) if found else None


def infer_company(stem: str) -> str:
    words = [w for w in re.split(r"[\s_\-]+", _YEAR.sub(" ", stem)) if w]
    words = [w for w in words if not _NOISE.fullmatch(w)]
    guess = " ".join(w if w.isupper() else w.capitalize() for w in words)
    return CANONICAL.get(guess.casefold(), guess)
