"""Reading a company and a year out of a report's filename.

Used by the ingest CLI and by the upload endpoint, which is why it lives here
rather than in scripts/. Both need the same answer, and two implementations of
"what company is this" would disagree within a week.

Filename inference is a convenience for seeding a folder of reports in bulk.
The upload form asks for the company and the year directly and only falls back
to guessing, because a form field is always going to beat a regex.
"""

from __future__ import annotations

import re

# Filler that appears in report filenames and tells you nothing about who
# published them. Stripped before what remains is treated as a company name.
_NOISE = re.compile(
    r"\b(annual|integrated|financial|statements?|report|jaarverslag|final|en|nl)\b",
    re.IGNORECASE,
)
_YEAR = re.compile(r"(?:19|20)\d{2}")

# Casing cannot be recovered by rule. "abn-amro" could reasonably be Abn Amro
# or ABN AMRO, and only one of those is the bank's name. Known issuers are
# corrected here and everything else falls back to the heuristic.
#
# This table exists because seeding in bulk from a folder is a convenience for
# getting five reports in. The real answer is the upload form, which asks for
# the company and the year rather than guessing them from a filename.
CANONICAL = {
    "abn amro": "ABN AMRO",
    "asml": "ASML",
    "cm": "CM",
    "heineken": "Heineken N.V.",
    "heineken nv": "Heineken N.V.",
    # Kept so the distinction is on the record. Heineken Holding N.V. is a
    # separate filer with no operations and, in its own words, no employees, so
    # its report is the wrong one for this system.
    "heineken holding nv": "Heineken Holding N.V.",
    "shell": "Shell",
}


# Core


def infer_year(stem: str) -> int | None:
    """Pull a reporting year out of a filename.

    Args:
        stem: Filename without its extension.

    Returns:
        The last four digit year in the name, or None if there is not one.
        The last rather than the first, because a name like
        "2024_2025_annual_report" is a report for the later year.
    """
    found = _YEAR.findall(stem)
    return int(found[-1]) if found else None


def infer_company(stem: str) -> str:
    """Pull a company name out of a filename.

    Words already in uppercase are left alone, so a filename that spells ASML
    correctly keeps it. Downloaded reports are usually lowercased throughout,
    though, so the result is then checked against CANONICAL to recover the
    casing no rule can infer.

    Args:
        stem: Filename without its extension.

    Returns:
        A best guess at the company name. Always check it with --dry-run before
        a long run, since this is pattern matching on filenames and no more.
    """
    words = [w for w in re.split(r"[\s_\-]+", _YEAR.sub(" ", stem)) if w]
    words = [w for w in words if not _NOISE.fullmatch(w)]
    guess = " ".join(w if w.isupper() else w.capitalize() for w in words)
    return CANONICAL.get(guess.casefold(), guess)
