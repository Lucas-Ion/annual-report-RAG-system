"""Register annual reports and parse them into blocks.

    uv run python -m scripts.ingest data/pdfs

Takes directories or individual files. Prefer a directory: PowerShell does not
expand *.pdf into a list of arguments the way a Unix shell does, so a glob that
works on a Mac silently arrives as the literal string "*.pdf" on Windows.

Run it with -m rather than by path. Invoking the file directly puts scripts/ on
the import path instead of the repository root, and `import app` then fails.

This is the long job. Parsing runs at a few seconds per page, so five 400 page
reports take hours on a laptop and a good deal less on a machine with a real
GPU. Everything downstream (chunking, embedding, extraction) reads the blocks
this produces and runs in minutes, which is why this stage is the one worth
doing once on the fastest machine available and then committing.

Safe to interrupt. Progress is committed after every batch of pages, so
rerunning the same command picks up where the last run stopped rather than
starting the document again.

Company and year are read from the filename, so name the files with both in
them, for example:

    Shell_2025.pdf
    ABN_AMRO_Annual_Report_2025.pdf

Pass --company and --year to override, which only makes sense for a single
file. Use --dry-run first to check what was understood before committing hours
to it.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
import time
from collections.abc import Callable
from pathlib import Path

from app.db.connection import init_db, transaction
from app.db.models import Document, Stage
from app.db.repositories import BlockRepository, DocumentRepository, StageRunRepository
from app.ingest.parse import (
    DEFAULT_BATCH_SIZE,
    build_converter,
    page_count,
    parse_document,
)

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


def format_duration(seconds: float) -> str:
    """Render a duration as h/m/s, for progress lines nobody wants to decode.

    Args:
        seconds: Elapsed or remaining time.

    Returns:
        Something like "2h 14m", "3m 20s", or "45s".
    """
    seconds = max(0, int(seconds))
    if seconds >= 3600:
        return f"{seconds // 3600}h {seconds % 3600 // 60:02d}m"
    if seconds >= 60:
        return f"{seconds // 60}m {seconds % 60:02d}s"
    return f"{seconds}s"


# Shell


def collect_pdfs(paths: list[Path]) -> list[Path]:
    """Expand command line arguments into a sorted list of PDF files.

    A directory contributes every PDF directly inside it. A file is taken as
    given, which keeps `--company` and `--year` usable for a single report.

    Sorted so that a run is reproducible: directory listing order is whatever
    the filesystem feels like, and it differs between macOS and Windows, which
    would quietly change the order documents are ingested in.

    Args:
        paths: Files and directories from the command line.

    Returns:
        Every PDF to ingest, in a stable order.

    Raises:
        SystemExit: If a path does not exist, or a directory holds no PDFs.
            Both are almost always a typo, and finding out now beats finding
            out after the layout model has loaded.
    """
    found: list[Path] = []
    for path in paths:
        if path.is_dir():
            inside = sorted(path.glob("*.pdf"))
            if not inside:
                sys.exit(f"no PDFs in {path}")
            found.extend(inside)
        elif path.is_file():
            found.append(path)
        else:
            sys.exit(f"no such file or directory: {path}")
    return found


def file_hash(pdf: Path) -> str:
    """Fingerprint a file by its contents.

    Read in chunks rather than all at once. These are 15MB each today, which
    would be fine to slurp, but nothing about this function should care how big
    a report gets.

    Args:
        pdf: Path to the file.

    Returns:
        Hex sha256 of the bytes.
    """
    digest = hashlib.sha256()
    with pdf.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    """Parse every PDF given on the command line.

    Returns:
        0 if every document finished, 1 if any of them failed. A failure in one
        report does not abandon the rest: losing two finished hours because the
        fourth file is corrupt would be a poor trade.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths", nargs="+", type=Path, help="PDF files, or directories of them"
    )
    parser.add_argument("--company", help="override the name read from the filename")
    parser.add_argument(
        "--year", type=int, help="override the year read from the filename"
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="show what would be ingested and stop",
    )
    args = parser.parse_args()

    pdfs = collect_pdfs(args.paths)
    if (args.company or args.year) and len(pdfs) > 1:
        sys.exit("--company and --year apply to one report, so pass one file")

    # Work out what everything is before touching the database, so a
    # misunderstood filename is caught in seconds rather than after an hour.
    plan = []
    for pdf in pdfs:
        company = args.company or infer_company(pdf.stem)
        year = args.year or infer_year(pdf.stem)
        if not company or year is None:
            sys.exit(
                f"could not read a company and year from {pdf.name}.\n"
                f"got company={company!r} year={year!r}. "
                f"Rename the file or pass --company and --year."
            )
        plan.append((pdf, company, year, page_count(pdf)))

    total_pages = sum(pages for _, _, _, pages in plan)
    print(f"{'document':<40}{'company':<16}{'year':>6}{'pages':>8}")
    print("-" * 70)
    for pdf, company, year, pages in plan:
        print(f"{pdf.name[:39]:<40}{company[:15]:<16}{year:>6}{pages:>8}")
    print("-" * 70)
    print(f"{'':<62}{total_pages:>8} pages total\n")

    if args.dry_run:
        print("dry run, nothing written")
        return 0

    conn = init_db()
    documents = DocumentRepository(conn)
    blocks = BlockRepository(conn)
    stages = StageRunRepository(conn)

    # One converter for the whole run. Building it loads the layout model,
    # which costs around 21 seconds, and doing that per document would waste
    # most of two minutes on a five report run.
    print("loading the layout model, this takes a moment ...")
    converter = build_converter()

    started = time.perf_counter()
    pages_done = 0
    failures = 0

    def batch_reporter(pages_before: int) -> Callable[[int, int, int], None]:
        """Build the per batch progress callback for one document.

        A factory rather than a closure written inside the loop, because a
        closure would capture the running page total by reference. That happens
        to work today, since the callback is only called before the total moves
        on, but it is the kind of thing that breaks silently the moment the
        loop is rearranged.

        Args:
            pages_before: Pages already accounted for by earlier documents.

        Returns:
            A callback matching parse_document's on_batch.
        """

        def report(first: int, last: int, count: int) -> None:
            elapsed = time.perf_counter() - started
            done = pages_before + last
            rate = elapsed / done if done else 0.0
            remaining = (total_pages - done) * rate
            print(
                f"  pages {first:>4}-{last:<4} {count:>4} blocks   "
                f"{done}/{total_pages} pages   "
                f"{rate:.1f}s/page   about {format_duration(remaining)} left"
            )

        return report

    for pdf, company, year, pages in plan:
        digest = file_hash(pdf)
        document = documents.read_by_hash(digest)
        if document is None:
            with transaction(conn):
                document = documents.create(
                    Document(
                        filename=pdf.name,
                        file_hash=digest,
                        company=company,
                        year=year,
                        page_count=pages,
                    )
                )
            print(f"\n{pdf.name}: registered as document {document.id}")
        else:
            print(f"\n{pdf.name}: already registered as document {document.id}")

        assert document.id is not None
        if stages.is_done(document.id, Stage.PARSE):
            print("  already parsed, skipping")
            pages_done += pages
            continue

        resumed = blocks.last_parsed_page(document.id)
        if resumed:
            print(f"  resuming after page {resumed} of {pages}")

        # Each of these gets its own transaction. Repositories never commit on
        # their own, and without one here the final finish() of the run is
        # followed by nothing but conn.close(), which throws it away. The
        # document would then look unparsed forever and be redone on the next
        # run, having quietly done all the work.
        with transaction(conn):
            stages.start(document.id, Stage.PARSE)
        try:
            written = parse_document(
                conn,
                document,
                pdf,
                converter=converter,
                batch_size=args.batch_size,
                on_batch=batch_reporter(pages_done),
            )
        except Exception as exc:  # one bad report must not abandon the others
            with transaction(conn):
                stages.fail(document.id, Stage.PARSE, repr(exc))
            print(f"  FAILED: {exc!r}")
            failures += 1
        else:
            with transaction(conn):
                stages.finish(document.id, Stage.PARSE)
            stored = len(blocks.read_for_document(document.id))
            print(f"  done, {written} blocks this run, {stored} stored in total")
        pages_done += pages

    conn.close()
    print(f"\nfinished in {format_duration(time.perf_counter() - started)}")
    if failures:
        print(f"{failures} document(s) failed, rerun to retry them")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
