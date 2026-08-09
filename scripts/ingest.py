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
import sys
import time
from pathlib import Path

from app.config import load_environment
from app.db.connection import init_db
from app.db.models import Stage
from app.ingest.naming import infer_company, infer_year
from app.ingest.parse import DEFAULT_BATCH_SIZE, build_converter, page_count
from app.ingest.pipeline import DEFAULT_STAGES, ingest
from app.providers import BGEEmbeddings, ClaudeProvider


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


# Shell


def main() -> int:
    """Ingest every PDF given on the command line.

    Returns:
        0 if every document finished, 1 if any of them failed. A failure in
        one report does not abandon the rest: losing two finished hours
        because the fourth file is corrupt would be a poor trade.
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
        "--stages",
        nargs="+",
        choices=[stage.value for stage in DEFAULT_STAGES],
        default=[stage.value for stage in DEFAULT_STAGES],
        help="which stages to run, default all of them",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="show what would be ingested and stop"
    )
    args = parser.parse_args()

    load_environment()
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
    print(f"{'':<62}{total_pages:>8} pages total")
    print(f"\nstages: {', '.join(args.stages)}\n")

    if args.dry_run:
        print("dry run, nothing written")
        return 0

    stages = [Stage(name) for name in args.stages]
    conn = init_db()

    # Both of these load large models, so they are built once for the whole
    # run and only if a stage actually needs them. Constructing the embedding
    # provider is free; it loads on first use.
    converter = build_converter() if Stage.PARSE in stages else None
    embeddings = BGEEmbeddings() if {Stage.EMBED, Stage.EXTRACT} & set(stages) else None
    model = ClaudeProvider() if Stage.EXTRACT in stages else None

    started = time.perf_counter()
    failures = 0
    for pdf, company, year, _pages in plan:
        print(f"\n{pdf.name}")
        try:
            ingest(
                conn,
                pdf,
                company=company,
                year=year,
                converter=converter,
                embeddings=embeddings,
                model=model,
                stages=stages,
                parse_batch_size=args.batch_size,
                on_progress=lambda stage, message: print(
                    f"  {stage.value:<8}{message}"
                ),
            )
        except Exception as exc:  # one bad report must not abandon the others
            print(f"  FAILED: {exc!r}")
            failures += 1

    conn.close()
    print(f"\nfinished in {format_duration(time.perf_counter() - started)}")
    if failures:
        print(f"{failures} document(s) failed, rerun to retry them")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
