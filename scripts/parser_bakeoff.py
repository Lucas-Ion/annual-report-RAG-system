"""Compare PDF parsers on a single page of a real annual report.

    uv run python scripts/parser_bakeoff.py data/pdfs/ABN_AMRO_Annual_Report.pdf 295

Writes one file per parser into scratch/ so the outputs can be read side by side.

This answers exactly one question: did the numbers keep their row and column
labels? A bare "2,347" with no indication of which line item and which year it
belongs to is unretrievable no matter how good the retrieval stage is.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import fitz  # PyMuPDF


def slice_page(pdf: Path, page_no: int, dest: Path) -> Path:
    """Extract a single 1-based page into its own PDF.

    Docling converts whole documents and this report is 434 pages. Slicing first
    keeps each run to seconds and makes the comparison reproducible.
    """
    with fitz.open(pdf) as src:
        if not 1 <= page_no <= src.page_count:
            sys.exit(
                f"page {page_no} out of range (document has {src.page_count} pages)"
            )
        with fitz.open() as out:
            out.insert_pdf(src, from_page=page_no - 1, to_page=page_no - 1)
            out.save(dest)
    return dest


def pymupdf_plain(page_pdf: Path) -> str:
    """The naive baseline: raw text in reading order, no structure."""
    with fitz.open(page_pdf) as doc:
        # get_text() returns different shapes depending on the mode argument,
        # so it is typed as a union. "text" always gives a string.
        return str(doc[0].get_text("text"))


def pymupdf_tables(page_pdf: Path) -> str:
    """PyMuPDF's own table detection, rendered as markdown."""
    with fitz.open(page_pdf) as doc:
        found = doc[0].find_tables()
        if found is None or not found.tables:
            return "(no tables detected on this page)"
        return "\n\n".join(t.to_markdown() for t in found.tables)


def docling_markdown(page_pdf: Path) -> str:
    """Layout-aware conversion to markdown, tables included."""
    from docling.document_converter import DocumentConverter

    result = DocumentConverter().convert(str(page_pdf))
    return result.document.export_to_markdown()


PARSERS = {
    "pymupdf-plain": pymupdf_plain,
    "pymupdf-tables": pymupdf_tables,
    "docling": docling_markdown,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pdf", type=Path)
    ap.add_argument("page", type=int, help="1-based page number in the PDF")
    ap.add_argument("--out", type=Path, default=Path("scratch"))
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    page_pdf = slice_page(args.pdf, args.page, args.out / f"page_{args.page}.pdf")

    for name, parse in PARSERS.items():
        started = time.perf_counter()
        try:
            text = parse(page_pdf)
        except Exception as exc:  # a parser failing outright is a result too
            text = f"FAILED: {exc!r}"
        elapsed = time.perf_counter() - started

        dest = args.out / f"page_{args.page}.{name}.txt"
        dest.write_text(text, encoding="utf-8")
        print(f"{name:<16}{elapsed:7.2f}s{len(text):>8} chars  ->  {dest}")

    print(
        "\nRead the files side by side. For each number, ask whether it still "
        "carries\nits row label and its column (year) header."
    )


if __name__ == "__main__":
    main()
