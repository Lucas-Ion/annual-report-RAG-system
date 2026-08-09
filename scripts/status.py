"""Report what is in the database and how far each report has been processed.

    uv run python -m scripts.status

Read only, so it is safe to run at any time, including while an ingest is
still going. Written as a script rather than a shell one liner because those
stop being portable the moment Windows is involved, and this gets run on both
machines.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from app.db.connection import connect, database_path
from app.db.models import Stage

_STAGES = (Stage.PARSE, Stage.CHUNK, Stage.EMBED, Stage.EXTRACT)

# Shown instead of a status when a stage has never been attempted, which is
# different from having been attempted and failed.
_NEVER_RUN = "."


@dataclass(slots=True, kw_only=True)
class DocumentStatus:
    """Everything the status table shows for one report.

    Attributes:
        company: Issuing company.
        year: Reporting year.
        pages: Pages in the source PDF, or None before parsing recorded it.
        blocks: Rows in blocks.
        tables: Blocks that are tables. Zero here on a parsed document means
            the financial statements did not survive, which is worth catching
            before anything is built on top of it.
        chunks: Rows in chunks.
        vectors: Chunks that have an embedding.
        facts: Rows in extracted_facts.
        stages: Status of each of the four stages, in pipeline order.
    """

    company: str
    year: int
    pages: int | None
    blocks: int
    tables: int
    chunks: int
    vectors: int
    facts: int
    stages: dict[Stage, str]


# Core


def format_table(rows: list[DocumentStatus]) -> str:
    """Render the status of every document as a table.

    Args:
        rows: One entry per document, in display order.

    Returns:
        The table as a single string, totals included.
    """
    if not rows:
        return "no documents ingested yet"

    header = (
        f"{'company':<18}{'year':>5}{'pages':>7}{'blocks':>8}"
        f"{'tables':>8}{'chunks':>8}{'vectors':>8}{'facts':>7}"
    )
    lines = [header, "-" * len(header)]
    for row in rows:
        lines.append(
            f"{row.company[:17]:<18}{row.year:>5}{row.pages or 0:>7}{row.blocks:>8}"
            f"{row.tables:>8}{row.chunks:>8}{row.vectors:>8}{row.facts:>7}"
        )
    lines.append("-" * len(header))
    lines.append(
        f"{'':<18}{'':>5}{sum(r.pages or 0 for r in rows):>7}"
        f"{sum(r.blocks for r in rows):>8}{sum(r.tables for r in rows):>8}"
        f"{sum(r.chunks for r in rows):>8}{sum(r.vectors for r in rows):>8}"
        f"{sum(r.facts for r in rows):>7}"
    )
    return "\n".join(lines)


def format_stages(rows: list[DocumentStatus]) -> str:
    """Render per document stage progress.

    Args:
        rows: One entry per document, in display order.

    Returns:
        One line per document showing where each stage stands.
    """
    if not rows:
        return ""

    lines = [f"{'company':<18}" + "".join(f"{s.value:>10}" for s in _STAGES)]
    lines.append("-" * len(lines[0]))
    for row in rows:
        statuses = "".join(f"{row.stages.get(s, _NEVER_RUN):>10}" for s in _STAGES)
        lines.append(f"{row.company[:17]:<18}{statuses}")
    return "\n".join(lines)


def warnings(rows: list[DocumentStatus]) -> list[str]:
    """Point out results that look wrong rather than merely incomplete.

    Args:
        rows: One entry per document.

    Returns:
        Human readable problems, empty when nothing stands out.
    """
    found = []
    for row in rows:
        if row.stages.get(Stage.PARSE) == "failed":
            found.append(f"{row.company}: parsing failed, rerun the ingest")
        elif row.blocks == 0:
            found.append(f"{row.company}: no blocks at all")
        elif row.tables == 0:
            found.append(
                f"{row.company}: parsed but found no tables, so its financial "
                f"statements are probably missing"
            )
        elif row.pages and row.blocks / row.pages < 3:
            found.append(
                f"{row.company}: only {row.blocks / row.pages:.1f} blocks per "
                f"page, which is unusually sparse"
            )
        if row.chunks and row.vectors < row.chunks:
            found.append(
                f"{row.company}: {row.chunks - row.vectors} chunks still have "
                f"no embedding"
            )
    return found


# Shell


def read_status(conn: sqlite3.Connection) -> list[DocumentStatus]:
    """Gather counts for every document.

    One query per document rather than a single wide join. There are five rows
    and this runs in milliseconds either way, so the readable version wins.

    Args:
        conn: An open connection.

    Returns:
        One entry per document, ordered by company.
    """
    rows = []
    documents = conn.execute(
        "SELECT id, company, year, page_count FROM documents ORDER BY company, year"
    ).fetchall()

    for document in documents:
        counts = conn.execute(
            """
            SELECT count(*) AS blocks,
                   coalesce(sum(label = 'table'), 0) AS tables
              FROM blocks WHERE document_id = ?
            """,
            (document["id"],),
        ).fetchone()
        chunks = conn.execute(
            "SELECT count(*) AS n FROM chunks WHERE document_id = ?",
            (document["id"],),
        ).fetchone()["n"]
        vectors = conn.execute(
            """
            SELECT count(*) AS n FROM chunk_vectors v
              JOIN chunks c ON c.id = v.chunk_id
             WHERE c.document_id = ?
            """,
            (document["id"],),
        ).fetchone()["n"]
        facts = conn.execute(
            "SELECT count(*) AS n FROM extracted_facts WHERE document_id = ?",
            (document["id"],),
        ).fetchone()["n"]
        stages = {
            Stage(row["stage"]): row["status"]
            for row in conn.execute(
                "SELECT stage, status FROM stage_runs WHERE document_id = ?",
                (document["id"],),
            ).fetchall()
        }
        rows.append(
            DocumentStatus(
                company=document["company"],
                year=document["year"],
                pages=document["page_count"],
                blocks=counts["blocks"],
                tables=counts["tables"],
                chunks=chunks,
                vectors=vectors,
                facts=facts,
                stages=stages,
            )
        )
    return rows


def main() -> int:
    """Print the status tables.

    Returns:
        0 always. This reports, it does not judge, and a half finished ingest
        is a normal thing to be looking at.
    """
    path = database_path()
    if not path.exists():
        print(f"no database at {path}")
        return 0

    conn = connect(path)
    rows = read_status(conn)
    conn.close()

    size = path.stat().st_size / 1_000_000
    print(f"{path}  ({size:.1f} MB)\n")
    print(format_table(rows))
    if rows:
        print()
        print(format_stages(rows))
    for warning in warnings(rows):
        print(f"\n  WARNING  {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
