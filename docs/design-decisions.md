# Design decisions

Measurements that shaped the system, kept because the reasoning is not
recoverable from the code once the scripts that produced it are gone.

## PDF parsing: Docling, not PyMuPDF

Compared on page 295 of the ABN AMRO report, the consolidated income statement.
The only question that mattered: does a number keep its row label and its
column year? A bare `2,347` with no indication of which line item and which
year it belongs to cannot be retrieved no matter how good retrieval is.

| Parser | Result |
|---|---|
| PyMuPDF, plain text | Numbers present in reading order, no column association. The Note column value `4` is indistinguishable from a figure |
| PyMuPDF, table detection | Catastrophic. Produced `\|Col1\|14,205<br>253<br>8,019<br>103\|Col3\|`, destroying every row label |
| **Docling** | **Correct.** `\| (in millions) \| Note \| 2025 \| 2024 \|` with every row aligned |

Docling costs roughly 5.5 seconds per page against PyMuPDF's milliseconds. The
whole ingest architecture, persisted stage artifacts and resumable batches,
exists to make that price payable once.

Settings: `do_ocr=False`, `do_table_structure=True`. OCR was measured and made
no difference to the extracted tables while costing 13% more time, because
these PDFs already carry a real text layer.

## Parsing throughput

| Machine | Seconds per page | 1,687 pages |
|---|---|---|
| MacBook Pro, MPS | 5.5 | about 2h 35m |
| RTX 5090, CUDA | 0.3 | 52 seconds |

The gap is why blocks are stored as their own stage. Chunking re-runs against
stored blocks in 2.3 seconds, so a chunking rule can be changed and tested
immediately rather than by reparsing.

## Table padding

Docling pads markdown table cells so columns align when read as plain text.
Across the five reports, **56% of all table text was padding**: 4.16 MB down to
1.84 MB once collapsed. One 51,603 character table held 128 characters of
header spread across 12,900. Tables exceeding the chunk ceiling fell from 204
to 30.

## Where the headcount figure hides

Every report leads with a rounded figure and buries the exact one.

| | Rounded, easy to find | Actual |
|---|---|---|
| ABN AMRO | "More than 20,000 employees" p9 | 23,126 FTE p47 |
| ASML | "> 44,000 Total employees" p4 | 44,209 FTE p301 |
| Shell | — | 85,000 headcount p116 |
| Heineken | "more than 80,000" p31 | 87,870 FTE p95 |
| CM | — | 648 FTE p55 |

Shell never uses the word FTE, not once in 462 pages. Defining the field by
those three letters finds nothing for one report in five. These findings are
written into the field instructions in `app/ingest/fields.py`.
