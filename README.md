# Annual report RAG system

## Introduction

This is a retrieval augmented generation system over corporate annual reports.
It ingests PDF filings, parses them into structured blocks with a layout aware
parser so that a figure keeps its row label and its column year, groups those
blocks into retrievable chunks, indexes each chunk both lexically and as a dense
vector, and extracts named datapoints at ingest time. The application serves three pages: a list of the indexed reports
with everything extracted from each, a comparison view putting one field side by
side across all reports, and a streaming chat interface.
## How to set up

### Requirements

Python 3.13, provided by `uv`. Nothing else needs to be installed. 

If `uv` is not present:

```
pip install uv
```

### Install and run

```bash
uv sync
uv run uvicorn app.main:app --reload
```

The application is then at `http://127.0.0.1:8000`.

### The database ships populated

`data/rag.db` is committed, along with the source PDFs in `data/pdfs`. No
ingestion is required. The reports, the extracted datapoints, their quotations
and the page links into the PDFs all work on a fresh clone.

### The API key is optional

Only free form chat calls a model. Everything else runs locally with no account
and no network. To enable chat:

```bash
cp .env.example .env
```

Then set `ANTHROPIC_API_KEY` in `.env`. Without a key the application still
starts and every page except chat behaves normally. The chat will return a 503 with however with an explanation.

### Download the embedding model first

The first chat question loads `bge-m3`, which is roughly 2GB and is downloaded
on first use. In the browser depending on your browser and internet speed this can look like a hang, because the download
reports no progress to the page. I advise to pull it in advance so the download is visible
in the terminal:

```bash
uv run python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"
```

This is only needed once per machine.

### Adding a report

Upload a PDF on the reports page. Parsing runs at several seconds per page, so a
400 page report takes a while on a laptop. The work continues on the server if
the browser navigates away, and it is safe to interrupt, since progress is
committed after every batch of 25 pages.

### Run the test suite

```bash
uv run pytest
```

209 tests, about five seconds. No network, no API key and no model load, because the two model providers are substituted with fakes.

## Design decisions

### Docling for PDF parsing

I compared Docling against PyMuPDF on page 295 of the ABN AMRO report, the
consolidated income statement, on the criterion that in my opinion was the focus of this assignment being: does a
number keep its row label and its column year. PyMuPDF's plain text lost the
column association entirely, and its table detection destroyed every row label.
Docling produced a correct markdown table, so I accepted its cost of roughly 5.5
seconds per page against PyMuPDF's milliseconds. For me this s is a correctness decision
rather than a performance one, because a figure that has lost its row and its
year cannot be retrieved however good retrieval is so the trade off seemed worth it.

### A staged and resumable pipeline

Ingestion runs in four stages, parse, chunk, embed and extract, and each stage
persists its output. This exists because parsing is expensive: with blocks
stored, a chunking rule can be changed and re-run much faster. Progress is recorded per document and per stage, so an interrupted run
resumes from the last committed batch rather than starting again, and the web
progress display reads the same records rather than holding state in memory.

### Layout aware chunking

Chunks target 2,000 characters with a hard ceiling of 6,000. Tables are also never
mixed with prose: a table becomes its own chunk and is split by rows with the
header repeated on every piece, because a fragment carrying figures without its
column years is not useful. I measured that 56 percent of all table
text was cell alignment padding, and removing it took table text from 4.16MB to
1.84MB and dropped the number of tables exceeding the chunk ceiling from 204 to
30.

### bge-m3 as the embedding model

I chose bge-m3 for its context length. It accepts 8,192 tokens where most
English alternatives stop at 512, and my chunks reach roughly 1,500 tokens, so a
512 token model would truncate the largest quarter of the index. Text
past that cutoff would still be stored, displayed and citable, and completely
unsearchable. It is also multilingual, which helps parsing multilingual reports.

### Local embeddings rather than a hosted API

Embedding runs locally through sentence-transformers. This removes an API key
from the setup path.

### SQLite as the whole storage layer

One file does four jobs: relational storage, full text search through FTS5,
vector search through the sqlite-vec extension, and ingest job state. There is
no server to install and the populated database ships in the repository.

### Hybrid retrieval

Every question is answered by two searches, BM25 over the FTS5 index and nearest
neighbour over the vectors. They are diametrically opposed, which is the
reason to run both.

### Reciprocal rank fusion

The two result lists are merged by rank rather than by score. BM25 returns negative values whose scale depends on the documents statistics and the
vector search returns distances, so there is no normalisation which makes them
comparable, but their orderings are comparable.


### Scope detection by company name

If a question names exactly one indexed company, retrieval is confined to that
report.

### Verbatim verification as a hard gate

Every extracted value and every citation in an answer must carry a quotation
that is found in the text that was supplied to the model. Only whitespace is
normalised before comparison.

### Extraction as an ingest stage with a field registry

Datapoints are extracted when a report is ingested, not when a page is opened,
so the reports list renders from the database. The database stores a field key and a value rather than one column per
datapoint, which means adding a field is one entry in `app/ingest/fields.py` and
nothing else.

### Streaming over server sent events

I used server sent events rather than a plain text stream because the response
carries more than text: the retrieved sources are sent before the first token
and the verified citations after the last one. 


## The application, folder by folder

### `app/`

| File | Purpose |
|---|---|
| `main.py` | The application factory which loads the environment, applies the schema, mounts static files and templates, attaches the routers, and defines the helper for static assets. |
| `config.py` | Reads `.env` without overriding variables already exported, and resolves a stored filename to a PDF on disk with a containment check that blocks path traversal. |
| `verify.py` | The verbatim checker, shared by extraction and by chat. Normalises whitespace, finds which chunk a quotation came from, and pins a quotation to the page it is printed on when a chunk spans a page break. |
| `branding.py` | Maps a company name to a logo file, reusing the alias logic from the retrieval layer. |

### `app/db/`

| File | Purpose |
|---|---|
| `schema.sql` | Creates eleven tables. Every statement is guarded by `IF NOT EXISTS`, so it applies safely on every startup and the repository can use a populated database instead of a migration history. |
| `connection.py` | Where the connection is opened. It loads sqlite-vec, applies the four pragmas the schema depends on, and provides the transaction context manager. |
| `models.py` | The domain objects as dataclasses. Holds the property that prefixes a chunk with its context header for embedding. |

### `app/db/repositories/`

| File | Purpose |
|---|---|
| `base.py` | The generic repository contract every repository implements, plus a bulk create and a helper that turns an insert into a guaranteed row id. |
| `documents.py` | Reports. Lookup by file hash is what makes re-ingesting the same PDF idempotent, and the delete method removes vectors before rows because a vec0 table is unable to cascade. |
| `stage_runs.py` | Ingest progress, which is keyed on the pair of document and stage. It provides the lifecycle methods the pipeline uses to skip finished work and record failures. |
| `blocks.py` | The parser output. It overrides the bulk insert for volume, and provides the two lookups a resumed parse needs: the last page stored and the next free sequence number. |
| `chunks.py` | The retrieval unit, and both halves of the search. Here it holds the BM25 query over FTS5, the nearest neighbour query over the vectors, and the implements a sanitization function that turns free text into a safe match expression. |
| `facts.py` | The extracted datapoints. |
| `chat.py` | Conversations, messages and citations. |

### `app/providers/`

| File | Purpose |
|---|---|
| `base.py` | The language model is split into a text generator and a structured extractor so a call site declares only what it uses. |
| `embeddings.py` | bge-m3 using sentence-transformers. Loads lazily on first use, selects cuda, mps or cpu, normalises vectors to unit length, and asserts the vector width matches the schema. |
| `claude.py` | Claude Opus 5. Provides a single completion, a streamed completion and a structured parse, and raises a distinct error when no API key is configured so the web layer can tell setup apart from failure. |

### `app/ingest/`

| File | Purpose |
|---|---|
| `parse.py` | Stage one. Converts a PDF into blocks in batches of pages, resuming from the last page already stored, and disables model compilation if needed. |
| `chunk.py` | Stage two. Groups blocks into chunks, collapses the table padding, splits oversized tables by row and then by column, and builds the context header each chunk is embedded with. |
| `embed.py` | Stage three. Embeds only the chunks that have no vector yet and commits every hundred. |
| `extract.py` | Stage four. Retrieves candidate excerpts per field, asks the model for structured values, and discards any value whose quotation cannot be found. |
| `fields.py` | The registry of what to extract. |
| `pipeline.py` | Runs the four stages in order and skips those already finished, records failures against the stage that caused them, and identifies a document by the hash of its contents. |
| `naming.py` | Reads a company and a year out of a filename. |

### `app/retrieve/`

| File | Purpose |
|---|---|
| `hybrid.py` | Here we perform the reciprocal rank fusion, round robin interleaving, and the two search entry point being one query, and several phrasings of one query. |
| `scope.py` | Works out which report a question is about by matching the company names and their aliases. |

### `app/chat/`

| File | Purpose |
|---|---|
| `prompts.py` | The system prompt and the prompt builders. |
| `answer.py` | The whole turn in a conversation. Retrieves context, appends the excerpts the extracted facts came from, builds the prompt, parses and verifies the citations in the reply, and stores the question, the answer and the citations in one transaction. |

### `app/routes/`

| File | Purpose |
|---|---|
| `deps.py` | Dependency wiring. The providers are process singletons because the embedding model is expensive to load and the database connection is per request because a connection belongs to the thread that opened it. |
| `pages.py` | The HTML pages, and the route that serves a source PDF inline so a page link can open it at the cited page. |
| `api.py` | The JSON endpoints and the streaming chat endpoint, which emits the retrieved sources, then the answer in fragments and then finally the verified citations. |
| `uploads.py` | This file accepts a PDF, checks it really is one and ingests it on a single background worker while reporting progress from the database and removing a report from the index when needed. |

### `app/templates/` and `app/static/`

| File | Purpose |
|---|---|
| `base.html` | The shared layout. |
| `documents.html` | The landing page which consists of every report with its extracted datapoints, plus the upload form. |
| `document.html` | One report in detail, with each datapoint's quotation and page link. |
| `compare.html` | One field across every report. |
| `chat.html` | The chat interface and the conversation list. |
| `partials/` | The shared rendering of a single datapoint with a few small macros. |
| `static/css/basecoat.css` | Basecoat, which is vendored rahther than using a CDN so the application works with no network. |
| `static/css/app.css` | The palette, the page layout, stlying, etc. |
| `static/js/chat.js` | Reads the event stream, renders the answer and its citations, and manages renaming and deleting conversations. |
| `static/js/upload.js` | Uploads a report, polls for progress, and removes a report from the index. |
| `static/js/confirm.js` | A promise wrapper around the native dialog element. |
| `static/images/` | Company logos. |

### Outside `app/`

| Path | Purpose |
|---|---|
| `tests/` | `conftest.py` holds the fixtures and the two fakes to replace network calls and model loads. |
| `data/pdfs/` | The source reports, committed so the index can be rebuilt from scratch. |
| `data/rag.db` | The populated database. |

## References

Auer, C., Lysak, M., Nassar, A., Dolfi, M., Livathinos, N., Vagenas, P.,
Ramis, C. B., Omenetti, M., Lindlbauer, F., Dinkla, K., Mishra, L., Kim, Y.,
Gupta, S., de Lima, R. T., Weber, V., Morin, L., Meijer, I., Kuropiatnyk, V., &
Staar, P. W. J. (2024). *Docling technical report* (arXiv:2408.09869). arXiv.
https://arxiv.org/abs/2408.09869

Chen, J., Xiao, S., Zhang, P., Luo, K., Lian, D., & Liu, Z. (2024). *BGE
M3-Embedding: Multi-lingual, multi-functionality, multi-granularity text
embeddings through self-knowledge distillation* (arXiv:2402.03216). arXiv.
https://arxiv.org/abs/2402.03216

Cormack, G. V., Clarke, C. L. A., & Buettcher, S. (2009). Reciprocal rank
fusion outperforms Condorcet and individual rank learning methods. In
*Proceedings of the 32nd International ACM SIGIR Conference on Research and
Development in Information Retrieval* (pp. 758-759).
https://doi.org/10.1145/1571941.1572114

Liu, N. F., Lin, K., Hewitt, J., Paranjape, A., Bevilacqua, M., Petroni, F., &
Liang, P. (2024). Lost in the middle: How language models use long contexts.
*Transactions of the Association for Computational Linguistics, 12*, 157-173.
https://doi.org/10.1162/tacl_a_00638

Porter, M. F. (1980). An algorithm for suffix stripping. *Program, 14*(3),
130-137. https://doi.org/10.1108/eb046814

Robertson, S., & Zaragoza, H. (2009). The probabilistic relevance framework:
BM25 and beyond. *Foundations and Trends in Information Retrieval, 3*(4),
333-389. https://doi.org/10.1561/1500000019

Anthropic. (2026). *Anthropic API documentation*.
https://docs.anthropic.com/en/api

Astral. (2026). *uv: An extremely fast Python package and project manager*.
https://docs.astral.sh/uv/

Garcia, A. (2025). *sqlite-vec: A vector search SQLite extension* [Computer
software]. https://github.com/asg017/sqlite-vec

Hunkeler, R. (2025). *Basecoat: A components library built with Tailwind CSS*
[Computer software]. https://basecoatui.com/

SQLite Consortium. (2025). *SQLite FTS5 extension documentation*.
https://www.sqlite.org/fts5.html

Tiangolo. (2025). *FastAPI documentation*. https://fastapi.tiangolo.com/
