-- Applied by db.connection.init_db() on every startup. Every statement is
-- idempotent, this is to ensure if the database is already populated we do not overwrite. 

-- Three requirements guide the schema, those being:
-- ingestion is resumable, so stage progress is tracked per document
-- answers are traceable, so verbatim text and a page ride on every fact
-- retrieval is hybrid, so chunks are indexed both lexically and densely

CREATE TABLE IF NOT EXISTS documents (
    id          INTEGER PRIMARY KEY,
    filename    TEXT    NOT NULL,
    file_hash   TEXT    NOT NULL UNIQUE,
    company     TEXT    NOT NULL,
    year        INTEGER NOT NULL,
    page_count  INTEGER,
    created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_documents_company_year ON documents(company, year);

CREATE TABLE IF NOT EXISTS stage_runs (
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    stage       TEXT    NOT NULL CHECK (stage  IN ('parse','chunk','embed','extract')),
    status      TEXT    NOT NULL CHECK (status IN ('pending','running','done','failed')),
    started_at  TEXT,
    finished_at TEXT,
    error       TEXT,
    PRIMARY KEY (document_id, stage)
);

CREATE TABLE IF NOT EXISTS blocks (
    id          INTEGER PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    seq         INTEGER NOT NULL,
    page_no     INTEGER NOT NULL,
    label       TEXT    NOT NULL,
    -- Whatever heading depth the parser reported, and NULL for anything that
    -- is not a heading. 0 is a document title, 1 a top level section.
    --
    -- Be aware that Docling reports every heading in these reports as level 1,
    -- so on this corpus the column carries no more information than `label`
    -- does. It is captured anyway because it is free to record and can only be
    -- recovered by parsing the document again, and because a differently
    -- structured PDF may well populate it properly.
    --
    -- Real heading hierarchy, if chunking turns out to need it, is derivable
    -- from `bbox`: a heading's height tracks its font size closely enough to
    -- rank headings against each other. That works off stored data, so it
    -- costs seconds rather than another parse.
    level       INTEGER,
    text        TEXT    NOT NULL,
    bbox        TEXT,               
    UNIQUE (document_id, seq)
);

CREATE INDEX IF NOT EXISTS idx_blocks_doc_page ON blocks(document_id, page_no);

CREATE TABLE IF NOT EXISTS chunks (
    id             INTEGER PRIMARY KEY,
    document_id    INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    seq            INTEGER NOT NULL,
    page_start     INTEGER NOT NULL,
    page_end       INTEGER NOT NULL,
    section        TEXT,            -- nearest preceding section_header
    chunk_type     TEXT    NOT NULL CHECK (chunk_type IN ('prose','table')),
    context_header TEXT    NOT NULL,
    text           TEXT    NOT NULL,
    token_count    INTEGER,
    UNIQUE (document_id, seq)
);

CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(document_id);


-- Lexical half of hybrid retrieval. This is an external content table: the
-- rows live in `chunks` and FTS5 stores only the inverted index, so chunk text
-- is not duplicated. Gives BM25 ranking with no extra dependency.
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text,
    content='chunks',
    content_rowid='id',
    tokenize='porter unicode61'
);

-- External content FTS5 tables are not maintained automatically. Without these
-- three triggers the index silently drifts out of sync with `chunks`.
CREATE TRIGGER IF NOT EXISTS chunks_fts_insert AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
END;

CREATE TRIGGER IF NOT EXISTS chunks_fts_delete AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.id, old.text);
END;

CREATE TRIGGER IF NOT EXISTS chunks_fts_update AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.id, old.text);
    INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
END;


-- Dense half of hybrid retrieval, provided by the sqlite-vec extension.
--
-- The dimension is fixed at write time and must match the embedding model,
-- where bge-m3 produces 1024 values. A model swap that changes dimensions
-- fails loudly on the first insert rather than corrupting the index, which is
-- why no separate model version bookkeeping is needed here.
--
-- vec0 tables cannot declare foreign keys, so `chunk_id` is a logical
-- reference only. Deletes have to be cascaded in application code.
CREATE VIRTUAL TABLE IF NOT EXISTS chunk_vectors USING vec0(
    chunk_id  INTEGER PRIMARY KEY,
    embedding FLOAT[1024]
);


-- Extracted facts

-- Computed at ingest time so the UI never waits on a model call. This is what
-- backs the "pre-extracted data visible in the application" requirement.
--
-- Generic key and value rather than one column per field, so adding a field is
-- an entry in fields.py rather than a schema change.
--
-- Deliberately not unique on (document_id, field_key). A company has one FTE
-- figure but several sustainability goals, and a unique constraint would
-- quietly discard all but one of them.
CREATE TABLE IF NOT EXISTS extracted_facts (
    id                INTEGER PRIMARY KEY,
    document_id       INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    field_key         TEXT    NOT NULL,
    value_raw         TEXT,             -- as printed in the report, e.g. "20,417"
    value_numeric     REAL,             -- parsed, for sorting and comparison
    unit              TEXT,             -- FTE, EUR m, tCO2e, ...
    -- Must appear byte for byte in the referenced chunk's `text`. Checked
    -- before the row is written; extractions that fail the check are rejected.
    verbatim_quote    TEXT    NOT NULL,
    page_no           INTEGER NOT NULL,
    chunk_id          INTEGER REFERENCES chunks(id) ON DELETE SET NULL,
    confidence        REAL,
    -- Lets a re-run supersede earlier rows instead of duplicating them.
    extractor_version TEXT    NOT NULL,
    created_at        TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_facts_doc_field ON extracted_facts(document_id, field_key);


-- Chat

CREATE TABLE IF NOT EXISTS conversations (
    id         INTEGER PRIMARY KEY,
    title      TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS messages (
    id              INTEGER PRIMARY KEY,
    conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role            TEXT    NOT NULL CHECK (role IN ('user','assistant')),
    content         TEXT    NOT NULL,
    created_at      TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id, id);


-- One row per citation attached to an assistant message. `verified` records
-- whether the quote was found byte for byte in the cited chunk. Unverified
-- citations are dropped before display rather than shown with a caveat.
CREATE TABLE IF NOT EXISTS citations (
    id         INTEGER PRIMARY KEY,
    message_id INTEGER NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    chunk_id   INTEGER NOT NULL REFERENCES chunks(id)   ON DELETE CASCADE,
    quote      TEXT    NOT NULL,
    page_no    INTEGER NOT NULL,
    verified   INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_citations_message ON citations(message_id);
