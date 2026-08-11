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

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text,
    content='chunks',
    content_rowid='id',
    tokenize='porter unicode61'
);

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

CREATE VIRTUAL TABLE IF NOT EXISTS chunk_vectors USING vec0(
    chunk_id  INTEGER PRIMARY KEY,
    embedding FLOAT[1024]
);

CREATE TABLE IF NOT EXISTS extracted_facts (
    id                INTEGER PRIMARY KEY,
    document_id       INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    field_key         TEXT    NOT NULL,
    value_raw         TEXT,             
    value_numeric     REAL,             
    unit              TEXT,             -- FTE, EUR...
    verbatim_quote    TEXT    NOT NULL,
    page_no           INTEGER NOT NULL,
    chunk_id          INTEGER REFERENCES chunks(id) ON DELETE SET NULL,
    confidence        REAL,
    extractor_version TEXT    NOT NULL,
    created_at        TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_facts_doc_field ON extracted_facts(document_id, field_key);

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

CREATE TABLE IF NOT EXISTS citations (
    id         INTEGER PRIMARY KEY,
    message_id INTEGER NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    chunk_id   INTEGER NOT NULL REFERENCES chunks(id)   ON DELETE CASCADE,
    quote      TEXT    NOT NULL,
    page_no    INTEGER NOT NULL,
    verified   INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_citations_message ON citations(message_id);
