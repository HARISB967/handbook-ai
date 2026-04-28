-- Run this in your Supabase SQL editor

-- Session documents table: tracks every file ingested per session
CREATE TABLE IF NOT EXISTS session_documents (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id    UUID NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    workspace_id  TEXT NOT NULL,
    filename      TEXT NOT NULL,
    file_size_bytes BIGINT,
    word_count    INTEGER,
    page_count    INTEGER,
    status        TEXT DEFAULT 'success',  -- 'success' | 'partial' | 'failed' | 'duplicate'
    ingested_at   TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_session_documents_session_id
    ON session_documents(session_id);
