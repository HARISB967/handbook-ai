-- Run this in Supabase SQL Editor

-- Add lightrag_doc_id to session_documents for real deletion support
ALTER TABLE session_documents
    ADD COLUMN IF NOT EXISTS lightrag_doc_id TEXT;

-- Index for fast lookup
CREATE INDEX IF NOT EXISTS idx_session_documents_lightrag_doc_id
    ON session_documents(lightrag_doc_id)
    WHERE lightrag_doc_id IS NOT NULL;
