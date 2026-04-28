-- Supabase pgvector setup

-- Enable pgvector extension
create extension if not exists vector;

-- Create documents table for embeddings
create table if not exists documents (
  id uuid primary key default gen_random_uuid(),
  content text,
  embedding vector(384),
  metadata jsonb,
  created_at timestamptz default now()
);

-- Create chat_history table for persistent memory
create table if not exists chat_history (
  id uuid primary key default gen_random_uuid(),
  role text, -- 'user' or 'assistant'
  content text,
  created_at timestamptz default now()
);

-- Create match_documents RPC function for similarity search
create or replace function match_documents (
  query_embedding vector(384),
  match_threshold float,
  match_count int
)
returns table (
  id uuid,
  content text,
  metadata jsonb,
  similarity float
)
language sql stable
as $$
  select
    documents.id,
    documents.content,
    documents.metadata,
    1 - (documents.embedding <=> query_embedding) as similarity
  from documents
  where 1 - (documents.embedding <=> query_embedding) > match_threshold
  order by documents.embedding <=> query_embedding
  limit match_count;
$$;
