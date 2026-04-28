-- Chat Sessions Table
create table if not exists chat_sessions (
  id uuid primary key default gen_random_uuid(),
  name text not null default 'New Chat',
  created_at timestamptz default now(),
  updated_at timestamptz default now()
);

-- Alter existing chat_history to link to a session
-- (Run this if chat_history table already exists)
alter table chat_history add column if not exists session_id uuid references chat_sessions(id) on delete cascade;
create index if not exists idx_chat_history_session on chat_history(session_id);
