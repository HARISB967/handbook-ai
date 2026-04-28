# 🚀 AGENTS.md — HandBook.ai AI Agent Operating Rules & Project Spec

---

## 🎯 Project Objective

**HandBook.ai** is an autonomous AI-powered document intelligence platform. Its mission is to let users upload PDFs, engage in contextual Q&A, and generate exhaustive long-form handbooks (up to 20,000 words) — all through a clean, conversational chat interface.

The agent (you) is responsible for designing, building, debugging, and continuously improving this application end-to-end.

Always optimize for: **Correctness → Simplicity → Maintainability → Performance**.

---

## 🧠 Core Agent Behavior

| Principle | Rule |
|---|---|
| **Think First** | Analyze before acting. Break every problem into steps before writing a single line of code. |
| **Read Before Writing** | Always read existing files before modifying them. Never rewrite a codebase unnecessarily. |
| **Stay DRY** | No duplication. Abstract repeated logic into shared utilities or services. |
| **Fail Loudly** | Meaningful error messages and logs everywhere. Silent failures are bugs. |
| **Scope Discipline** | Create new files only when genuinely necessary. Respect the established architecture. |

---

## 🏗️ Architecture Overview

```
handbook/
├── app.py              # FastAPI backend — routes & orchestration
├── ui.py               # Streamlit frontend — all UI components
├── ingestion.py        # PDF parsing, chunking, embedding
├── database.py         # Supabase client + pgvector operations
├── longwriter.py       # AgentWrite pipeline — 20,000-word generation
├── requirements.txt    # Python dependencies
└── .env                # Secrets (never committed)
```

**Separation of Concerns:**
- **`ui.py`** — UI only. No business logic.
- **`app.py`** — Route definitions only. Delegates to services.
- Heavy logic lives in `ingestion.py`, `database.py`, and `longwriter.py`.

---

## 🔐 Security Rules (Non-Negotiable)

- **NEVER** hardcode API keys, secrets, or credentials anywhere in source code.
- Load all secrets exclusively via `python-dotenv` from the `.env` file.
- `.env` must be listed in `.gitignore` — no exceptions.

Required `.env` keys:
```
# Supabase
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_KEY=your-service-key

# LLM (NVIDIA)
NVIDIA_API_KEY=your-nvidia-key

# Embeddings (Hugging Face)
HF_TOKEN=your-hf-token
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | Streamlit |
| **Backend** | Python + FastAPI |
| **Database** | Supabase PostgreSQL with `pgvector` |
| **RAG Framework** | LightRAG |
| **LLM** | NVIDIA Nemotron-3-Super-120B (`nvidia/nemotron-3-super-120b-a12b`) |
| **Embeddings** | HuggingFace `BAAI/bge-small-en-v1.5` via `transformers` pipeline |
| **PDF Parsing** | `pdfplumber` (preferred) or `PyPDF2` |

---

## ⚡ Performance Guidelines

- Avoid unnecessary Streamlit re-renders. Use `st.session_state` to cache state across interactions.
- Batch embedding calls — never embed one chunk at a time in a loop.
- Optimize pgvector queries with appropriate indexes (`ivfflat` or `hnsw`).
- Stream LLM responses wherever possible to reduce perceived latency.

---

## 🧪 Testing & Debugging

- Write modular, testable functions. Each function does one thing.
- Log meaningful debug information at ingestion, retrieval, and generation steps.
- Validate LightRAG context retrieval before passing to the LLM.
- For AgentWrite, log word count per section to catch truncation issues early.

---

---

# 📋 BUILD PIPELINE

---

## Phase 1 — Environment & Supabase Setup

### 1.1 Project Initialization
Create the following files to bootstrap the project:
- `app.py` — FastAPI application entry point
- `ui.py` — Streamlit UI entry point
- `requirements.txt` — all Python dependencies pinned

### 1.2 Supabase Integration (`database.py`)
- Use the official Supabase Python client.
- Read `SUPABASE_URL` and `SUPABASE_SERVICE_KEY` from `.env`.
- Implement helper functions for:
  - Persistent chat memory operations (`add_chat_history`, `get_chat_history`).
- Database Tables: Create the `chat_history` table for memory. (Note: LightRAG will automatically create its own Postgres tables for vectors and graphs later).

-- Persistent Chat History Table
create table chat_history (
  id uuid primary key default gen_random_uuid(),
  role text, -- 'user' or 'assistant'
  content text,
  created_at timestamptz default now()
);

---

## Phase 2 — Ingestion & LightRAG Setup

### 2.1 PDF Extraction & Single Storage (`ingestion.py`)
- Create a function that accepts a list of uploaded PDF files.
- Extract and clean raw text using `pdfplumber` (fallback to `PyPDF2`).
- Provide the raw extracted text directly to LightRAG (`rag.insert()`).
- LightRAG natively handles text chunking, Knowledge Graph entity extraction, and vector storage seamlessly via Postgres.

### 2.2 Embeddings
Use the HuggingFace `transformers` pipeline for efficient local embedding — no external API calls needed for embeddings:

```python
from transformers import pipeline

pipe = pipeline("feature-extraction", model="BAAI/bge-small-en-v1.5")


```

### 2.3 LightRAG Integration
- Connect LightRAG's `PGKVStorage`, `PGVectorStorage`, and `PGGraphStorage` natively to the Supabase PostgreSQL connection string.
- Inject the HuggingFace `transformers` pipeline as LightRAG's `embedding_func`.
- **CRITICAL:** Inject the NVIDIA Nemotron LLM client as LightRAG's `llm_model_func`. LightRAG requires this LLM during `insert()` to extract entities and build the knowledge graph.
- Expose a `query(question: str) -> str` utility that retrieves the top-k most relevant nodes/chunks from the graph and returns formatted context.
---

## Phase 3 — NVIDIA LLM & AgentWrite Pipeline

### 3.1 NVIDIA LLM Client
HandBook.ai uses the NVIDIA Nemotron-3-Super-120B model accessed via NVIDIA's OpenAI-compatible inference API:

```python
from openai import OpenAI

client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = "$NVIDIA_API_KEY"
)


completion = client.chat.completions.create(
  model="nvidia/nemotron-3-super-120b-a12b",
  messages=[{"role":"user","content":""}],
  temperature=1,
  top_p=0.95,
  extra_body={"chat_template_kwargs":{"enable_thinking":True},"reasoning_budget":16384},
  stream=True
)

```

Always read `NVIDIA_API_KEY` from the environment. Never hardcode it.

### 3.2 The Intent Router (Dynamic Routing)
- Accepts a user `question` via POST request.

Before processing any user message, implement a lightweight router function to determine if the query requires standard RAG or the LongWriter pipeline.
```python
ROUTER_PROMPT = """
Analyze the following user request. Does the user want a short conversational answer, or are they asking for a comprehensive, long-form document, essay, or handbook (especially if they mention a high word count)?
Reply with ONLY the word "SHORT" or "LONG". Do not output anything else.

User Request: {question}
"""
```

### 3.3 Standard Chat Endpoint (`app.py`)

If the Router returns "SHORT":
- Retrieves relevant context from LightRAG.
- Sends context + question to the NVIDIA model.
- Returns a concise, well-cited answer.
- Stream the response token-by-token for a responsive UI experience.
If the Router returns "LONG":
- the longwriter.py should handle that


### 3.4 AgentWrite — Long-Form Handbook Generator (`longwriter.py`)

HandBook.ai's flagship feature: generating a coherent, structured 20,000-word handbook from ingested PDF knowledge using the divide-and-conquer writing strategy with exact prompts validated by the LongWriter research paper.

#### Step 1 — Plan (Outline Generation)
Prompt the NVIDIA model to break the instruction down into paragraph-by-paragraph subtasks.

```python
PLAN_PROMPT = """
I need you to help me break down the following long-form writing instruction into multiple subtasks. Each subtask will guide the writing of one paragraph in the essay, and should include the main points and word count requirements for that paragraph.

The writing instruction is as follows:
{instruction}
Context from Knowledge Base:
{context}

Please break it down in the following format, with each subtask taking up one line:
Paragraph 1 - Main Point: [Describe the main point of the paragraph, in detail] - Word Count: [Word count requirement, e.g., 400 words]
Paragraph 2 - Main Point: [Describe the main point of the paragraph, in detail] - Word Count: [word count requirement, e.g. 1000 words].
...

Make sure that each subtask is clear and specific, and that all subtasks cover the entire content of the writing instruction. Do not split the subtasks too finely; each subtask's paragraph should be no less than 200 words and no more than 1000 words. Do not output any other content.
"""
```

#### Step 2 — Write (Iterative Generation)
Parse the output of the Plan step into a list of steps. Loop through the steps. 
**CRITICAL RAG STEP:** For each step, query LightRAG using the `current_step` description to retrieve fresh, highly relevant context from the uploaded PDFs.

```python
WRITE_PROMPT = """
You are an excellent technical writer. I will give you an original writing instruction, my planned writing steps, the text I have already written, and factual context retrieved from the user's documents. 

Writing instruction:
{instruction}

Writing steps:
{plan}

Knowledge Base Context (MUST USE THIS TO GROUND YOUR WRITING):
{context}

Already written text:
{already_written_text}

Please integrate the instruction, the factual context, and the already written text, and now write: {current_step}. 
If needed, you can add a small subtitle at the beginning. Remember to only output the paragraph you write, without repeating the already written text. As this is an ongoing work, omit open-ended conclusions or other rhetorical hooks.
You MUST include inline citations to the provided Knowledge Base Context (e.g., [Source Name]) to back up your claims.
"""

#### Step 3 — Assembly

- Concatenate already_written_text as the loop progresses.
- Yield/stream progress updates to the frontend (e.g., "Writing Paragraph 4 of 25...").
- Return the final concatenated markdown string.
- Concatenate chunks. Prepend a Table of Contents. Save to Supabase chat_history


## Phase 4 — Streamlit UI

### 4.1 Sidebar — Document Ingestion
- `st.file_uploader` accepting `.pdf` files only.
- MUST set accept_multiple_files=True. Triggers the FastAPI ingestion endpoint.
- On upload, POST the file to the FastAPI ingestion endpoint.
- Display a progress spinner during ingestion.
- Show a success toast with chunk count on completion.
- Store ingestion state in `st.session_state` to avoid re-uploading on re-render.

### 4.2 Main Interface — Conversational Q&A
- Load past conversation history from Supabase on startup. Use st.chat_message and st.chat_input for standard Q&A.
- Use `st.chat_message` for both user and assistant turns.
- Use `st.chat_input` for message entry.
- Stream assistant responses token-by-token using `st.write_stream`.
- Maintain full conversation history in `st.session_state.messages`.

### 4.3 Handbook Trigger — AgentWrite Activation
Because the backend Router handles the logic, you do not need specific UI buttons for the handbook.

When the user submits a message, the UI waits for the backend response.

If the backend returns a type: "stream" response, render it normally in the chat.

If the backend detects a "LONG" request, it should return status updates to the UI. Display a st.status progress block that updates as the backend works:

✅ Outline generated (22 paragraphs planned)
✍️ Writing Paragraph 1...
✍️ Writing Paragraph 2...
...
✅ Handbook complete — 20,412 words
On completion, render the full handbook in a scrollable st.expander and provide a st.download_button to export it as a .md file.    

### 4.4 UX Rules
- Never block the UI without a spinner or progress indicator.
- Always display word count and section count after handbook generation.
- Show clear error messages (not raw stack traces) when backend calls fail.
- Keep the sidebar and main chat visually separated.

---

## 📦 `requirements.txt` (Baseline)

```
fastapi
uvicorn
streamlit
supabase
python-dotenv
openai
transformers
torch
pdfplumber
PyPDF2
lightrag
pgvector
httpx
```

---

## 🗂️ File Responsibility Matrix

| File | Owns |
|---|---|
| `app.py` | API routes, request validation, response formatting |
| `ui.py` | All Streamlit UI, session state, user interactions |
| `database.py` | Supabase connection, upsert, similarity search, memory |
| `ingestion.py` | PDF parsing, chunking, embedding, indexing |
| `longwriter.py` | `AgentWrite` class, plan/write/assemble pipeline |
| `.env` | All secrets — never imported directly except via `dotenv` |

---

*HandBook.ai — Escape the gravity of information overload.* 🚀
