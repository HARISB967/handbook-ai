import asyncio
import os
import json
import time
import sys
from collections import deque
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from contextlib import asynccontextmanager

# ── Ports ─────────────────────────────────────────────────────────────────────
BACKEND_PORT  = 8001
FRONTEND_PORT = 8502

# ── In-memory log buffer ───────────────────────────────────────────────────────
_LOG_BUFFER: deque = deque(maxlen=200)

class _Tee:
    def __init__(self, stream):
        self._stream = stream
    def write(self, msg):
        self._stream.write(msg)
        if msg.strip():
            _LOG_BUFFER.append(msg.rstrip())
    def flush(self):
        self._stream.flush()

sys.stdout = _Tee(sys.stdout)

load_dotenv()
print("[APP] Environment variables loaded.")
print(f"[APP] Backend port: {BACKEND_PORT}")

from ingestion import process_pdf, delete_document_from_workspace, _rag_pool
from longwriter import generate_short_answer, generate_handbook
from router import classify_intent
from database import add_document_record, get_session_documents, delete_document_record


# ── Lifespan ───────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[APP] HandBook.ai API starting up.")
    print("[APP] LightRAG pool: instances created lazily per session workspace.")
    print(f"[APP] Models: Router=Llama-8B | KG=gpt-4o-mini | Generation=gpt-4o")
    yield
    print(f"[APP] Shutdown: clearing {len(_rag_pool)} LightRAG instance(s) from pool.")
    _rag_pool.clear()


app = FastAPI(
    title="HandBook.ai API",
    description="Autonomous Document Intelligence — multi-model, per-session RAG",
    version="2.0.0",
    lifespan=lifespan,
)


class ChatRequest(BaseModel):
    message:    str
    session_id: str          # Routes query to the correct LightRAG workspace
    doc_names:  list[str] = []   # All filenames in this session (for multi-doc awareness)
    history:    list[dict] = []  # Recent messages for contextual RAG retrieval


# ── Root / Health ──────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {
        "service": "HandBook.ai API",
        "status": "running",
        "port": BACKEND_PORT,
        "active_workspaces": len(_rag_pool),
    }

@app.get("/health")
async def health_check():
    print("[API] GET /health — OK")
    return {"status": "healthy", "workspaces": len(_rag_pool)}


# ── Live Logs ──────────────────────────────────────────────────────────────────
@app.get("/logs")
async def get_logs():
    return list(_LOG_BUFFER)

@app.get("/logs/clear")
async def clear_logs():
    _LOG_BUFFER.clear()
    print("[APP] Log buffer cleared.")
    return {"status": "cleared"}


# ── Delete document metadata record ────────────────────────────────────────────
@app.delete("/document/{doc_id}")
async def delete_document(doc_id: str):
    """
    Delete a document:
    1. Fetch metadata (workspace_id + lightrag_doc_id) from Supabase
    2. Call LightRAG.adelete_by_doc_id → removes chunks, vectors, KG entities
    3. Remove metadata record from Supabase
    """
    print(f"[API] DELETE /document/{doc_id}")
    try:
        # Fetch the record first so we have workspace_id and lightrag_doc_id
        from database import supabase
        rec = supabase.table("session_documents").select("*").eq("id", doc_id).execute()
        if not rec.data:
            print(f"[API] DELETE: doc_id {doc_id} not found in session_documents")
            return {"status": "not_found", "doc_id": doc_id}

        row            = rec.data[0]
        workspace_id   = row.get("workspace_id")
        lightrag_id    = row.get("lightrag_doc_id")
        filename       = row.get("filename", "unknown")
        print(f"[API] DELETE: '{filename}' | workspace={workspace_id[:8] if workspace_id else '?'}... | rag_id={lightrag_id}")

        # Delete from LightRAG (chunks, vectors, KG)
        rag_deleted = False
        if workspace_id and lightrag_id:
            rag_deleted = await delete_document_from_workspace(workspace_id, lightrag_id)
        else:
            print(f"[API] DELETE: Missing workspace_id or lightrag_doc_id — skipping RAG deletion")

        # Delete metadata record from Supabase
        delete_document_record(doc_id)

        return {
            "status":      "deleted",
            "doc_id":      doc_id,
            "filename":    filename,
            "rag_deleted": rag_deleted,
        }
    except Exception as e:
        print(f"[API] DELETE /document ERROR: {e}")
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail=str(e))


# ── List session documents ──────────────────────────────────────────────────
@app.get("/session/{session_id}/documents")
async def list_session_documents(session_id: str):
    """List all documents ingested for a session."""
    print(f"[API] GET /session/{session_id[:8]}.../documents")
    docs = get_session_documents(session_id)
    return docs

@app.get("/session/{session_id}/graph")
async def get_session_graph(session_id: str):
    """Serve the session's Knowledge Graph in GraphML format."""
    print(f"[API] GET /session/{session_id[:8]}.../graph")
    # Check both potential paths (single-nested vs double-nested by workspace param)
    path1 = os.path.join("lightrag_sessions", session_id, "graph_chunk_entity_relation.graphml")
    path2 = os.path.join("lightrag_sessions", session_id, session_id, "graph_chunk_entity_relation.graphml")
    
    graph_path = path2 if os.path.exists(path2) else path1
    
    if not os.path.exists(graph_path):
        print(f"[GRAPH] Not found at {path1} or {path2}")
        return {"error": "Graph not generated yet."}
    
    print(f"[GRAPH] Serving from {graph_path}")
    with open(graph_path, "r", encoding="utf-8") as f:
        content = f.read()

    return {"graphml": content}


# ── Ingest (SSE, non-blocking, session-scoped) ─────────────────────────────────
@app.post("/ingest/stream")
async def ingest_stream(
    file: UploadFile = File(...),
    session_id: str  = Form(...)
):
    """
    Upload and index a PDF into the session-specific LightRAG workspace.
    Returns SSE progress updates.
    """
    filename = file.filename
    print(f"\n[API] POST /ingest/stream | file='{filename}' | workspace={session_id[:8]}...")

    async def generator():
        start = time.time()
        try:
            yield f"data: {json.dumps({'type': 'progress', 'content': f'📄 Reading {filename}…'})}\n\n"
            contents = await file.read()
            print(f"[API] Read {len(contents):,} bytes")

            yield f"data: {json.dumps({'type': 'progress', 'content': f'🔍 Extracting text from {filename}…'})}\n\n"
            await asyncio.sleep(0)

            yield f"data: {json.dumps({'type': 'progress', 'content': '🧠 Extracting Entities & Building Graph (OpenAI gpt-4o-mini)…'})}\n\n"
            meta = await process_pdf(filename, contents, session_id)


            # Persist document metadata to Supabase (with lightrag_doc_id for real deletion later)
            try:
                add_document_record(
                    session_id=session_id,
                    workspace_id=session_id,
                    filename=meta["filename"],
                    file_size_bytes=meta["file_size_bytes"],
                    word_count=meta["word_count"],
                    page_count=meta["page_count"],
                    status=meta["status"],
                    lightrag_doc_id=meta.get("lightrag_doc_id"),
                )
            except Exception as db_err:
                print(f"[API] WARNING: Failed to save document metadata: {db_err}")

            elapsed = time.time() - start
            if meta["status"] == "failed":
                print(f"[API] /ingest/stream FAILED for {filename}")
                yield f"data: {json.dumps({'type': 'error', 'content': f'❌ {filename} failed to ingest. Check terminal logs.'})}\n\n"
            else:
                status_label = "✅" if meta["status"] in ["success", "duplicate"] else "⚠️ Partial"
                summary = (
                    f"{status_label} {filename} ingested "
                    f"({meta['word_count']:,} words, {meta['page_count']} pages, {elapsed:.0f}s)"
                )
                print(f"[API] /ingest/stream done: {summary}")
                yield f"data: {json.dumps({'type': 'done', 'content': summary})}\n\n"

        except Exception as e:
            print(f"[API] ERROR during ingestion: {e}")
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

    return StreamingResponse(generator(), media_type="text/event-stream")


# ── Chat (SSE, session-scoped) ─────────────────────────────────────────────────
@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """
    Intent Router → SHORT (stream RAG answer) | LONG (AgentWrite).
    Queries are scoped to the session's LightRAG workspace.
    """
    print(f"\n[API] POST /chat | workspace={request.session_id[:8]}... | msg='{request.message[:60]}'")

    async def sse_generator():
        start = time.time()
        try:
            print("[ROUTER] Classifying intent...")
            intent = await classify_intent(request.message)
            print(f"[ROUTER] Intent = {intent}")
            yield f"data: {json.dumps({'type': 'intent', 'content': intent})}\n\n"

            if intent == "SHORT":
                print(f"[ROUTER] → SHORT path | docs_in_session={request.doc_names}")
                count = 0
                async for chunk in generate_short_answer(
                    request.message, request.session_id, request.doc_names, request.history
                ):
                    count += 1
                    yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"
                print(f"[ROUTER] SHORT complete. {count} chunks.")
            else:
                print("[ROUTER] → LONG path (AgentWrite)")
                async for obj in generate_handbook(request.message, request.session_id, request.history):
                    print(f"[AGENTWRITE] → {obj.get('type')} | {str(obj.get('content',''))[:50]}")
                    yield f"data: {json.dumps(obj)}\n\n"

        except Exception as e:
            print(f"[API] ERROR in /chat: {e}")
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

        print(f"[API] /chat closed. {time.time()-start:.1f}s total.\n")

    return StreamingResponse(sse_generator(), media_type="text/event-stream")
