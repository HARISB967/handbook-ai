import os
import io
import hashlib
import asyncio
import pdfplumber
from PyPDF2 import PdfReader
from transformers import pipeline
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
from lightrag.llm.openai import openai_complete_if_cache
from llm_pool import pool_complete
import numpy as np
from typing import Dict
import glob
from database import (
    add_document_record, get_session_documents, 
    delete_document_record, upload_workspace_file,
    supabase
)

# ─────────────────────────────────────────────────────────────────────────────
# Cloud Backup Logic
# ─────────────────────────────────────────────────────────────────────────────
async def backup_workspace(workspace_id: str):
    """Scan the workspace directory and sync all critical files to Supabase Cloud."""
    wdir = os.path.join(BASE_DIR, workspace_id)
    if not os.path.exists(wdir):
        return
    
    # Files to backup: GraphML, JSON KV storages, etc.
    patterns = ["*.graphml", "*.json", "*.bin"]
    files_to_sync = []
    for p in patterns:
        files_to_sync.extend(glob.glob(os.path.join(wdir, p)))
    
    print(f"[BACKUP] Syncing {len(files_to_sync)} file(s) for {workspace_id[:8]}...")
    for fpath in files_to_sync:
        # Run in thread to avoid blocking the event loop
        await asyncio.to_thread(upload_workspace_file, workspace_id, fpath)
    print(f"[BACKUP] Sync complete.")

# ─── Local HuggingFace Embeddings (384 dims — fast, free, matches LightRAG PG schema)
from transformers import pipeline as hf_pipeline

print("[INGESTION] Loading HuggingFace embedding model: BAAI/bge-small-en-v1.5")
_embedding_pipe = hf_pipeline("feature-extraction", model="BAAI/bge-small-en-v1.5")
print("[INGESTION] Embedding model loaded.")

async def local_embedding_func(texts: list[str]) -> np.ndarray:
    """Fast local CLS-token embeddings. Batch size 50 for paid-tier throughput."""
    print(f"[EMBEDDING] Batch embedding {len(texts)} text(s) (local BGE-small)...")
    outs = await asyncio.to_thread(
        _embedding_pipe, texts, truncation=True, max_length=512, batch_size=50
    )
    embeddings = [out[0][0] for out in outs]
    print(f"[EMBEDDING] Done. {len(embeddings)} vectors.")
    return np.array(embeddings)



# ─────────────────────────────────────────────────────────────────────────────
# KG Extraction LLM — Gemini-2.5-Flash @ OpenRouter (Huge rate limits, native JSON)
# ─────────────────────────────────────────────────────────────────────────────
# ── Dynamic LLM Router: OpenRouter (Extraction) vs Groq (Retrieval) ─────────────
from openai import AsyncOpenAI as OpenAIAsync

OAI_CLIENT = OpenAIAsync(api_key=os.getenv("OPENAI_API_KEY"))
KG_MODEL   = "gpt-4o-mini"
KG_CLIENT  = OAI_CLIENT

async def compress_context(raw_context: str, max_tokens: int = 3000) -> str:
    """
    Summarise raw LightRAG context using gpt-4o-mini.
    Increased max_tokens to 3000 to preserve detail.
    Only compresses if context is very large (>8000 chars).
    """
    if not raw_context or len(raw_context) < 8000:
        return raw_context  # Skip if already reasonable
    
    print(f"[COMPRESS] Raw context: {len(raw_context)} chars → compressing (fidelity priority)...")
    try:
        resp = await KG_CLIENT.chat.completions.create(
            model=KG_MODEL,
            messages=[
                {"role": "system", "content": (
                    "You are a master context synthesizer. Condense the knowledge graph context but KEEP ALL CRITICAL DETAILS. "
                    "CRITICAL: Preserving all [SOURCE: ...] tags. "
                    "Do NOT merge different sources. Keep source-specific information grouped. "
                    "Output a detailed, factual summary. Aim for high density."
                )},
                {"role": "user", "content": f"Synthesize this context while preserving citations:\n\n{raw_context[:20000]}"}
            ],
            temperature=0.0,
            max_tokens=max_tokens
        )
        compressed = resp.choices[0].message.content
        print(f"[COMPRESS] Done: {len(raw_context)} → {len(compressed)} chars")
        return compressed
    except Exception as e:
        print(f"[COMPRESS] Error: {e} — using raw context")
        return raw_context


async def kg_llm_func(prompt, system_prompt=None, history_messages=[], **kwargs):
    """
    LLM function for LightRAG:
    - All calls (extraction + retrieval keywords) → gpt-4o-mini (fast, cheap)
    - Pool is NOT used here — clean, direct OpenAI calls
    """
    sys_msg = system_prompt or "You are a precise knowledge extraction assistant."
    # Sanitize: gpt-4o-mini does not accept None content
    if not sys_msg or sys_msg.strip() == "":
        sys_msg = "You are a helpful assistant."
    
    print(f"[KG-LLM] gpt-4o-mini | Prompt: {len(prompt)} chars")
    try:
        response = await KG_CLIENT.chat.completions.create(
            model=KG_MODEL,
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user",   "content": prompt}
            ],
            temperature=kwargs.get("temperature", 0.0),
            max_tokens=kwargs.get("max_tokens", 2000)
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"[KG-LLM] ERROR: {e}")
        raise


# ─────────────────────────────────────────────────────────────────────────────
# Per-session LightRAG Instance Pool
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = "./lightrag_sessions"
os.makedirs(BASE_DIR, exist_ok=True)

_rag_pool: Dict[str, LightRAG] = {}


async def get_rag(workspace_id: str) -> LightRAG:
    """
    Lazy factory: get or create the LightRAG instance for a session.
    workspace_id = session UUID → each session has its own KG + vector namespace.
    """
    if workspace_id in _rag_pool:
        # Update "Last Used" by re-inserting at the end of the dict (Ordered in Python 3.7+)
        instance = _rag_pool.pop(workspace_id)
        _rag_pool[workspace_id] = instance
        print(f"[INGESTION] Pool hit for workspace: {workspace_id[:8]}... (pool size: {len(_rag_pool)})")
        return instance

    # Keep pool size under control (LRU: remove oldest if we exceed 5 sessions)
    if len(_rag_pool) >= 5:
        oldest_sid = next(iter(_rag_pool))
        print(f"[INGESTION] Pool full. Evicting oldest session: {oldest_sid[:8]}...")
        _rag_pool.pop(oldest_sid)

    print(f"[INGESTION] Creating new LightRAG workspace: {workspace_id[:8]}...")
    wdir = os.path.join(BASE_DIR, workspace_id)
    os.makedirs(wdir, exist_ok=True)

    embedding_fn = EmbeddingFunc(
        embedding_dim=384,         # BGE-small-en-v1.5: 384 dims (matches PG schema)
        max_token_size=512,
        func=local_embedding_func
    )

    try:
        instance = LightRAG(
            working_dir=wdir,
            workspace=workspace_id,
            llm_model_func=kg_llm_func,
            llm_model_max_async=8,   # 16→8: halves parallel token usage
            chunk_token_size=600,
            chunk_overlap_token_size=50,
            embedding_func=embedding_fn,
            kv_storage="PGKVStorage",
            graph_storage="NetworkXStorage",
            vector_storage="PGVectorStorage",
        )
        print(f"[INGESTION] LightRAG ready (BGE-small embeddings | gpt-4o-mini KG).")
    except TypeError:
        print(f"[INGESTION] workspace param unsupported — falling back to working_dir.")
        instance = LightRAG(
            working_dir=wdir,
            llm_model_func=kg_llm_func,
            llm_model_max_async=8,   # 16→8: halves parallel token usage
            chunk_token_size=600,
            chunk_overlap_token_size=50,
            embedding_func=embedding_fn,
            kv_storage="PGKVStorage",
            graph_storage="NetworkXStorage",
            vector_storage="PGVectorStorage",
        )

    print(f"[INGESTION] Initialising storages for workspace {workspace_id[:8]}...")
    await instance.initialize_storages()
    _rag_pool[workspace_id] = instance
    print(f"[INGESTION] Workspace {workspace_id[:8]}... ready. Total pool: {len(_rag_pool)} instance(s).")
    return instance


# ─────────────────────────────────────────────────────────────────────────────
# PDF Text Extraction
# ─────────────────────────────────────────────────────────────────────────────
def extract_text_from_pdf_meta(file_bytes: bytes, filename: str) -> tuple[str, int]:
    """
    Extract raw text + page count.
    Injects [SOURCE: {filename}, PAGE: {n}] tags into the text for downstream citation accuracy.
    """
    text = ""
    page_count = 0
    print(f"[PDF] Attempting extraction with pdfplumber for '{filename}'...")
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            page_count = len(pdf.pages)
            print(f"[PDF] Found {page_count} page(s).")
            for i, page in enumerate(pdf.pages):
                extracted = page.extract_text()
                if extracted:
                    # Inject source tag at the start of every page
                    text += f"\n\n[SOURCE: {filename}, PAGE: {i+1}]\n{extracted}\n"
        print(f"[PDF] pdfplumber extracted {len(text):,} tagged characters from {page_count} pages.")
    except Exception as e:
        print(f"[PDF] pdfplumber failed ({e}), falling back to PyPDF2...")
        reader = PdfReader(io.BytesIO(file_bytes))
        page_count = len(reader.pages)
        for i, page in enumerate(reader.pages):
            extracted = page.extract_text()
            if extracted:
                text += f"\n\n[SOURCE: {filename}, PAGE: {i+1}]\n{extracted}\n"
        print(f"[PDF] PyPDF2 extracted {len(text):,} tagged characters from {page_count} pages.")
    return text, page_count


# ─────────────────────────────────────────────────────────────────────────────
# Ingestion Entry Point
# ─────────────────────────────────────────────────────────────────────────────
async def process_pdf(filename: str, file_bytes: bytes, workspace_id: str) -> dict:
    """
    Extract text from PDF and insert into the session-specific LightRAG workspace.
    Returns a metadata dict: {filename, word_count, page_count, file_size_bytes, status}
    """
    print(f"\n{'='*60}")
    print(f"[INGESTION] START: '{filename}' → workspace {workspace_id[:8]}...")
    print(f"[INGESTION] File size: {len(file_bytes):,} bytes")

    meta = {
        "filename":        filename,
        "file_size_bytes": len(file_bytes),
        "word_count":      0,
        "page_count":      0,
        "status":          "success",
        "lightrag_doc_id": None,    # Computed below after text extraction
    }

    raw_text, page_count = await asyncio.to_thread(extract_text_from_pdf_meta, file_bytes, filename)
    meta["page_count"] = page_count

    if not raw_text.strip():
        print(f"[INGESTION] ERROR: No text extracted from '{filename}'. Aborting.")
        meta["status"] = "failed"
        return meta

    word_count = len(raw_text.split())
    meta["word_count"] = word_count

    # Compute the LightRAG document ID EXACTLY as LightRAG does internally
    from lightrag.utils import compute_mdhash_id
    lightrag_doc_id = compute_mdhash_id(raw_text.strip(), prefix="doc-")
    meta["lightrag_doc_id"] = lightrag_doc_id
    print(f"[INGESTION] Extracted {word_count:,} words across {page_count} pages.")
    print(f"[INGESTION] LightRAG doc_id will be: {lightrag_doc_id}")
    print(f"[INGESTION] Sending to LightRAG (Llama-70B KG, serial, 1–2 min/chunk)...")

    rag = await get_rag(workspace_id)

    try:
        await rag.ainsert(raw_text)
        
        # Small delay to allow LightRAG to flush the .graphml file to disk
        await asyncio.sleep(2)
        
        print(f"[INGESTION] SUCCESS: '{filename}' stored in workspace {workspace_id[:8]}...")
        meta["status"] = "success"
        
        # Trigger background backup to Cloud
        asyncio.create_task(backup_workspace(workspace_id))
    except Exception as e:
        err = str(e).lower()
        if "duplicate document" in err or "no new unique documents" in err:
            print(f"[INGESTION] INFO: '{filename}' already in workspace — skipping duplicate.")
            meta["status"] = "duplicate"
        elif "timeout" in err:
            print(f"[INGESTION] WARNING: Extraction timed out for some chunks. Status: PARTIAL.")
            meta["status"] = "partial"
        else:
            print(f"[INGESTION] FATAL ERROR: {e}")
            meta["status"] = "failed"
            # We DON'T raise here so the metadata can still be returned to the API
            # but meta["status"] will trigger the error message in app.py
    
    print(f"{'='*60}\n")
    return meta


# ─────────────────────────────────────────────────────────────────
# Document Deletion
# ─────────────────────────────────────────────────────────────────
async def delete_document_from_workspace(workspace_id: str, lightrag_doc_id: str) -> bool:
    """
    Remove a document from LightRAG: chunks, vectors, and KG entities/relations
    that are exclusive to this document (shared entities are preserved).
    Uses: rag.adelete_by_doc_id(doc_id) — LightRAG 1.4+
    """
    print(f"[DELETE] Removing doc '{lightrag_doc_id}' from workspace {workspace_id[:8]}...")
    rag = await get_rag(workspace_id)
    deleted = False

    # Primary method — LightRAG 1.4+ native
    if hasattr(rag, "adelete_by_doc_id"):
        try:
            await rag.adelete_by_doc_id(lightrag_doc_id)
            print(f"[DELETE] adelete_by_doc_id succeeded for {lightrag_doc_id}")
            deleted = True
        except Exception as e:
            print(f"[DELETE] adelete_by_doc_id error: {e}")

    # Fallback — some builds expose a generic adelete(ids=[...])
    if not deleted and hasattr(rag, "adelete"):
        try:
            await rag.adelete(ids=[lightrag_doc_id])
            print(f"[DELETE] adelete fallback succeeded for {lightrag_doc_id}")
            deleted = True
        except Exception as e:
            print(f"[DELETE] adelete fallback error: {e}")

    # Always evict from pool so next query reloads a fresh instance
    if workspace_id in _rag_pool:
        _rag_pool.pop(workspace_id)
        print(f"[DELETE] Pool cleared for workspace {workspace_id[:8]}... (will reinitialise on next query)")

    # CRITICAL: Clear the LLM cache for this workspace to prevent stale query answers
    try:
        supabase.table("lightrag_llm_cache").delete().eq("workspace", workspace_id).execute()
        print(f"[DELETE] Cleared LLM cache for workspace {workspace_id[:8]}...")
    except Exception as e:
        print(f"[DELETE] Warning: could not clear LLM cache: {e}")

    if not deleted:
        print(f"[DELETE] WARNING: No deletion method available on this LightRAG version. "
              f"Metadata removed from Supabase but KG data persists until workspace is wiped.")

    return deleted


# ─────────────────────────────────────────────────────────────────────────────
# Query Entry Point
# ─────────────────────────────────────────────────────────────────────────────
async def query(question: str, workspace_id: str, history: str = "", top_k: int = 10) -> str:
    """
    Query ONLY the LightRAG workspace for this session.
    - Hybrid mode: graph traversal + vector similarity
    - top_k set to 10 for better context depth.
    - history: Optional conversation context to resolve pronouns/references.
    """
    # Dynamic top_k for broad questions
    broad_keywords = ["summarize", "overall", "what does", "meaning", "about"]
    if any(k in question.lower() for k in broad_keywords):
        top_k = 20
        print(f"[QUERY] Broad question detected. Increasing top_k to {top_k}")

    short_q = question[:80] + ("..." if len(question) > 80 else "")
    print(f"[QUERY] workspace={workspace_id[:8]}... | '{short_q}' (history: {len(history)} chars, top_k: {top_k})")
    
    rag = await get_rag(workspace_id)
    
    full_query = f"Conversation History:\n{history}\n\nCurrent Question: {question}" if history else question

    result = await rag.aquery(
        full_query,
        param=QueryParam(
            mode="hybrid",
            top_k=top_k,
            enable_rerank=False,
            only_need_context=True  # CRITICAL: We need the raw context for our 120B model
        )
    )
    # If result is empty or just whitespace, return a helpful error
    if not result or len(result.strip()) < 5:
        print(f"[QUERY] WARNING: No context found for '{short_q}'")
        return ""

    print(f"[QUERY] Context retrieved: {len(result)} chars.")
    return result
