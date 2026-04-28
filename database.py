import os
from supabase import create_client, Client
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
import uuid

load_dotenv()

SUPABASE_URL         = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    raise ValueError("Supabase URL and Service Key must be set in the environment.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)


# ─────────────────────────────────────────────────────────────────────────────
#  Chat Session CRUD
# ─────────────────────────────────────────────────────────────────────────────

def create_session(name: str = "New Chat") -> Dict[str, Any]:
    print(f"[DB] create_session: name='{name}'")
    try:
        response = supabase.table("chat_sessions").insert({"name": name}).execute()
        result = response.data[0]
        print(f"[DB] create_session: id={result['id']}")
        return result
    except Exception as e:
        print(f"[DB] create_session ERROR: {e}")
        raise

def list_sessions() -> List[Dict[str, Any]]:
    print("[DB] list_sessions...")
    try:
        response = (
            supabase.table("chat_sessions")
            .select("*")
            .order("created_at", desc=True)
            .execute()
        )
        print(f"[DB] list_sessions: {len(response.data)} session(s).")
        return response.data
    except Exception as e:
        print(f"[DB] list_sessions ERROR: {e}")
        raise

def rename_session(session_id: str, new_name: str) -> None:
    print(f"[DB] rename_session: {session_id[:8]}... → '{new_name}'")
    try:
        supabase.table("chat_sessions").update({"name": new_name}).eq("id", session_id).execute()
        print(f"[DB] rename_session: OK")
    except Exception as e:
        print(f"[DB] rename_session ERROR: {e}")
        raise

def delete_session(session_id: str) -> None:
    print(f"[DB] delete_session: {session_id[:8]}...")
    try:
        supabase.table("chat_sessions").delete().eq("id", session_id).execute()
        print(f"[DB] delete_session: OK (cascade)")
    except Exception as e:
        print(f"[DB] delete_session ERROR: {e}")
        raise


# ─────────────────────────────────────────────────────────────────────────────
#  Chat History (per-session)
# ─────────────────────────────────────────────────────────────────────────────

def add_chat_history(role: str, content: str, session_id: Optional[str] = None) -> None:
    preview = content[:60] + ("..." if len(content) > 60 else "")
    print(f"[DB] add_chat_history: role={role}, session={session_id[:8] if session_id else None}, '{preview}'")
    try:
        supabase.table("chat_history").insert({
            "role": role, "content": content, "session_id": session_id,
        }).execute()
        print(f"[DB] add_chat_history: OK")
    except Exception as e:
        print(f"[DB] add_chat_history ERROR: {e}")
        raise

def get_chat_history(session_id: Optional[str] = None) -> List[Dict[str, Any]]:
    print(f"[DB] get_chat_history: session={session_id[:8] if session_id else None}")
    try:
        q = supabase.table("chat_history").select("*").order("created_at", desc=False)
        if session_id:
            q = q.eq("session_id", session_id)
        response = q.execute()
        print(f"[DB] get_chat_history: {len(response.data)} message(s).")
        return response.data
    except Exception as e:
        print(f"[DB] get_chat_history ERROR: {e}")
        raise


# ─────────────────────────────────────────────────────────────────────────────
#  Session Documents (ingestion metadata)
# ─────────────────────────────────────────────────────────────────────────────

def add_document_record(
    session_id:      str,
    workspace_id:    str,
    filename:        str,
    file_size_bytes: int  = 0,
    word_count:      int  = 0,
    page_count:      int  = 0,
    status:          str  = "success",
    lightrag_doc_id: str  = None,
) -> Dict[str, Any]:
    """Record a document ingested into a session workspace."""
    print(f"[DB] add_document_record: session={session_id[:8]}... file='{filename}' status={status} doc_id={lightrag_doc_id}")
    try:
        payload = {
            "session_id":       session_id,
            "workspace_id":     workspace_id,
            "filename":         filename,
            "file_size_bytes":  file_size_bytes,
            "word_count":       word_count,
            "page_count":       page_count,
            "status":           status,
        }
        if lightrag_doc_id:
            payload["lightrag_doc_id"] = lightrag_doc_id
        response = supabase.table("session_documents").insert(payload).execute()
        result = response.data[0]
        print(f"[DB] add_document_record: OK id={result['id']}")
        return result
    except Exception as e:
        print(f"[DB] add_document_record ERROR: {e}")
        raise

def get_session_documents(session_id: str) -> List[Dict[str, Any]]:
    """Return all documents ingested for a session, newest first."""
    print(f"[DB] get_session_documents: session={session_id[:8]}...")
    try:
        response = (
            supabase.table("session_documents")
            .select("*")
            .eq("session_id", session_id)
            .order("ingested_at", desc=False)
            .execute()
        )
        docs = response.data
        print(f"[DB] get_session_documents: {len(docs)} document(s).")
        return docs
    except Exception as e:
        print(f"[DB] get_session_documents ERROR: {e}")
        return []

def delete_document_record(doc_id: str) -> None:
    """Remove a document metadata record. (KG data in LightRAG is not deleted — re-ingest to overwrite.)"""
    print(f"[DB] delete_document_record: id={doc_id}")
    try:
        supabase.table("session_documents").delete().eq("id", doc_id).execute()
        print(f"[DB] delete_document_record: OK")
    except Exception as e:
        print(f"[DB] delete_document_record ERROR: {e}")
        raise

# ─────────────────────────────────────────────────────────────────────────────
#  Cloud Workspace Backup (Supabase Storage)
# ─────────────────────────────────────────────────────────────────────────────

def upload_workspace_file(workspace_id: str, local_path: str) -> bool:
    """Upload a specific workspace file to Supabase Storage."""
    filename = os.path.basename(local_path)
    storage_path = f"{workspace_id}/{filename}"
    print(f"[DB] upload_workspace_file: {storage_path}...")
    try:
        with open(local_path, "rb") as f:
            supabase.storage.from_("workspaces").upload(
                path=storage_path,
                file=f,
                file_options={"upsert": "true"}
            )
        print(f"[DB] upload_workspace_file: OK")
        return True
    except Exception as e:
        print(f"[DB] upload_workspace_file ERROR: {e}")
        return False
