"""
tests/test_database.py
Unit tests for database.py — Supabase CRUD operations.
Run: pytest tests/test_database.py -v
"""
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from database import (
    create_session, list_sessions, rename_session, delete_session,
    add_chat_history, get_chat_history,
    add_document_record, get_session_documents, delete_document_record,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
_created_session_ids = []


def make_session(name="__test_session__"):
    s = create_session(name)
    _created_session_ids.append(s["id"])
    return s


def cleanup():
    for sid in _created_session_ids:
        try:
            delete_session(sid)
        except Exception:
            pass
    _created_session_ids.clear()


# ─────────────────────────────────────────────────────────────────────────────
# Session CRUD
# ─────────────────────────────────────────────────────────────────────────────
def test_create_session():
    s = make_session("Test Create")
    assert "id" in s
    assert s["name"] == "Test Create"
    print(f"  ✅ Created session: {s['id'][:8]}...")
    cleanup()


def test_list_sessions_includes_new():
    s = make_session("Test List")
    sessions = list_sessions()
    ids = [x["id"] for x in sessions]
    assert s["id"] in ids, "Newly created session not found in list"
    print(f"  ✅ Session appears in list ({len(sessions)} total)")
    cleanup()


def test_rename_session():
    s = make_session("Before Rename")
    rename_session(s["id"], "After Rename")
    sessions = list_sessions()
    match = next((x for x in sessions if x["id"] == s["id"]), None)
    assert match is not None
    assert match["name"] == "After Rename"
    print(f"  ✅ Renamed to 'After Rename'")
    cleanup()


def test_delete_session():
    s = make_session("To Delete")
    sid = s["id"]
    delete_session(sid)
    sessions = list_sessions()
    ids = [x["id"] for x in sessions]
    assert sid not in ids, "Session still present after deletion"
    print(f"  ✅ Deleted session {sid[:8]}...")


# ─────────────────────────────────────────────────────────────────────────────
# Chat History
# ─────────────────────────────────────────────────────────────────────────────
def test_add_and_get_chat_history():
    s = make_session("Chat History Test")
    sid = s["id"]
    add_chat_history("user",      "Hello from test", session_id=sid)
    add_chat_history("assistant", "Hi back!",        session_id=sid)
    history = get_chat_history(session_id=sid)
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"
    print(f"  ✅ 2 messages stored and retrieved for session {sid[:8]}...")
    cleanup()


def test_chat_history_isolation():
    s1 = make_session("Session 1")
    s2 = make_session("Session 2")
    add_chat_history("user", "Message for S1", session_id=s1["id"])
    add_chat_history("user", "Message for S2", session_id=s2["id"])
    h1 = get_chat_history(session_id=s1["id"])
    h2 = get_chat_history(session_id=s2["id"])
    assert len(h1) == 1 and "S1" in h1[0]["content"]
    assert len(h2) == 1 and "S2" in h2[0]["content"]
    print(f"  ✅ Chat history is isolated between sessions")
    cleanup()


# ─────────────────────────────────────────────────────────────────────────────
# Document Records
# ─────────────────────────────────────────────────────────────────────────────
def test_add_and_get_document_record():
    s = make_session("Doc Record Test")
    sid = s["id"]
    doc = add_document_record(
        session_id=sid,
        workspace_id=sid,
        filename="test_paper.pdf",
        file_size_bytes=102400,
        word_count=5000,
        page_count=12,
        status="success",
    )
    assert doc["id"] is not None
    docs = get_session_documents(sid)
    assert len(docs) == 1
    assert docs[0]["filename"] == "test_paper.pdf"
    assert docs[0]["word_count"] == 5000
    print(f"  ✅ Document record created and retrieved (id={doc['id'][:8]}...)")
    cleanup()


def test_delete_document_record():
    s = make_session("Del Doc Test")
    sid = s["id"]
    doc = add_document_record(sid, sid, "del_me.pdf", status="success")
    delete_document_record(doc["id"])
    docs = get_session_documents(sid)
    assert all(d["id"] != doc["id"] for d in docs)
    print(f"  ✅ Document record deleted")
    cleanup()


def test_document_records_isolated_by_session():
    s1 = make_session("DocIso1")
    s2 = make_session("DocIso2")
    add_document_record(s1["id"], s1["id"], "doc_s1.pdf")
    add_document_record(s2["id"], s2["id"], "doc_s2.pdf")
    d1 = get_session_documents(s1["id"])
    d2 = get_session_documents(s2["id"])
    assert all(d["session_id"] == s1["id"] for d in d1)
    assert all(d["session_id"] == s2["id"] for d in d2)
    print(f"  ✅ Documents are isolated by session_id")
    cleanup()


if __name__ == "__main__":
    print("\n=== test_database.py ===\n")
    test_create_session()
    test_list_sessions_includes_new()
    test_rename_session()
    test_delete_session()
    test_add_and_get_chat_history()
    test_chat_history_isolation()
    test_add_and_get_document_record()
    test_delete_document_record()
    test_document_records_isolated_by_session()
    print("\n✅ All database tests passed!\n")
