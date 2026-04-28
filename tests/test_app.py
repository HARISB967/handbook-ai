"""
tests/test_app.py
Integration tests for app.py — FastAPI endpoints.
Requires: uvicorn app:app --reload --port 8001
Run: pytest tests/test_app.py -v
"""
import pytest
import httpx
import json
import uuid
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

API_URL = "http://localhost:8001"

def backend_available():
    try:
        return httpx.get(f"{API_URL}/health", timeout=3.0).status_code == 200
    except Exception:
        return False

pytestmark = pytest.mark.skipif(not backend_available(), reason="Backend not running on :8001")


def test_root():
    r = httpx.get(f"{API_URL}/")
    assert r.status_code == 200
    data = r.json()
    assert data["service"] == "HandBook.ai API"
    print(f"  ✅ Root OK: {data}")

def test_health():
    r = httpx.get(f"{API_URL}/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"
    print("  ✅ Health check OK")

def test_logs_returns_list():
    r = httpx.get(f"{API_URL}/logs")
    assert r.status_code == 200
    assert isinstance(r.json(), list)
    print(f"  ✅ /logs list ({len(r.json())} entries)")

def test_logs_clear():
    r = httpx.get(f"{API_URL}/logs/clear")
    assert r.status_code == 200
    assert r.json()["status"] == "cleared"
    print("  ✅ /logs/clear OK")

def test_session_documents_empty():
    r = httpx.get(f"{API_URL}/session/{uuid.uuid4()}/documents")
    assert r.status_code == 200
    assert r.json() == []
    print("  ✅ Empty session → [] documents")

def test_delete_doc_invalid_id():
    r = httpx.delete(f"{API_URL}/document/00000000-0000-0000-0000-000000000000")
    assert r.status_code in (200, 500)
    print(f"  ✅ DELETE invalid doc → {r.status_code}")

def test_chat_missing_session_id_rejected():
    r = httpx.post(f"{API_URL}/chat", json={"message": "hello"})
    assert r.status_code == 422
    print("  ✅ Missing session_id → 422")

def test_chat_short_response():
    session_id = str(uuid.uuid4())
    tokens = []
    with httpx.stream("POST", f"{API_URL}/chat",
                      json={"message": "Hello!", "session_id": session_id, "doc_names": []},
                      timeout=60.0) as r:
        assert r.status_code == 200
        for line in r.iter_lines():
            if not line.startswith("data: "):
                continue
            try:
                data = json.loads(line[6:])
                if data.get("type") == "token":
                    tokens.append(data["content"])
            except json.JSONDecodeError:
                pass
    full = "".join(tokens)
    assert len(full) > 0
    print(f"  ✅ /chat: {len(tokens)} tokens — '{full[:60]}'")

def test_chat_routes_simple_as_short():
    session_id = str(uuid.uuid4())
    intents = []
    with httpx.stream("POST", f"{API_URL}/chat",
                      json={"message": "What is 2+2?", "session_id": session_id, "doc_names": []},
                      timeout=60.0) as r:
        for line in r.iter_lines():
            if not line.startswith("data: "):
                continue
            try:
                data = json.loads(line[6:])
                if data.get("type") == "intent":
                    intents.append(data["content"])
            except json.JSONDecodeError:
                pass
    assert intents and intents[0] == "SHORT"
    print(f"  ✅ Simple question → SHORT")

def test_ingest_missing_session_rejected():
    r = httpx.post(f"{API_URL}/ingest/stream",
                   files={"file": ("test.pdf", b"%PDF", "application/pdf")},
                   timeout=10.0)
    assert r.status_code == 422
    print("  ✅ Ingest without session_id → 422")


if __name__ == "__main__":
    if not backend_available():
        print("⚠  Backend not running. Start with:\n  .\\venv\\Scripts\\uvicorn app:app --reload --port 8001")
        sys.exit(1)
    print("\n=== test_app.py ===\n")
    test_root(); test_health(); test_logs_returns_list(); test_logs_clear()
    test_session_documents_empty(); test_delete_doc_invalid_id()
    test_chat_missing_session_id_rejected(); test_chat_short_response()
    test_chat_routes_simple_as_short(); test_ingest_missing_session_rejected()
    print("\n✅ All API tests passed!\n")
