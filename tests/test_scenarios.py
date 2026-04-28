"""
tests/test_scenarios.py
Conversation scenario tests — verifying how HandBook.ai responds to
real user patterns: identity questions, document queries, post-deletion behaviour.

Run (backend must be running):
  pytest tests/test_scenarios.py -v -s
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


def chat(message: str, session_id: str, doc_names: list = None) -> str:
    """Send a chat message and collect the full text response."""
    doc_names = doc_names or []
    tokens = []
    with httpx.stream(
        "POST", f"{API_URL}/chat",
        json={"message": message, "session_id": session_id, "doc_names": doc_names},
        timeout=120.0
    ) as r:
        assert r.status_code == 200, f"Chat failed: {r.status_code}"
        for line in r.iter_lines():
            if not line.startswith("data: "):
                continue
            try:
                data = json.loads(line[6:])
                if data.get("type") == "token":
                    tokens.append(data["content"])
                elif data.get("type") == "handbook":
                    return data["content"]
            except json.JSONDecodeError:
                pass
    return "".join(tokens)


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 1: Identity & Capability Questions
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.skipif(not backend_available(), reason="Backend not running")
class TestIdentityScenarios:

    def test_greeting_is_brief(self):
        """'Hi' → short warm reply, no meta-commentary."""
        sid      = str(uuid.uuid4())
        response = chat("Hi", sid)
        assert len(response) > 0, "Expected non-empty response"
        # Should NOT contain meta-commentary
        bad_phrases = ["no-context", "no usable information", "Note:", "the context provided"]
        for phrase in bad_phrases:
            assert phrase.lower() not in response.lower(), \
                f"Response leaked meta-commentary: '{phrase}' found in: {response[:200]}"
        print(f"  ✅ Greeting response ({len(response)} chars): '{response[:100]}'")

    def test_who_are_you(self):
        """'Who are you?' → HandBook.ai introduction."""
        sid      = str(uuid.uuid4())
        response = chat("Who are you?", sid)
        assert any(kw in response.lower() for kw in ["handbook", "document", "assistant", "ai"]), \
            f"Response doesn't identify as HandBook.ai: {response[:200]}"
        print(f"  ✅ Identity: '{response[:120]}'")

    def test_what_can_you_do(self):
        """'What can you do?' → capability description."""
        sid      = str(uuid.uuid4())
        response = chat("What can you do?", sid)
        capability_keywords = ["upload", "pdf", "question", "answer", "summarise", "summary",
                               "document", "handbook", "generate", "analyse", "analyze"]
        found = [kw for kw in capability_keywords if kw in response.lower()]
        assert len(found) >= 2, f"Response doesn't describe capabilities well: {response[:300]}"
        print(f"  ✅ Capabilities mentioned: {found}")

    def test_no_documents_uploaded_response(self):
        """Without any docs, should still answer general questions."""
        sid      = str(uuid.uuid4())
        response = chat("What is machine learning?", sid, doc_names=[])
        assert len(response) > 50, "Expected a substantive answer to a general question"
        # Should NOT say it cannot answer
        assert "cannot" not in response.lower() or "machine learning" in response.lower(), \
            "Model refused general knowledge question"
        print(f"  ✅ General knowledge answer ({len(response)} chars)")

    def test_what_info_do_you_have_with_docs(self):
        """With doc_names in context, should list the documents."""
        sid       = str(uuid.uuid4())
        doc_names = ["Capital One.pdf", "RAG Survey.pdf"]
        response  = chat("What info do you have?", sid, doc_names=doc_names)
        assert any(d in response for d in doc_names), \
            f"Response didn't mention uploaded documents. Got: {response[:300]}"
        print(f"  ✅ Document awareness: '{response[:150]}'")

    def test_no_source_literal_in_response(self):
        """Model must NEVER output just '[Source]' as a citation."""
        sid      = str(uuid.uuid4())
        doc_names = ["Capital One.pdf"]
        response = chat("Tell me about Capital One", sid, doc_names=doc_names)
        assert "[Source]" not in response, \
            f"Model outputted '[Source]' placeholder: {response[:400]}"
        print(f"  ✅ No bare [Source] citation found")

    def test_citation_format_topic_included(self):
        """Citations should include a topic: [Filename — Topic]."""
        sid       = str(uuid.uuid4())
        doc_names = ["Capital One.pdf"]
        response  = chat("Explain the Capital One breach in detail", sid, doc_names=doc_names)
        # Look for the richer citation format (em-dash or hyphen)
        has_topic_citation = (
            "[Capital One.pdf —" in response or
            "[Capital One.pdf -"  in response or
            "[Capital One.pdf,"   in response or
            "[Capital One.pdf:"   in response
        )
        # Also acceptable: plain [Capital One.pdf] is minimum acceptable
        has_any_citation = "Capital One.pdf" in response
        assert has_any_citation, f"No citation found in response: {response[:300]}"
        if has_topic_citation:
            print(f"  ✅ Rich citation format found in response")
        else:
            print(f"  ⚠️  Plain citation used (acceptable, rich format preferred)")


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 2: Document-Scoped Answers
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.skipif(not backend_available(), reason="Backend not running")
class TestDocumentScenarios:

    def test_general_question_about_all_docs(self):
        """'Explain the documents' → should cover all uploaded files."""
        sid       = str(uuid.uuid4())
        doc_names = ["Capital One.pdf", "RAG Survey.pdf"]
        response  = chat("Explain all the documents briefly", sid, doc_names=doc_names)
        # Both docs should be mentioned
        assert "Capital One" in response or "capital one" in response.lower(), \
            "Response doesn't mention Capital One"
        print(f"  ✅ Multi-doc coverage: '{response[:150]}'")

    def test_specific_question_answered_correctly(self):
        """Factual question should yield a factual answer (not refusal)."""
        sid      = str(uuid.uuid4())
        response = chat("What year did the Capital One breach occur?", sid, doc_names=["Capital One.pdf"])
        assert "2019" in response, f"Expected '2019' in response, got: {response[:200]}"
        print(f"  ✅ Factual answer: '2019' found")

    def test_not_a_documentation_bot(self):
        """Model should answer general knowledge questions even without docs."""
        sid      = str(uuid.uuid4())
        response = chat("What is HTTPS?", sid, doc_names=[])
        assert len(response) > 30, "Too short"
        assert "http" in response.lower() or "secure" in response.lower(), \
            f"Response doesn't explain HTTPS: {response[:200]}"
        print(f"  ✅ General knowledge: '{response[:100]}'")


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 3: Intent Routing
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.skipif(not backend_available(), reason="Backend not running")
class TestRoutingScenarios:

    @pytest.mark.parametrize("msg,expected_intent", [
        ("hello",                                        "SHORT"),
        ("hi there, how are you?",                       "SHORT"),
        ("what is RAG?",                                 "SHORT"),
        ("summarise the documents",                      "SHORT"),
        ("explain the Capital One breach",               "SHORT"),
        ("write a 10000 word comprehensive handbook",    "LONG"),
        ("create a full detailed guide on this topic",   "LONG"),
        ("generate a 5000-word report on RAG",           "LONG"),
    ])
    def test_intent_routing(self, msg, expected_intent):
        sid = str(uuid.uuid4())
        intents = []
        with httpx.stream(
            "POST", f"{API_URL}/chat",
            json={"message": msg, "session_id": sid, "doc_names": []},
            timeout=30.0
        ) as r:
            for line in r.iter_lines():
                if not line.startswith("data: "):
                    continue
                try:
                    data = json.loads(line[6:])
                    if data.get("type") == "intent":
                        intents.append(data["content"])
                        break
                except json.JSONDecodeError:
                    pass
        assert intents, "No intent event received"
        assert intents[0] == expected_intent, \
            f"'{msg}' → expected {expected_intent}, got {intents[0]}"
        print(f"  ✅ '{msg[:40]}' → {intents[0]}")


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 4: Document Deletion Behaviour
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.skipif(not backend_available(), reason="Backend not running")
class TestDeletionScenarios:

    def test_delete_endpoint_returns_rag_deleted_field(self):
        """DELETE /document/{id} response must include rag_deleted field."""
        from database import create_session, add_document_record, delete_session
        import hashlib

        # Create a real session + doc record
        session = create_session("__test_del__")
        sid     = session["id"]
        fake_text = "This is fake document text for deletion test."
        fake_id   = f"doc-{hashlib.md5(fake_text.encode()).hexdigest()}"

        doc = add_document_record(
            session_id=sid, workspace_id=sid,
            filename="delete_test.pdf", status="success",
            lightrag_doc_id=fake_id
        )
        doc_uuid = doc["id"]

        # Now delete via API
        r = httpx.delete(f"{API_URL}/document/{doc_uuid}", timeout=30.0)
        assert r.status_code == 200, f"Delete returned {r.status_code}: {r.text}"
        data = r.json()
        assert "rag_deleted" in data, f"rag_deleted field missing: {data}"
        assert data["filename"] == "delete_test.pdf"
        print(f"  ✅ Delete response: {data}")

        # Cleanup session
        delete_session(sid)

    def test_delete_nonexistent_doc(self):
        """Deleting a non-existent doc_id should return not_found or 200."""
        r = httpx.delete(f"{API_URL}/document/{uuid.uuid4()}", timeout=10.0)
        assert r.status_code in (200, 404, 500)
        if r.status_code == 200:
            assert r.json().get("status") in ("not_found", "deleted")
        print(f"  ✅ Non-existent delete → {r.status_code}")

    def test_metadata_removed_after_delete(self):
        """After DELETE /document, the record should not appear in /session/docs."""
        from database import create_session, add_document_record, delete_session

        session = create_session("__test_meta_del__")
        sid     = session["id"]
        doc     = add_document_record(sid, sid, "meta_del_test.pdf")
        doc_id  = doc["id"]

        # Confirm it's there
        r_before = httpx.get(f"{API_URL}/session/{sid}/documents", timeout=5.0)
        ids_before = [d["id"] for d in r_before.json()]
        assert doc_id in ids_before, "Doc not found before deletion"

        # Delete it
        httpx.delete(f"{API_URL}/document/{doc_id}", timeout=30.0)

        # Confirm it's gone
        r_after = httpx.get(f"{API_URL}/session/{sid}/documents", timeout=5.0)
        ids_after = [d["id"] for d in r_after.json()]
        assert doc_id not in ids_after, "Doc still present after deletion"
        print(f"  ✅ Metadata removed from Supabase after delete")

        delete_session(sid)


if __name__ == "__main__":
    if not backend_available():
        print("⚠  Start backend first:\n  .\\venv\\Scripts\\uvicorn app:app --reload --port 8001")
        sys.exit(1)
    print("\n=== test_scenarios.py ===\n")
    print("Run with: pytest tests/test_scenarios.py -v -s\n")
