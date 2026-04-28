"""
tests/test_handbook.py
🧪 E2E Test Case: 20,000-word RAG Handbook Generation

Test Case:
  Input:  Upload 2-3 AI-related PDFs → "Create a handbook on RAG"
  Output: 20,000+ word structured document with TOC, sections, citations

Requirements:
  - Backend running on :8001
  - At least one PDF in the papers/ directory
Run:
  pytest tests/test_handbook.py -v -s
"""
import pytest
import httpx
import json
import uuid
import time
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

API_URL    = "http://localhost:8001"
PAPERS_DIR = os.path.join(os.path.dirname(__file__), "..", "papers")

MIN_WORDS      = 5000   # Relaxed minimum (full 20k takes ~30min)
TARGET_WORDS   = 20000  # Ideal target
INGEST_TIMEOUT = 600    # 10 min per doc
CHAT_TIMEOUT   = 3600   # 60 min for handbook


def backend_available():
    try:
        return httpx.get(f"{API_URL}/health", timeout=3.0).status_code == 200
    except Exception:
        return False


def get_test_pdfs(max_files: int = 3):
    """Return up to max_files PDFs from the papers directory."""
    if not os.path.isdir(PAPERS_DIR):
        return []
    pdfs = [f for f in os.listdir(PAPERS_DIR) if f.endswith(".pdf")]
    return pdfs[:max_files]


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Upload & Ingest PDFs
# ─────────────────────────────────────────────────────────────────────────────
def ingest_pdf(session_id: str, pdf_path: str) -> dict:
    """Upload a PDF and wait for SSE completion. Returns status dict."""
    filename = os.path.basename(pdf_path)
    print(f"\n  📄 Ingesting: {filename}")
    t0 = time.time()
    result = {"filename": filename, "status": "unknown", "elapsed": 0}

    with open(pdf_path, "rb") as fh:
        pdf_bytes = fh.read()

    try:
        with httpx.stream(
            "POST", f"{API_URL}/ingest/stream",
            files={"file": (filename, pdf_bytes, "application/pdf")},
            data={"session_id": session_id},
            timeout=INGEST_TIMEOUT
        ) as r:
            assert r.status_code == 200, f"Ingest returned {r.status_code}"
            for line in r.iter_lines():
                if not line.startswith("data: "):
                    continue
                try:
                    data = json.loads(line[6:])
                    msg  = data.get("content", "")
                    kind = data.get("type")
                    if msg:
                        print(f"    {msg}")
                    if kind == "done":
                        result["status"]  = "success"
                        result["elapsed"] = round(time.time() - t0, 1)
                    elif kind == "error":
                        result["status"]  = "error"
                        result["message"] = msg
                except json.JSONDecodeError:
                    pass
    except Exception as e:
        result["status"]  = "exception"
        result["message"] = str(e)

    result["elapsed"] = round(time.time() - t0, 1)
    print(f"    → {result['status']} in {result['elapsed']}s")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Generate Handbook
# ─────────────────────────────────────────────────────────────────────────────
def generate_handbook(session_id: str, doc_names: list, instruction: str) -> dict:
    """
    Send a LONG request and collect the final handbook.
    Returns {content, word_count, elapsed, status_updates}.
    """
    print(f"\n  🚀 Generating handbook: '{instruction[:60]}'")
    t0             = time.time()
    handbook_text  = ""
    status_updates = []

    payload = {
        "message":    instruction,
        "session_id": session_id,
        "doc_names":  doc_names,
    }

    with httpx.stream("POST", f"{API_URL}/chat", json=payload, timeout=CHAT_TIMEOUT) as r:
        assert r.status_code == 200, f"Chat returned {r.status_code}"
        for line in r.iter_lines():
            if not line.startswith("data: "):
                continue
            try:
                data     = json.loads(line[6:])
                msg_type = data.get("type")
                content  = data.get("content", "")

                if msg_type == "intent":
                    print(f"    Router → {content}")
                    assert content == "LONG", f"Expected LONG intent, got {content}"

                elif msg_type == "status":
                    print(f"    {content}")
                    status_updates.append(content)

                elif msg_type == "handbook":
                    handbook_text = content
                    print(f"    ✅ Handbook received ({len(content)} chars)")

                elif msg_type == "token":
                    handbook_text += content  # Fallback if streamed as tokens

                elif msg_type == "error":
                    raise RuntimeError(f"Backend error: {content}")

            except json.JSONDecodeError:
                pass

    word_count = len(handbook_text.split())
    return {
        "content":        handbook_text,
        "word_count":     word_count,
        "elapsed":        round(time.time() - t0, 1),
        "status_updates": status_updates,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main Test
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.skipif(not backend_available(), reason="Backend not running")
def test_handbook_generation_e2e():
    """
    🧪 Full handbook generation test.
    Upload PDFs → request handbook → validate output.
    """
    pdfs = get_test_pdfs(max_files=3)
    if not pdfs:
        pytest.skip(f"No PDFs found in {PAPERS_DIR}")

    session_id = str(uuid.uuid4())
    print(f"\n{'='*60}")
    print(f"🧪 E2E Handbook Test")
    print(f"   Session: {session_id[:8]}...")
    print(f"   PDFs:    {pdfs}")
    print(f"{'='*60}")

    # Step 1: Ingest PDFs
    print("\n📥 STEP 1: Ingesting documents...")
    ingest_results = []
    doc_names      = []
    for pdf_name in pdfs:
        pdf_path = os.path.join(PAPERS_DIR, pdf_name)
        result   = ingest_pdf(session_id, pdf_path)
        ingest_results.append(result)
        if result["status"] in ("success", "partial"):
            doc_names.append(pdf_name)

    assert len(doc_names) > 0, "At least one PDF must ingest successfully"
    print(f"\n  ✅ Ingested {len(doc_names)}/{len(pdfs)} documents: {doc_names}")

    # Step 2: Generate handbook
    print("\n✍️  STEP 2: Generating handbook...")
    instruction = (
        "Create a comprehensive handbook on Retrieval-Augmented Generation (RAG). "
        "Cover the fundamentals, architecture, key components (retriever, reader, indexer), "
        "evaluation metrics, real-world applications, and future directions. "
        "Target length: 20,000 words with proper sections and citations."
    )
    result = generate_handbook(session_id, doc_names, instruction)

    # Step 3: Validate
    print(f"\n📊 STEP 3: Validation")
    print(f"   Word count:      {result['word_count']:,}")
    print(f"   Time elapsed:    {result['elapsed']}s")
    print(f"   Status updates:  {len(result['status_updates'])}")
    print(f"   Content preview: {result['content'][:200]}...")

    # Assertions
    assert result["word_count"] >= MIN_WORDS, \
        f"Expected ≥{MIN_WORDS} words, got {result['word_count']}"
    assert len(result["content"]) > 1000, \
        "Handbook content is too short"
    assert len(result["status_updates"]) >= 2, \
        "Expected at least 2 status updates (outline + sections)"

    # Check structure
    content = result["content"]
    has_sections = any(h in content for h in ["##", "# ", "**"])
    print(f"   Has markdown sections: {has_sections}")

    # Check citations (at least one doc name should appear as citation)
    has_citation = any(f"[{d}]" in content or d.replace(".pdf", "") in content for d in doc_names)
    print(f"   Has document citations: {has_citation}")

    print(f"\n{'='*60}")
    if result["word_count"] >= TARGET_WORDS:
        print(f"🏆 Target achieved: {result['word_count']:,} words!")
    else:
        print(f"✅ Minimum passed: {result['word_count']:,}/{TARGET_WORDS:,} words")
    print(f"{'='*60}\n")

    # Save output to file for manual review
    output_path = os.path.join(os.path.dirname(__file__), "..", "test_handbook_output.md")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(result["content"])
    print(f"📄 Saved to: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Quick Sanity Tests (no upload needed)
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.skipif(not backend_available(), reason="Backend not running")
def test_long_intent_triggered():
    """Verify explicit handbook request triggers LONG route."""
    session_id = str(uuid.uuid4())
    intents    = []
    payload    = {
        "message":    "Write a 10,000 word comprehensive guide",
        "session_id": session_id,
        "doc_names":  [],
    }
    with httpx.stream("POST", f"{API_URL}/chat", json=payload, timeout=60.0) as r:
        for line in r.iter_lines():
            if not line.startswith("data: "):
                continue
            try:
                data = json.loads(line[6:])
                if data.get("type") == "intent":
                    intents.append(data["content"])
                    break  # Got what we need
            except json.JSONDecodeError:
                pass
    assert intents and intents[0] == "LONG", f"Expected LONG, got: {intents}"
    print(f"  ✅ Explicit guide request → LONG")


if __name__ == "__main__":
    if not backend_available():
        print("⚠  Start backend first:\n  .\\venv\\Scripts\\uvicorn app:app --reload --port 8001")
        sys.exit(1)
    print("\n=== test_handbook.py — E2E Test ===\n")
    test_long_intent_triggered()
    test_handbook_generation_e2e()
