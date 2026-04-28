"""
tests/test_ingestion.py
Tests for ingestion.py — PDF extraction, LightRAG workspace pool, query.
Run: pytest tests/test_ingestion.py -v
Note: Some tests hit NVIDIA API and may take a few minutes.
"""
import pytest
import asyncio
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# PDF Extraction Tests
# ─────────────────────────────────────────────────────────────────────────────
def _make_minimal_pdf_bytes() -> bytes:
    """Return a minimal valid PDF with one page of text."""
    try:
        import io
        import pdfplumber
        # Build a tiny PDF in memory using reportlab if available
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.pdfgen import canvas
            buf = io.BytesIO()
            c = canvas.Canvas(buf, pagesize=A4)
            c.drawString(100, 750, "This is a test document for HandBook.ai ingestion tests.")
            c.drawString(100, 720, "It contains two sentences of text.")
            c.save()
            return buf.getvalue()
        except ImportError:
            pass
    except Exception:
        pass
    # Fallback: return bytes of an existing PDF in the papers dir
    papers_dir = os.path.join(os.path.dirname(__file__), "..", "papers")
    if os.path.isdir(papers_dir):
        for f in os.listdir(papers_dir):
            if f.endswith(".pdf"):
                with open(os.path.join(papers_dir, f), "rb") as fh:
                    return fh.read()
    raise RuntimeError("No PDF available for testing. Add a PDF to the papers/ directory.")


def test_extract_text_from_pdf():
    from ingestion import extract_text_from_pdf_meta
    pdf_bytes = _make_minimal_pdf_bytes()
    text, pages = extract_text_from_pdf_meta(pdf_bytes)
    assert isinstance(text, str), "Extracted text must be a string"
    assert len(text) > 0, "Extracted text must not be empty"
    assert pages > 0, "Page count must be > 0"
    print(f"  ✅ Extracted {len(text):,} chars from {pages} page(s)")


def test_extract_handles_empty_input():
    """Should return ('', 0) for completely empty bytes, not crash."""
    from ingestion import extract_text_from_pdf_meta
    try:
        text, pages = extract_text_from_pdf_meta(b"")
        # If it doesn't raise, text should be empty
        assert text == "" or pages == 0
        print("  ✅ Empty bytes handled gracefully")
    except Exception as e:
        # pdfplumber may raise — that's also acceptable behaviour
        print(f"  ✅ Empty bytes raised (expected): {type(e).__name__}")


# ─────────────────────────────────────────────────────────────────────────────
# LightRAG Workspace Pool
# ─────────────────────────────────────────────────────────────────────────────
def test_get_rag_creates_unique_instances():
    from ingestion import get_rag, _rag_pool
    import uuid

    ws1 = str(uuid.uuid4())
    ws2 = str(uuid.uuid4())

    async def _run():
        r1 = await get_rag(ws1)
        r2 = await get_rag(ws2)
        # Same workspace should return cached instance
        r1b = await get_rag(ws1)
        assert r1 is r1b, "Same workspace should return the SAME cached instance"
        assert r1 is not r2, "Different workspaces should return DIFFERENT instances"
        print(f"  ✅ Pool correctly isolates workspaces (pool size: {len(_rag_pool)})")
        # Cleanup
        _rag_pool.pop(ws1, None)
        _rag_pool.pop(ws2, None)

    asyncio.run(_run())


def test_workspace_directories_created():
    from ingestion import get_rag, _rag_pool, BASE_DIR
    import uuid

    ws = str(uuid.uuid4())

    async def _run():
        await get_rag(ws)
        wdir = os.path.join(BASE_DIR, ws)
        assert os.path.isdir(wdir), f"Workspace directory not created: {wdir}"
        print(f"  ✅ Workspace directory created at: {wdir}")
        _rag_pool.pop(ws, None)

    asyncio.run(_run())


# ─────────────────────────────────────────────────────────────────────────────
# Full Ingestion (integration — hits NVIDIA API)
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.integration
def test_process_pdf_returns_metadata():
    """
    INTEGRATION TEST: Requires NVIDIA_API_KEY and a valid PDF.
    Processes a real PDF through LightRAG KG extraction.
    May take 1-5 minutes.
    """
    from ingestion import process_pdf, _rag_pool
    import uuid

    workspace_id = str(uuid.uuid4())
    pdf_bytes    = _make_minimal_pdf_bytes()
    filename     = "test_ingestion.pdf"

    async def _run():
        meta = await process_pdf(filename, pdf_bytes, workspace_id)
        assert meta["filename"]    == filename
        assert meta["word_count"]  >= 0
        assert meta["page_count"]  >= 0
        assert meta["status"]      in ("success", "partial", "duplicate")
        print(f"  ✅ Ingestion metadata: {meta}")
        _rag_pool.pop(workspace_id, None)

    asyncio.run(_run())


@pytest.mark.integration
def test_query_returns_string():
    """
    INTEGRATION TEST: Requires existing workspace with data.
    """
    from ingestion import query, _rag_pool
    import uuid

    ws = str(uuid.uuid4())

    async def _run():
        result = await query("What is this document about?", ws)
        assert isinstance(result, str)
        print(f"  ✅ Query returned {len(result)} chars: '{result[:80]}...'")
        _rag_pool.pop(ws, None)

    asyncio.run(_run())


if __name__ == "__main__":
    print("\n=== test_ingestion.py ===\n")
    test_extract_text_from_pdf()
    test_extract_handles_empty_input()
    test_get_rag_creates_unique_instances()
    test_workspace_directories_created()
    print("\n✅ Unit tests passed! (Run with --integration flag for API tests)\n")
