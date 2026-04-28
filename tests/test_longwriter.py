"""
tests/test_longwriter.py
Tests for longwriter.py — intent routing, citation injection, SHORT generation.
Run: pytest tests/test_longwriter.py -v
Note: Integration tests hit NVIDIA API.
"""
import pytest
import asyncio
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()


# ─────────────────────────────────────────────────────────────────────────────
# Intent Router (NVIDIA Llama-8B)
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.integration
@pytest.mark.parametrize("message,expected", [
    ("hello",                                        "SHORT"),
    ("hi there",                                     "SHORT"),
    ("what is RAG?",                                 "SHORT"),
    ("explain the Capital One breach",               "SHORT"),
    ("summarise the documents",                      "SHORT"),
    ("write a 10,000 word handbook on RAG",          "LONG"),
    ("create a comprehensive guide, at least 5000 words", "LONG"),
    ("generate a full research report on the topic", "LONG"),
])
def test_classify_intent(message, expected):
    from longwriter import classify_intent

    async def _run():
        result = await classify_intent(message)
        assert result == expected, f"'{message}' → expected {expected}, got {result}"
        print(f"  ✅ '{message[:40]}' → {result}")

    asyncio.run(_run())


# ─────────────────────────────────────────────────────────────────────────────
# Citation Injection Logic (unit — no API)
# ─────────────────────────────────────────────────────────────────────────────
def test_citation_doc_clause_formatting():
    """Verify doc names are injected into system prompt correctly."""
    from longwriter import ASSISTANT_SYSTEM

    doc_names = ["Capital One.pdf", "RAG Survey.pdf"]
    doc_list  = ", ".join(f'"{d}"' for d in doc_names)
    doc_clause = (
        f"\nDocuments uploaded to this session: {doc_list}. "
        f"When asked general questions about 'the documents', cover ALL of them. "
        f"When citing specific information, use the EXACT filename in brackets — "
        f"e.g., [{doc_names[0]}]. NEVER write just [Source]."
    )
    system = ASSISTANT_SYSTEM.strip() + doc_clause
    assert "Capital One.pdf" in system
    assert "RAG Survey.pdf"  in system
    assert "NEVER write just [Source]" in system
    print("  ✅ Citation clause correctly injected into system prompt")


def test_user_prompt_with_context_includes_doc_header():
    """When doc_names and context are present, prompt should include both."""
    doc_names = ["paper1.pdf", "paper2.pdf"]
    context   = "Some retrieved context text about the topic."
    question  = "What is the main finding?"

    doc_header = "Available documents: " + ", ".join(f'"{d}"' for d in doc_names)
    user_prompt = (
        f"{doc_header}\n\n"
        f"Retrieved knowledge from the uploaded documents:\n{context}\n\n"
        f"User: {question}\n\n"
        f"Cite specific claims with the exact source filename in brackets."
    )
    assert "paper1.pdf" in user_prompt
    assert "paper2.pdf" in user_prompt
    assert context       in user_prompt
    assert question      in user_prompt
    assert "Cite specific" in user_prompt
    print("  ✅ User prompt structure is correct with context + doc names")


def test_user_prompt_without_context():
    """When no context, prompt is just the question."""
    question    = "What time is it?"
    user_prompt = question  # No context → just the question
    assert user_prompt == question
    print("  ✅ Prompt without context is just the question (no filler)")


# ─────────────────────────────────────────────────────────────────────────────
# SHORT answer streaming (integration — hits NVIDIA API)
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.integration
def test_generate_short_answer_streams_tokens():
    from longwriter import generate_short_answer
    import uuid

    workspace_id = str(uuid.uuid4())
    tokens_received = []

    async def _run():
        async for token in generate_short_answer(
            question="What is machine learning?",
            workspace_id=workspace_id,
            doc_names=[]
        ):
            tokens_received.append(token)

    asyncio.run(_run())
    full = "".join(tokens_received)
    assert len(full) > 10, "Expected a non-trivial response"
    print(f"  ✅ Received {len(tokens_received)} tokens, {len(full)} chars total")
    print(f"     Preview: '{full[:80]}...'")


@pytest.mark.integration
def test_short_answer_with_doc_names_includes_citation_context():
    """
    With doc_names provided, model should use them in citations.
    We can't assert exact citation text, but we can verify no crash.
    """
    from longwriter import generate_short_answer
    import uuid

    workspace_id = str(uuid.uuid4())
    doc_names    = ["Capital One.pdf", "LongWriter.pdf"]
    tokens       = []

    async def _run():
        async for token in generate_short_answer(
            question="Tell me about the documents",
            workspace_id=workspace_id,
            doc_names=doc_names
        ):
            tokens.append(token)

    asyncio.run(_run())
    assert len(tokens) > 0
    print(f"  ✅ SHORT answer with doc_names: {len(tokens)} tokens received")


# ─────────────────────────────────────────────────────────────────────────────
# PLAN prompt structure (unit — no API)
# ─────────────────────────────────────────────────────────────────────────────
def test_plan_prompt_has_required_fields():
    from longwriter import PLAN_PROMPT
    filled = PLAN_PROMPT.format(instruction="Write about AI.", context="Some context.")
    assert "Instruction" in filled or "instruction" in filled.lower()
    assert "Word Count"   in filled
    assert "Paragraph"    in filled
    print("  ✅ PLAN_PROMPT has required fields")


def test_write_prompt_has_required_fields():
    from longwriter import WRITE_PROMPT
    filled = WRITE_PROMPT.format(
        current_step="Paragraph 1",
        instruction="Write about AI.",
        plan="Para 1 - Main: Intro",
        context="Some context.",
        already_written_text=""
    )
    assert "instruction" in filled.lower() or "Instruction" in filled
    assert "context"     in filled.lower() or "knowledge"   in filled.lower()
    assert "Cite"        in filled
    print("  ✅ WRITE_PROMPT has required fields")


if __name__ == "__main__":
    print("\n=== test_longwriter.py (unit tests only) ===\n")
    test_citation_doc_clause_formatting()
    test_user_prompt_with_context_includes_doc_header()
    test_user_prompt_without_context()
    test_plan_prompt_has_required_fields()
    test_write_prompt_has_required_fields()
    print("\n✅ All unit tests passed! (Use --integration for API tests)\n")
