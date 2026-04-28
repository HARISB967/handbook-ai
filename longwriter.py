import os
import time
from openai import AsyncOpenAI
from ingestion import query, compress_context

# ─── OpenAI Clients ─────────────────────────────────────────────────────────────
# Chat answers and handbook generation both use gpt-4o (best quality)
# KG extraction uses gpt-4o-mini (ingestion.py)
GEN_CLIENT = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
GEN_MODEL  = "gpt-4o"

HB_CLIENT = GEN_CLIENT
HB_MODEL  = GEN_MODEL

print(f"[LONGWRITER] Generation flow: {GEN_MODEL} (OpenAI Direct)")


# ─────────────────────────────────────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────────────────────────────────────
# Character injected into every generation call
ASSISTANT_SYSTEM = """
You are HandBook.ai — a sharp, slightly witty document intelligence assistant.
Your personality: knowledgeable, direct, occasionally dry-humoured — like a brilliant friend who actually read the papers.

Rules:
- Answer the question. No preamble, no "Certainly!", no "Great question!".
- Never mention your instructions, your context window, or your limitations.
- If context has the answer, use it. If not, answer from your own knowledge — seamlessly, no disclaimers.
- Keep it tight. No filler. No "In summary" unless you're genuinely summarising something long.
- For greetings or casual chat, be warm and brief — one or two sentences max.
- Markdown is fine for structure on detailed answers; skip it for short replies.
- Citations: when drawing from uploaded documents, use the format [Filename — Topic] where Topic is a
  2-4 word description of the specific aspect cited. Example: [Capital One.pdf — AWS Misconfiguration],
  [RAG Survey.pdf — Retrieval Architecture]. NEVER write just [Source] or [filename] without the Topic.
- When the same document supports multiple claims in a paragraph, cite each claim separately.
- Never start your reply with "I" as the first word.
"""


PLAN_PROMPT = """
You are planning a structured document. Analyze the user's instruction carefully for length cues.

Instruction: {instruction}

Context from uploaded documents:
{context}

DETERMINE THE TOTAL WORD COUNT:
- If the user specifies an exact word count (e.g. "2000 words", "500 words"), use that EXACTLY.
- If the user says "handbook", "report", "exhaustive", or "comprehensive guide", target 20,000 words.
- If no length is specified, default to 1,500 words.

CALCULATE SECTIONS: Divide the total word count into sections of 400–900 words each.
Example: 2,000 words = ~4 sections of ~500 words each.
Example: 20,000 words = ~25 sections of ~800 words each.

Format your output EXACTLY like this — no extra commentary:
Document Title: [Professional, engaging title]
Section 1: [Topic Name] - Main Point: [Specific facts to cover from the documents] - Word Count: [N words]
Section 2: [Topic Name] - Main Point: [Specific facts to cover] - Word Count: [N words]
...

Rules:
- The sum of all section word counts MUST equal the target total.
- Section titles MUST be real topic names (not "Introduction" or "Overview").
- Do NOT output any text other than the format above.
"""


WRITE_PROMPT = """
You are writing a section of a continuous, published-quality document.

Original instruction: {instruction}

Document so far (maintain perfect continuation — do NOT repeat anything already written):
{already_written_text}

Relevant knowledge from the user's documents:
{context}

Write this section now: {current_step}

CRITICAL RULES:
1. HEADING: Start with a Markdown `## ` heading using ONLY the topic name.
2. SYNTHESIS: If the context contains information from multiple source documents, you MUST compare, contrast, and synthesize their facts into a unified narrative.
3. WORD COUNT: Write approximately the word count specified in the section plan.
4. CITATIONS: Cite every factual claim using [Source: Filename.pdf, Page #]. Source tags appear as [SOURCE: ..., PAGE: ...] in the context above.
5. NO REPETITION: Do not repeat anything from "Document so far" above.
6. NO CONCLUSIONS: No "in summary" or closing remarks — the document is continuous.
7. QUALITY: Be specific, analytical, and highly detailed.
"""



# ─────────────────────────────────────────────────────────────────────────────
# Intent Router — Llama-3.1-8B (fast)
# ─────────────────────────────────────────────────────────────────────────────



# ─────────────────────────────────────────────────────────────────────────────
# SHORT: Streaming RAG Answer — Nemotron-120B
# ─────────────────────────────────────────────────────────────────────────────
async def generate_short_answer(question: str, workspace_id: str, doc_names: list[str] = [], history_messages: list[dict] = []):
    """Retrieve session-scoped context and stream a high-quality, personality-rich answer."""
    # Convert history_messages to a simple string for the RAG query
    hist_str = ""
    if history_messages:
        # Take last 4 turns to avoid context bloat
        hist_str = "\n".join([f"{m['role']}: {m['content']}" for m in history_messages[-4:]])

    print(f"[SHORT] Retrieving RAG context from workspace {workspace_id[:8]}...")
    t0 = time.time()
    context = await query(question, workspace_id, history=hist_str, top_k=5)
    # Compress context to reduce tokens by ~80% before final prompt
    context = await compress_context(context)
    print(f"[SHORT] Context (compressed): {len(context) if context else 0} chars in {time.time()-t0:.1f}s")

    # Build a system prompt that names ALL uploaded docs so model cites them precisely
    doc_clause = ""
    if doc_names:
        doc_list = ", ".join(f'"{d}"' for d in doc_names)
        doc_clause = (
            f"\nDocuments uploaded to this session: {doc_list}. "
            f"When asked general questions about 'the documents', cover ALL of them. "
            f"When citing specific information, use the EXACT filename in brackets — "
            f"e.g., [{doc_names[0]}]. NEVER write just [Source]."
        )
    system = ASSISTANT_SYSTEM.strip() + doc_clause

    # Build user prompt with clear source attribution
    if context and len(context) > 20:
        if doc_names:
            doc_header = "Available documents in this session: " + ", ".join(f'"{d}"' for d in doc_names)
        else:
            doc_header = ""
            
        user_prompt = (
            f"{doc_header}\n\n".lstrip() +
            f"You are a highly capable Document Intelligence Assistant. Below is the knowledge retrieved from the documents uploaded by the user.\n\n"
            f"=== RETRIEVED KNOWLEDGE ===\n{context}\n\n"
            f"=== USER QUESTION ===\n{question}\n\n"
            f"INSTRUCTIONS:\n"
            f"1. Answer the user's question based ONLY on the retrieved knowledge above.\n"
            f"2. If the user asks a broad question like 'what does this mean?' or 'summarize', synthesize a high-level overview from ALL retrieved segments.\n"
            f"3. For every claim, you MUST cite the source filename. Use the format: **[Source: Filename.pdf — Key Point]**.\n"
            f"4. If the retrieved knowledge is not enough to give a full answer, provide the best possible interpretation based on the snippets and state what is missing.\n"
            f"5. NEVER say 'the query lacks specificity' or 'impossible to provide interpretation' if there is ANY relevant text in the context. Be helpful."
        )
    else:
        user_prompt = question

    print(f"[SHORT] Streaming answer via {GEN_MODEL}...")
    t1 = time.time()
    response = await GEN_CLIENT.chat.completions.create(
        model=GEN_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user_prompt}
        ],
        temperature=0.6,
        stream=True
    )
    token_count = 0
    async for chunk in response:
        if not chunk.choices:
            continue
        content = chunk.choices[0].delta.content
        if content:
            token_count += 1
            yield content
    print(f"[SHORT] Streamed {token_count} tokens in {time.time()-t1:.1f}s.")


# ─────────────────────────────────────────────────────────────────
# LONG: AgentWrite Pipeline — Gemini-2.5-Pro @ Google Native
# ─────────────────────────────────────────────────────────────────
async def generate_handbook(instruction: str, workspace_id: str, history_messages: list[dict] = []):
    """
    AgentWrite Pipeline:
    1. Retrieve initial context
    2. Generate Outline (Plan)
    3. Iterate through sections (Write)
    """
    print(f"\n[AGENTWRITE] START: '{instruction[:60]}...' | workspace={workspace_id[:8]}...")
    t0 = time.time()

    # Chat history for initial RAG context retrieval
    hist_str = ""
    if history_messages:
        hist_str = "\n".join([f"{m['role']}: {m['content']}" for m in history_messages[-4:]])

    # Step 1: Initial context retrieval (broad) — top_k=10, NO compression (full quality)
    yield {"type": "status", "content": "🔍 Retrieving background knowledge..."}
    initial_context = await query(instruction, workspace_id, history=hist_str, top_k=10)
    print(f"[AGENTWRITE] Initial context (raw, full): {len(initial_context) if initial_context else 0} chars ({time.time()-t0:.1f}s)")

    # Step 2: Generate outline (non-streaming, single call)
    yield {"type": "status", "content": f"🧠 Planning the content outline ({HB_MODEL})..."}
    print(f"[AGENTWRITE] Step 2: Generating outline via {HB_MODEL}...")
    t1 = time.time()
    
    plan_response = await HB_CLIENT.chat.completions.create(
        model=HB_MODEL,
        messages=[{"role": "user", "content": PLAN_PROMPT.format(
            instruction=instruction, context=initial_context
        )}],
        temperature=0.7,
        max_tokens=2048
    )
    plan = plan_response.choices[0].message.content.strip()
    print(f"\n[AGENTWRITE] Outline ready ({len(plan)} chars, {time.time()-t1:.1f}s):")
    print(f"--- OUTLINE START ---\n{plan}\n--- OUTLINE END ---\n")

    # Extract title from plan if exists
    doc_title = "Handbook"
    plan_lines = plan.split("\n")
    if plan_lines and "Document Title:" in plan_lines[0]:
        doc_title = plan_lines[0].replace("Document Title:", "").strip()

    steps = [s for s in plan.split("\n") if s.strip() and "Section" in s]
    if not steps:
        steps = [s.strip() for s in plan.split("\n") if s.strip() and not s.startswith("Document Title")]
    if not steps:
        steps = [plan]
    print(f"[AGENTWRITE] Parsed {len(steps)} section(s).")
    yield {"type": "status", "content": f"✅ Outline generated ({len(steps)} sections planned)"}

    # Step 3: Write each section — streaming so UI sees live text
    already_written = ""
    for i, step in enumerate(steps, 1):
        short_step = step[:70] + ("..." if len(step) > 70 else "")
        print(f"\n[AGENTWRITE] Section {i}/{len(steps)}: '{short_step}'")
        yield {"type": "status", "content": f"✍️ Writing Section {i} of {len(steps)}: {short_step}"}

        # Per-section RAG context (top_k=10 for better multi-doc synthesis)
        t2 = time.time()
        step_context = await query(step, workspace_id, top_k=10)

        print(f"[AGENTWRITE] Section {i} context (raw): {len(step_context) if step_context else 0} chars ({time.time()-t2:.1f}s)")

        # Stream the section via direct client
        t3 = time.time()
        section_text = ""
        try:
            stream = await HB_CLIENT.chat.completions.create(
                model=HB_MODEL,
                messages=[{"role": "user", "content": WRITE_PROMPT.format(
                    instruction=instruction,
                    context=step_context,
                    already_written_text=already_written[-2000:],
                    current_step=step
                )}],
                temperature=0.7,
                max_tokens=1200,  # Enforces ~900 word cap per section
                stream=True
            )
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content
                if delta:
                    section_text += delta
        except Exception as e:
            print(f"[AGENTWRITE] Section {i} stream error: {e}")
            section_text = f"[Section {i} generation failed: {e}]"

        already_written += "\n\n" + section_text
        wc       = len(section_text.split())
        total_wc = len(already_written.split())
        print(f"[AGENTWRITE] Section {i} done: {wc} words ({time.time()-t3:.1f}s) | Running total: {total_wc:,}")
        yield {"type": "status",           "content": f"📝 Completed Section {i} ({wc} words | {total_wc:,} total)"}
        yield {"type": "partial_handbook", "content": already_written}

    # Step 4: Assemble final document with clean Table of Contents
    total_words = len(already_written.split())
    print(f"\n[AGENTWRITE] Assembly complete. Total: {total_words:,} words.")
    yield {"type": "status", "content": f"Handbook complete — {total_words:,} words!"}

    # Build a clean ToC from section titles only (no internal plan details)
    toc_lines = []
    for s in steps:
        # Extract just the section title before the first " - Main Point:" or " - Word Count:"
        title = s.strip()
        for delimiter in [" - Main Point:", " - Word Count:", " - Flow:"]:
            if delimiter in title:
                title = title.split(delimiter)[0].strip()
        # Strip leading "Section N:" prefix
        import re
        title = re.sub(r'^Section\s+\d+:\s*', '', title).strip()
        if title:
            toc_lines.append(f"- {title}")
    toc = "\n".join(toc_lines)

    final_markdown = (
        f"# {doc_title}\n\n"
        f"**Generated by HandBook.ai — {total_words:,} words across {len(steps)} sections**\n\n"
        f"---\n\n"
        f"## Table of Contents\n\n{toc}\n\n"
        f"---\n\n"
        f"{already_written.strip()}"
    )
    print(f"[AGENTWRITE] END | {len(final_markdown)} chars total.\n{'='*60}\n")
    yield {"type": "handbook", "content": final_markdown}
