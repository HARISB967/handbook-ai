import os
import time
from openai import AsyncOpenAI

_GROQ_BASE = "https://api.groq.com/openai/v1"
_GROQ_KEY  = os.getenv("GROQ_API_KEY")

ROUTER_CLIENT = AsyncOpenAI(base_url=_GROQ_BASE, api_key=_GROQ_KEY)
ROUTER_MODEL  = "llama-3.1-8b-instant"

ROUTER_PROMPT = """
Classify this user message as SHORT or LONG.

SHORT = any normal chat — greetings, questions, explanations, summaries, comparisons.
LONG  = ONLY when user explicitly asks for a multi-page document, handbook, or essay with a high word count.

Default to SHORT when unsure.

User message: {question}
"""

async def classify_intent(question: str) -> str:
    """
    Use Llama-3.1-8B to classify the intent — SHORT or LONG.
    8B model responds in ~1-2s.
    """
    print(f"[ROUTER] Classifying intent via {ROUTER_MODEL}...")
    t0 = time.time()
    try:
        response = await ROUTER_CLIENT.chat.completions.create(
            model=ROUTER_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict routing classifier. "
                        "Reply with ONLY the single word SHORT or LONG — nothing else. "
                        "No punctuation, no explanation."
                    )
                },
                {"role": "user", "content": ROUTER_PROMPT.format(question=question)}
            ],
            temperature=0.0,
            max_tokens=5
        )
        raw = response.choices[0].message.content.strip().upper()

        if "LONG" in raw:
            intent = "LONG"
        else:
            intent = "SHORT"
            
        print(f"[ROUTER] {ROUTER_MODEL} → '{raw}' → {intent} ({time.time()-t0:.1f}s)")
        return intent
    except Exception as e:
        print(f"[ROUTER] ERROR: {e} — falling back to SHORT")
        return "SHORT"
