"""
prompt_constructor.py
=====================
Campayn AI Assistant — Prompt Constructor

WHAT THIS FILE DOES
--------------------
Takes the outputs of all three modules and assembles them into
a single, well-structured prompt for the final LLM generation call.

Think of it as the "composer" — it knows what each module returned,
what role the user is, what the conversation history looks like,
and builds the optimal prompt from all of that.

INPUT  : outputs from Module 1, 2, 3 + role + query + history
OUTPUT : one complete prompt string ready for Gemini

PROMPT STRUCTURE (in order)
-----------------------------
1. System prompt       — role persona (brand strategist / creator coach)
2. Knowledge context   — Module 2 RAG chunks
3. Creator profiles    — Module 1 filtered profiles (brand role only)
4. Vision analysis     — Module 3 output (only if image was provided)
5. Conversation history— last N turns for memory
6. User query          — the actual question

TOKEN BUDGET MANAGEMENT
------------------------
Gemini 2.0 Flash has a 1M token context window, but we budget
conservatively — long prompts increase cost and latency.

Rough budgets per section:
  System prompt     ~300  tokens
  RAG chunks        ~800  tokens  (5 chunks × ~160 tokens each)
  Creator profiles  ~400  tokens  (10 profiles × ~40 tokens each)
  Vision analysis   ~250  tokens
  History           ~500  tokens  (last 6 turns)
  Query             ~100  tokens
  ─────────────────────────────
  Total             ~2350 tokens  — well within budget

Author : Mohit Chauhan — Campayn Multimodal AI Assistant
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# ─────────────────────────────────────────────────────────────────
# SYSTEM PROMPTS — role-specific personas for the LLM
# These are injected first so every response stays in character
# ─────────────────────────────────────────────────────────────────

BRAND_SYSTEM_PROMPT = """You are the Campayn Brand Strategist — a senior creator marketing expert \
embedded inside the Campayn platform for Indian D2C brands.

Your job: help brands plan campaigns, match creators, set CPV targets, \
write briefs, and diagnose performance — grounded in the Campayn knowledge base.

Rules:
- Always cite CPV ranges and niche context from the retrieved knowledge
- When recommending creators, explain WHY each fits — niche, tier, CPV rationale
- Never guarantee campaign outcomes — frame as probability and best practice
- If the retrieved context doesn't cover the question, say so clearly
- Be direct, specific, and strategic — no filler"""

CREATOR_SYSTEM_PROMPT = """You are the Campayn Creator Coach — a growth strategist and content \
advisor embedded inside the Campayn creator app for Indian Instagram creators.

Your job: help creators improve content quality, grow their audience, \
maximise earnings through CPV campaigns, and pitch brands effectively.

Rules:
- Give specific, actionable feedback — never generic advice
- Reference actual elements from the image analysis when available
- Be honest, not flattering — constructive criticism builds careers
- Ground advice in the Campayn knowledge base (algorithm mechanics, hooks, monetisation)
- Never promise specific follower counts or earnings numbers"""

# ─────────────────────────────────────────────────────────────────
# SECTION TEMPLATES — individual blocks assembled into the prompt
# ─────────────────────────────────────────────────────────────────

# Shown when Module 2 returns chunks
RAG_SECTION_TEMPLATE = """
━━━ CAMPAYN KNOWLEDGE BASE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{module2_block}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""".strip()

# Shown only for brand role when Module 1 returns profiles
PROFILES_SECTION_TEMPLATE = """
━━━ MATCHED CREATOR PROFILES ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{module1_block}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""".strip()

# Shown only when an image was provided
VISION_SECTION_TEMPLATE = """
━━━ IMAGE ANALYSIS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{module3_block}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""".strip()

# Shown when there is prior conversation history
HISTORY_SECTION_TEMPLATE = """
━━━ CONVERSATION HISTORY ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{history_text}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""".strip()

# Confidence-aware instruction appended to the query
LOW_CONFIDENCE_NOTE = """
Note: Retrieved context has low confidence for this query.
Answer from general Campayn principles where specific context is missing.
Be transparent about what you are inferring vs what is grounded in the knowledge base.
""".strip()


# ─────────────────────────────────────────────────────────────────
# DATA CLASS — structured input to the constructor
# ─────────────────────────────────────────────────────────────────

@dataclass
class PromptInput:
    """
    Everything the Prompt Constructor needs to build the final prompt.
    Passed in by the LangGraph node after all modules have run.

    All module blocks are pre-formatted strings from their respective
    format_for_prompt() functions. None means the module didn't run
    or returned no results.
    """
    # Core
    query:         str              # the user's question
    role:          str              # "brand" or "creator"

    # Module outputs (None = module didn't run or returned nothing useful)
    module1_block: Optional[str]   # formatted creator profiles list
    module2_block: Optional[str]   # formatted RAG knowledge chunks
    module3_block: Optional[str]   # formatted vision analysis

    # Module metadata (used for confidence-aware instructions)
    module1_meta:  Optional[dict] = None
    module2_meta:  Optional[dict] = None
    module3_meta:  Optional[dict] = None

    # Conversation history — list of {"role": "user"|"assistant", "content": str}
    history:       list[dict] = None

    # How many history turns to include (each turn = 1 user + 1 assistant)
    max_history_turns: int = 3


# ─────────────────────────────────────────────────────────────────
# HISTORY FORMATTER
# ─────────────────────────────────────────────────────────────────

def _format_history(
    history: list[dict],
    max_turns: int = 3,
) -> str:
    """
    Format conversation history into a compact readable block.

    We only include the last max_turns turns (user + assistant pairs)
    to stay within token budget. Older context is dropped — the
    current query + knowledge context is more important than old turns.

    Each turn is formatted as:
        User: <question>
        Assistant: <response>
    """
    if not history:
        return ""

    # Each "turn" is one user message + one assistant response = 2 items
    # Take the last max_turns × 2 messages
    recent = history[-(max_turns * 2):]

    lines = []
    for msg in recent:
        role_label = "User" if msg.get("role") == "user" else "Assistant"
        content = msg.get("content", "").strip()
        # Truncate very long assistant responses in history to save tokens
        if role_label == "Assistant" and len(content) > 400:
            content = content[:400] + "... [truncated]"
        lines.append(f"{role_label}: {content}")

    return "\n\n".join(lines)


# ─────────────────────────────────────────────────────────────────
# CONFIDENCE CHECKER
# ─────────────────────────────────────────────────────────────────

def _is_low_confidence(prompt_input: PromptInput) -> bool:
    """
    Returns True if the overall retrieval confidence is low.
    Used to append a transparency note to the query so the LLM
    doesn't hallucinate authoritative answers on thin context.
    """
    m2_conf = (prompt_input.module2_meta or {}).get("confidence", "high")
    m1_conf = (prompt_input.module1_meta or {}).get("confidence", "high")
    return m2_conf in ("low", "none") and m1_conf in ("low", "medium")


# ─────────────────────────────────────────────────────────────────
# MAIN CONSTRUCTOR
# ─────────────────────────────────────────────────────────────────

def build_prompt(prompt_input: PromptInput) -> str:
    """
    Assemble all sections into the final prompt string.

    Section order matters:
      1. System prompt first — establishes persona and rules
      2. Knowledge/profiles second — grounding context before the question
      3. Vision third — image-specific context
      4. History fourth — recent conversation for continuity
      5. Query last — what the LLM actually needs to answer

    This ordering follows the "context before question" principle:
    the LLM has all relevant information loaded before it sees the query,
    which produces more grounded, accurate responses.

    Args:
        prompt_input : PromptInput dataclass with all module outputs

    Returns:
        Complete prompt string ready to send to Gemini
    """
    sections: list[str] = []

    # ── 1. System prompt ──────────────────────────────────────────────────
    system = (
        BRAND_SYSTEM_PROMPT if prompt_input.role == "brand"
        else CREATOR_SYSTEM_PROMPT
    )
    sections.append(system)

    # ── 2. RAG knowledge chunks (Module 2) ────────────────────────────────
    # Always include if available — this is the primary grounding source
    if prompt_input.module2_block and prompt_input.module2_block.strip():
        sections.append(
            RAG_SECTION_TEMPLATE.format(module2_block=prompt_input.module2_block)
        )

    # ── 3. Creator profiles (Module 1) ────────────────────────────────────
    # Only inject for brand role — creators don't need a profile list
    if (
        prompt_input.role == "brand"
        and prompt_input.module1_block
        and prompt_input.module1_block.strip()
    ):
        # Check if Module 1 actually found profiles (not just a "no match" message)
        m1_matches = (prompt_input.module1_meta or {}).get("total_matches", 0)
        if m1_matches > 0:
            sections.append(
                PROFILES_SECTION_TEMPLATE.format(module1_block=prompt_input.module1_block)
            )

    # ── 4. Vision analysis (Module 3) ─────────────────────────────────────
    # Only inject if an image was analysed and analysis didn't fail
    if prompt_input.module3_block and prompt_input.module3_block.strip():
        m3_error = (prompt_input.module3_meta or {}).get("error", "")
        if not m3_error:
            sections.append(
                VISION_SECTION_TEMPLATE.format(module3_block=prompt_input.module3_block)
            )

    # ── 5. Conversation history ───────────────────────────────────────────
    history_text = _format_history(
        prompt_input.history or [],
        max_turns=prompt_input.max_history_turns,
    )
    if history_text:
        sections.append(
            HISTORY_SECTION_TEMPLATE.format(history_text=history_text)
        )

    # ── 6. User query ─────────────────────────────────────────────────────
    query_block = f"USER QUERY ({prompt_input.role.upper()}): {prompt_input.query}"

    # If context confidence is low, append a transparency note
    if _is_low_confidence(prompt_input):
        query_block += f"\n\n{LOW_CONFIDENCE_NOTE}"

    sections.append(query_block)

    # Join all sections with double newlines
    final_prompt = "\n\n".join(sections)

    # Log token estimate (rough: 1 token ≈ 4 chars for English)
    estimated_tokens = len(final_prompt) // 4
    print(f"[PromptConstructor] Built prompt: {len(sections)} sections, "
          f"~{estimated_tokens} tokens, role={prompt_input.role}")

    return final_prompt


# ─────────────────────────────────────────────────────────────────
# RESPONSE PARSER
# Parses the raw LLM response for the FastAPI / Streamlit layer
# ─────────────────────────────────────────────────────────────────

def parse_response(raw_response: str, role: str) -> dict:
    """
    Parse and clean the raw LLM response into a structured dict
    for the API response layer.

    For now this is simple cleaning — in future iterations this
    can extract structured data (recommended creators list, brief
    JSON, etc.) from the response using a second LLM pass.

    Returns:
        {
            "text":    clean response text for the UI
            "role":    "brand" | "creator"
            "sources": list of knowledge base sections referenced
        }
    """
    text = raw_response.strip()

    # Extract any Part references the LLM included (e.g. "[Part 2]")
    import re
    part_refs = re.findall(r"\[Part (\d+)\]", text)
    sources = list({f"Part {p}" for p in part_refs})

    return {
        "text":    text,
        "role":    role,
        "sources": sorted(sources),
    }