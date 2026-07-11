"""
intent_parser.py
================
Campayn AI Assistant — Intent Parser (Gemini via LangChain)

WHAT THIS FILE DOES
-------------------
Converts a vague user query like "find me a creator for my protein brand"
into a structured FilterSpec that Module 1 (pandas filter) can execute.

HOW IT WORKS — 3 stages
------------------------
Stage 1: Send the query to Gemini and ask it to extract filter fields
         Returns a partial spec + a list of fields it could NOT figure out

Stage 2: For every field Gemini couldn't figure out, generate a
         multiple-choice question (MCQ) so the user can fill the gap

Stage 3: Merge Gemini's answers + user's MCQ answers into one
         complete FilterSpec → hand off to Module 1

WHY NOT REGEX?
--------------
Regex only catches words it has seen before.
"gym culture", "protein brand", "saree ke liye koi creator chahiye" — 
these all fail regex but Gemini understands them instantly.

INSTALL (pick ONE — both work):
  Recommended:  pip install langchain-google-genai
  Alternative:  pip install google-generativeai   (older SDK, still supported)

Author : Mohit Chauhan — Campayn Multimodal AI Assistant
"""

# ─────────────────────────────────────────────────────────────────
# STANDARD LIBRARY IMPORTS (no install needed)
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations      # allows type hints like list[str] in older Python

import json                             # parse JSON responses from Gemini
import os                               # read environment variables (API keys, etc.)
import re                               # used only for cleaning raw text if needed
import urllib.request                   # fallback HTTP call to Gemini REST API
import urllib.error
from dataclasses import dataclass, asdict   # lightweight data containers
from typing import Optional                 # Optional[X] means "X or None"

# ─────────────────────────────────────────────────────────────────
# MODULE 1 IMPORTS
# ─────────────────────────────────────────────────────────────────
from api.modules.profile_filter import (
    FilterSpec,         # the structured object that pandas filter reads
    RAW_TO_CANONICAL,   # maps "Food " → "food", "Lifestyle & Living" → "lifestyle" etc.
    TIER_LABELS,        # ["Nano", "Micro", "Mid", "Macro", "Mega", "Celebrity"]
)

import pandas as pd     # used to pull real option values from the CSV for MCQ questions


# ─────────────────────────────────────────────────────────────────
# GEMINI CLIENT SETUP
#
# We support TWO SDKs because different machines may have
# different packages installed:
#
#   Path A: langchain-google-genai  ← cleanest, recommended
#           pip install langchain-google-genai
#
#   Path B: google-generativeai (new-style, v0.8+)
#           pip install google-generativeai
#
#   Path C: Direct REST API via urllib (zero dependencies fallback)
#
# The code tries A → B → C in order and uses whatever works.
# ─────────────────────────────────────────────────────────────────

# Read the API key from environment — never hardcode keys in source files!
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL   = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")

# We'll store which SDK path worked so we can use it later
_SDK_MODE = None   # will be set to "langchain", "generativeai", or "rest"
_lc_llm   = None   # LangChain LLM object (used in langchain mode)


def _setup_gemini_client():
    """
    Try to initialise the Gemini client.
    Sets the global _SDK_MODE and _lc_llm variables.
    Called once when the module is first imported.
    """
    global _SDK_MODE, _lc_llm

    # ── Path A: langchain-google-genai ────────────────────────────────────
    # This is the cleanest integration — LangChain handles retries,
    # structured output parsing, and prompt templates automatically.
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI

        # temperature=0 means deterministic output (same input → same output)
        # This is important for a parser that needs consistent JSON
        _lc_llm   = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=GEMINI_API_KEY,
            temperature=0,
            # gRPC (the default transport) deadlocks under uvicorn's fork/threads
            # — "Other threads are currently calling into gRPC, skipping fork()".
            # REST avoids the gRPC channel entirely.
            transport="rest",
            # Fail fast instead of hanging on the 600s default retry window,
            # so a stalled call surfaces as an error rather than a UI timeout.
            timeout=60,
            max_retries=2,
        )
        _SDK_MODE = "langchain"
        print(f"[IntentParser] SDK: langchain-google-genai ✓  model={GEMINI_MODEL}")
        return

    except ImportError:
        pass   # langchain-google-genai not installed, try next

    # ── Path B: google-generativeai (new-style v0.8+) ─────────────────────
    # In v0.8+ the GenerativeModel class was moved inside genai directly
    # configure() was removed in v0.8+. Use GenerativeModel() only.
    try:
        import google.generativeai as genai

        # v0.8+ style — no configure(), pass api_key directly to the model
        _lc_llm = genai.GenerativeModel(
            model_name=GEMINI_MODEL,
            # generation_config controls output behaviour
            generation_config={
                "temperature": 0,
                "response_mime_type": "application/json",  # force JSON output
            },
        )
        # We need the key for the actual API call — store it on the object
        # genai uses GOOGLE_API_KEY env var by default in v0.8+
        os.environ.setdefault("GOOGLE_API_KEY", GEMINI_API_KEY)

        _SDK_MODE = "generativeai"
        print(f"[IntentParser] SDK: google-generativeai (v0.8+) ✓  model={GEMINI_MODEL}")
        return

    except ImportError:
        pass   # not installed, fall through

    # ── Path C: Direct REST API (no SDK needed) ───────────────────────────
    # Uses Python's built-in urllib — always available.
    # Less elegant but works everywhere.
    _SDK_MODE = "rest"
    print(f"[IntentParser] SDK: REST fallback (urllib) ✓  model={GEMINI_MODEL}")


# Run setup immediately when this file is imported
_setup_gemini_client()


# ─────────────────────────────────────────────────────────────────
# CONSTANTS — categories, tiers, and field metadata
# ─────────────────────────────────────────────────────────────────

# All valid canonical category names (from Module 1's RAW_TO_CANONICAL map)
# Gemini is only allowed to return values from this list
ALL_CANONICAL_CATEGORIES = sorted(set(RAW_TO_CANONICAL.values()))

# Fields that Gemini can usually figure out from context alone
# We never show MCQ questions for these
ALWAYS_INFERRED_FIELDS = {"categories", "role"}

# Fields we show MCQ questions for ONLY IF Gemini couldn't figure them out
MCQ_ELIGIBLE_FIELDS = {"tiers", "cpv_budget_max", "subcategories", "top_n"}

# Maximum number of MCQ questions per conversation turn (UX limit)
MAX_MCQ_QUESTIONS = 3

# Tier → follower range mapping
# Used to derive followers_min/max when user picks a tier
TIER_RANGES = {
    "Nano":      (0,         10_000),
    "Micro":     (10_000,    50_000),
    "Mid":       (50_000,    100_000),
    "Macro":     (100_000,   500_000),
    "Mega":      (500_000,   1_000_000),
    "Celebrity": (1_000_000, None),
}

# Metadata for each filterable field:
# label        = human-readable question text shown to user
# options_src  = "static" (hardcoded list) or column name in the DataFrame
# static_opts  = the hardcoded list (only used when options_src = "static")
# hint         = short explanation shown under the MCQ question
FIELD_META = {
    "categories": {
        "label":       "Which creator niche fits your campaign?",
        "options_src": "category_norm",    # pull top values from this DataFrame column
        "hint":        "e.g. fitness, beauty, food, finance, gaming",
    },
    "tiers": {
        "label":       "What follower size are you targeting?",
        "options_src": "static",
        "static_opts": TIER_LABELS,        # Nano, Micro, Mid, Macro, Mega, Celebrity
        "hint":        "Nano=<10k  Micro=10k–50k  Mid=50k–100k  Macro=100k–500k  Mega=500k–1M",
    },
    "cpv_budget_max": {
        "label":       "What is your max CPV budget (₹ per view)?",
        "options_src": "static",
        "static_opts": ["₹0.10", "₹0.15", "₹0.20", "₹0.25", "₹0.35", "₹0.50", "No limit"],
        "hint":        "CPV = cost per view. Lower CPV = more views per rupee.",
    },
    "subcategories": {
        "label":       "Any specific content style?",
        "options_src": "subcategory_clean",  # pull top values from DataFrame
        "hint":        "e.g. Training & Fitness, Skin-Care, Food Reviews",
    },
    "top_n": {
        "label":       "How many creators do you need?",
        "options_src": "static",
        "static_opts": ["5", "10", "15", "20"],
        "hint":        "Number of creator profiles to return",
    },
}


# ─────────────────────────────────────────────────────────────────
# DATA CLASSES
# These are simple containers — think of them as labelled boxes
# that hold the data as it flows through the pipeline
# ─────────────────────────────────────────────────────────────────

@dataclass
class MCQQuestion:
    """
    One multiple-choice question shown to the user.
    
    Example:
        field   = "tiers"
        label   = "What follower size are you targeting?"
        options = ["Nano", "Micro", "Mid", "Macro", "No preference"]
        hint    = "Nano=<10k  Micro=10k–50k ..."
    """
    field:   str         # which FilterSpec field this question fills
    label:   str         # the question text
    options: list[str]   # clickable answer choices
    hint:    str         # helpful context shown below the question


@dataclass
class IntentParserResult:
    """
    What run_intent_parser() returns.

    Two possible states:
    
    State A — Gemini got everything it needs:
        needs_mcq   = False
        filter_spec = a ready FilterSpec  ← pass directly to Module 1
        mcq_questions = []
    
    State B — Gemini needs more info from the user:
        needs_mcq   = True
        filter_spec = None               ← not ready yet
        mcq_questions = [...]            ← show these to the user
        Then call finalize_with_mcq_answers() after user responds
    """
    needs_mcq:     bool                  # True = show MCQ before filtering
    filter_spec:   Optional[FilterSpec]  # ready spec (State A only)
    mcq_questions: list[MCQQuestion]     # questions for user (State B only)
    extracted:     dict                  # raw Gemini output (saved for Stage 3 merge)
    confidence:    str                   # "high" / "medium" / "low"
    reasoning:     str                   # Gemini's one-line explanation


# ─────────────────────────────────────────────────────────────────
# THE GEMINI EXTRACTION PROMPT
#
# This is the most important part of the whole file.
# The quality of Gemini's output depends entirely on how clearly
# we explain the task here.
# ─────────────────────────────────────────────────────────────────

def _build_prompt(query: str, role: str) -> str:
    """
    Build the full prompt we send to Gemini.
    
    We inject the real category list so Gemini can only return
    values that actually exist in our DataFrame.
    """
    categories_str   = ", ".join(ALL_CANONICAL_CATEGORIES)
    mcq_eligible_str = ", ".join(sorted(MCQ_ELIGIBLE_FIELDS))

    return f"""
You are the intent parser for Campayn, an Indian creator marketing platform.

TASK
----
Read the user query below and extract structured filter criteria for searching
a database of 56,000+ Indian Instagram creators.

USER ROLE : {role}
USER QUERY: {query}

VALID CATEGORIES (use ONLY these exact strings — no others):
{categories_str}

VALID FOLLOWER TIERS:
Nano (<10k), Micro (10k-50k), Mid (50k-100k), Macro (100k-500k),
Mega (500k-1M), Celebrity (1M+)

OUTPUT FORMAT
-------------
Return ONLY this JSON object. No explanation, no markdown, no extra text.

{{
  "extracted": {{
    "categories":     [],   // list of canonical category strings you are confident about
    "subcategories":  [],   // specific content styles e.g. "Training & Fitness"
    "tiers":          [],   // follower tier names from the valid list above
    "followers_min":  null, // integer or null
    "followers_max":  null, // integer or null
    "cpv_budget_max": null, // float (rupees per view) or null
    "sources":        [],   // from: hashfame, parallel, meme_pages, celebrities, publishers
    "top_n":          null, // integer number of results wanted, or null
    "sort_by":        null, // "followers" | "cpv_mid" | "relevance" | null
    "keyword":        null  // specific name/word to search for, or null
  }},
  "missing_fields": [],     // fields from [{mcq_eligible_str}] you CANNOT infer
  "confidence":     "medium", // "high" | "medium" | "low"
  "reasoning":      ""     // one sentence: what you inferred + what is missing
}}

EXTRACTION RULES
----------------
1. Be aggressive about inferring categories from context:
   "gym culture" or "fitness content"  → ["fitness"]
   "protein brand" or "supplement"     → ["health"]
   "saree" or "kurta" or "ethnic wear" → ["fashion"]
   "meme pages" or "funny content"     → ["comedy"]
   "college kids" or "hostel life"     → ["lifestyle", "comedy"]
   "CA firm" or "stock market"         → ["finance"]
   Hinglish queries are valid — parse them normally.

2. Only add a field to missing_fields if it would meaningfully narrow results
   AND you genuinely cannot infer it from the query.

3. If the user says "micro influencer" → tiers = ["Micro"], do NOT list it as missing.

4. categories is critical — if you truly cannot infer it, set confidence = "low"
   and add "categories" to missing_fields.

5. Never invent category strings not in the VALID CATEGORIES list above.
""".strip()


# ─────────────────────────────────────────────────────────────────
# STAGE 1 — GEMINI EXTRACTION
# ─────────────────────────────────────────────────────────────────

def stage1_extract(query: str, role: str = "brand") -> dict:
    """
    Send the query to Gemini and get back a partial FilterSpec as JSON.

    Returns a dict with these keys:
        extracted      → the filter fields Gemini could figure out
        missing_fields → fields Gemini could NOT infer (needs MCQ)
        confidence     → "high" / "medium" / "low"
        reasoning      → one-line explanation from Gemini

    This function handles all three SDK paths (langchain / generativeai / rest),
    and falls back to Anthropic Claude (api/llm_fallback.py) if Gemini itself
    is unreachable (quota exhausted, outage, bad key) rather than just missing
    an SDK.
    """
    prompt = _build_prompt(query, role)

    raw_text = ""  # will hold the raw string the model returns

    try:
        # ── Path A: LangChain ─────────────────────────────────────────────
        if _SDK_MODE == "langchain":
            # LangChain's .invoke() sends the prompt and returns an AIMessage object
            # .content gives us the text inside the message
            response = _lc_llm.invoke(prompt)
            raw_text = response.content

        # ── Path B: google-generativeai v0.8+ ────────────────────────────
        elif _SDK_MODE == "generativeai":
            # .generate_content() is the main call for the new-style SDK
            # It returns a GenerateContentResponse object
            response = _lc_llm.generate_content(prompt)
            raw_text = response.text

        # ── Path C: REST API fallback ────────────────────────────────────
        elif _SDK_MODE == "rest":
            raw_text = _call_gemini_rest(prompt)

    except Exception as gemini_error:
        print(f"[Stage 1] Gemini failed ({gemini_error}) — trying fallback chain (Nemotron → Claude)...")
        from api.llm_fallback import fallback_text
        raw_text, _provider = fallback_text(prompt, temperature=0, json_mode=True)

    # ── Clean the response ───────────────────────────────────────────────
    # Sometimes the model wraps JSON in ```json ... ``` markdown fences
    # even when we ask it not to. Strip them if present.
    raw_text = raw_text.strip()
    if raw_text.startswith("```"):
        # split on ``` → take middle part → strip "json" prefix if present
        raw_text = raw_text.split("```")[1]
        if raw_text.startswith("json"):
            raw_text = raw_text[4:]
        raw_text = raw_text.strip()

    # ── Parse JSON ───────────────────────────────────────────────────────
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as e:
        # If parsing fails, return a safe default that triggers MCQ for everything
        print(f"[Stage 1] JSON parse error: {e}")
        print(f"[Stage 1] Raw response was: {raw_text[:300]}")
        parsed = {
            "extracted":      {},
            "missing_fields": list(MCQ_ELIGIBLE_FIELDS),
            "confidence":     "low",
            "reasoning":      "Could not parse model response — will collect via MCQ.",
        }

    # Log what happened so you can see it in the terminal
    print(f"[Stage 1] confidence={parsed.get('confidence')}  "
          f"missing={parsed.get('missing_fields')}  "
          f"reasoning={parsed.get('reasoning')}")

    return parsed


def _call_gemini_rest(prompt: str) -> str:
    """
    Direct HTTP call to the Gemini REST API — no SDK needed.
    Only used when neither langchain-google-genai nor google-generativeai is installed.

    Gemini REST endpoint docs:
    https://ai.google.dev/api/generate-content
    """
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    )

    # Build the request body — Gemini expects this exact JSON structure
    body = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0,
            "responseMimeType": "application/json",
        },
    }).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
            # Navigate the nested response structure to get the text
            return data["candidates"][0]["content"]["parts"][0]["text"]
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8")
        raise RuntimeError(f"Gemini REST error {e.code}: {error_body}")


# ─────────────────────────────────────────────────────────────────
# STAGE 2 — MCQ QUESTION GENERATOR
# ─────────────────────────────────────────────────────────────────

def _get_options_for_field(field: str, df: pd.DataFrame) -> list[str]:
    """
    Build the list of answer choices for a given field.

    For "static" fields (tiers, cpv, top_n): use the hardcoded list.
    For DataFrame fields (categories, subcategories): pull the most
    common actual values from the CSV so options are always up-to-date.
    """
    meta = FIELD_META.get(field, {})

    if meta.get("options_src") == "static":
        # Use the hardcoded list directly
        options = list(meta.get("static_opts", []))
    else:
        # Pull the top 7 most frequent values from the DataFrame column
        col = meta.get("options_src", "")
        if col and col in df.columns:
            options = (
                df[col]
                .dropna()
                .value_counts()      # count how many rows have each value
                .head(7)             # take the top 7 most common
                .index.tolist()      # convert to a plain list
            )
        else:
            options = []

    # Always give the user a way to skip the question
    # (don't force a filter they don't care about)
    if "No preference" not in options and "No limit" not in options:
        options.append("No preference")

    return options


def stage2_generate_mcq(
    missing_fields: list[str],
    df: pd.DataFrame,
) -> list[MCQQuestion]:
    """
    For each field Gemini couldn't figure out, create an MCQQuestion.

    We cap at MAX_MCQ_QUESTIONS (3) per turn so we don't overwhelm the user.
    Fields are asked in priority order: categories first, then tiers, etc.

    Args:
        missing_fields : from Stage 1's output
        df             : the normalised DataFrame (for live option values)

    Returns:
        list of MCQQuestion objects to show the user
    """
    # Priority order — ask about the most important things first
    priority = ["categories", "tiers", "cpv_budget_max", "subcategories", "top_n"]

    # Only keep fields that are actually missing AND eligible for MCQ
    eligible = [
        f for f in priority
        if f in missing_fields and f in (MCQ_ELIGIBLE_FIELDS | ALWAYS_INFERRED_FIELDS)
    ]

    questions = []
    for field in eligible[:MAX_MCQ_QUESTIONS]:  # cap at 3 questions
        meta    = FIELD_META.get(field, {})
        options = _get_options_for_field(field, df)
        questions.append(MCQQuestion(
            field   = field,
            label   = meta.get("label", f"Please specify: {field}"),
            options = options,
            hint    = meta.get("hint", ""),
        ))

    print(f"[Stage 2] Generated {len(questions)} MCQ question(s): "
          f"{[q.field for q in questions]}")

    return questions


# ─────────────────────────────────────────────────────────────────
# STAGE 3 — FILTERSPEC MERGE
# ─────────────────────────────────────────────────────────────────

def stage3_merge(extracted: dict, mcq_answers: dict[str, str]) -> FilterSpec:
    """
    Combine Gemini's extracted fields + user's MCQ answers → FilterSpec.

    extracted   : the "extracted" dict from Stage 1
    mcq_answers : {field_name: selected_option_string} from Stage 2

    Priority rule: MCQ answers override Gemini when both exist,
    because the user explicitly chose them.
    """

    # ── CATEGORIES ───────────────────────────────────────────────────────
    categories = extracted.get("categories") or []
    if "categories" in mcq_answers:
        ans = mcq_answers["categories"]
        if ans not in ("No preference", ""):
            # Normalise to canonical key in case user picked a raw display name
            norm = RAW_TO_CANONICAL.get(ans.lower().strip(), ans.lower().strip())
            categories = [norm]

    # ── TIERS ────────────────────────────────────────────────────────────
    tiers = extracted.get("tiers") or []
    if "tiers" in mcq_answers:
        ans = mcq_answers["tiers"]
        if ans not in ("No preference", ""):
            # MCQ option might be "Micro (10k-50k)" — take just the tier name
            tiers = [ans.split("(")[0].strip()]

    # ── CPV BUDGET ───────────────────────────────────────────────────────
    cpv = extracted.get("cpv_budget_max")
    if "cpv_budget_max" in mcq_answers:
        ans = mcq_answers["cpv_budget_max"]
        if ans not in ("No limit", "No preference", ""):
            try:
                # Option is like "₹0.25" → strip ₹ → convert to float
                cpv = float(ans.replace("₹", "").strip())
            except ValueError:
                cpv = None   # couldn't parse — leave unconstrained

    # ── SUBCATEGORIES ─────────────────────────────────────────────────────
    subcats = extracted.get("subcategories") or []
    if "subcategories" in mcq_answers:
        ans = mcq_answers["subcategories"]
        if ans not in ("No preference", ""):
            subcats = [ans]

    # ── TOP-N ─────────────────────────────────────────────────────────────
    top_n = extracted.get("top_n") or 10
    if "top_n" in mcq_answers:
        try:
            top_n = int(mcq_answers["top_n"])
        except (ValueError, TypeError):
            top_n = 10

    # ── FOLLOWER RANGE ────────────────────────────────────────────────────
    # Use explicit range if Gemini found one
    followers_min = extracted.get("followers_min")
    followers_max = extracted.get("followers_max")

    # If tiers were selected but no explicit follower numbers were given,
    # derive min/max from the tier definition
    # e.g. "Micro" → followers_min=10000, followers_max=50000
    if tiers and not followers_min and not followers_max:
        mins, maxs = [], []
        for tier in tiers:
            lo, hi = TIER_RANGES.get(tier, (None, None))
            if lo is not None:
                mins.append(lo)
            if hi is not None:
                maxs.append(hi)
        followers_min = min(mins) if mins else None
        followers_max = max(maxs) if maxs else None

    # ── SORT ORDER ────────────────────────────────────────────────────────
    sort_by = extracted.get("sort_by") or "followers"
    valid_sorts = {"followers", "cpv_mid", "relevance"}
    if sort_by not in valid_sorts:
        sort_by = "followers"

    # ── ASSEMBLE FINAL FILTERSPEC ─────────────────────────────────────────
    spec = FilterSpec(
        categories     = categories     or None,
        subcategories  = subcats        or None,
        followers_min  = followers_min,
        followers_max  = followers_max,
        tiers          = tiers          or None,
        cpv_budget_max = cpv,
        sources        = extracted.get("sources") or None,
        keyword        = extracted.get("keyword"),
        top_n          = min(int(top_n), 50),   # cap at 50 to protect prompt size
        sort_by        = sort_by,
        sort_ascending = (sort_by == "cpv_mid"), # ascending only when sorting by cost
    )

    print(f"[Stage 3] FilterSpec ready → {asdict(spec)}")
    return spec


# ─────────────────────────────────────────────────────────────────
# PUBLIC API — WHAT OTHER FILES CALL
# ─────────────────────────────────────────────────────────────────

def run_intent_parser(
    query: str,
    df: pd.DataFrame,
    role: str = "brand",
) -> IntentParserResult:
    """
    Main entry point — called by the LangGraph orchestrator node.

    Runs Stage 1 (Gemini extraction) + Stage 2 (MCQ generation if needed).

    Returns an IntentParserResult. Check result.needs_mcq:
        False → result.filter_spec is ready, pass it to Module 1
        True  → show result.mcq_questions to user,
                 then call finalize_with_mcq_answers()
    """
    # ── Stage 1: ask Gemini to extract what it can ────────────────────────
    stage1_output  = stage1_extract(query, role)
    extracted      = stage1_output.get("extracted", {})
    missing_fields = stage1_output.get("missing_fields", [])
    confidence     = stage1_output.get("confidence", "low")
    reasoning      = stage1_output.get("reasoning", "")

    # ── Decide if we need MCQ ─────────────────────────────────────────────
    # Only ask MCQ for fields that are genuinely MCQ-eligible
    fields_to_ask = [
        f for f in missing_fields
        if f in (MCQ_ELIGIBLE_FIELDS | ALWAYS_INFERRED_FIELDS)
    ]

    if not fields_to_ask:
        # Gemini got everything — skip MCQ, build the spec now
        spec = stage3_merge(extracted, mcq_answers={})
        return IntentParserResult(
            needs_mcq     = False,
            filter_spec   = spec,
            mcq_questions = [],
            extracted     = extracted,
            confidence    = confidence,
            reasoning     = reasoning,
        )

    # ── Stage 2: generate MCQ questions for missing fields ────────────────
    questions = stage2_generate_mcq(fields_to_ask, df)

    return IntentParserResult(
        needs_mcq     = True,
        filter_spec   = None,        # not ready yet — waiting for MCQ answers
        mcq_questions = questions,
        extracted     = extracted,
        confidence    = confidence,
        reasoning     = reasoning,
    )


def finalize_with_mcq_answers(
    parser_result: IntentParserResult,
    mcq_answers: dict[str, str],
) -> FilterSpec:
    """
    Called AFTER the user has answered the MCQ questions.

    parser_result : the IntentParserResult from run_intent_parser()
    mcq_answers   : {field_name: selected_option_string}
                    e.g. {"tiers": "Micro (10k-50k)", "cpv_budget_max": "₹0.20"}

    Returns a complete FilterSpec ready for Module 1.
    """
    return stage3_merge(parser_result.extracted, mcq_answers)


# ─────────────────────────────────────────────────────────────────
# UI HELPERS — formatting for the chat interface
# ─────────────────────────────────────────────────────────────────

def format_mcq_for_chat(questions: list[MCQQuestion]) -> str:
    """
    Convert MCQQuestion objects into a readable chat message string.
    Used by the Streamlit UI or FastAPI response layer.

    Example output:
        To find the best creators, I need a few quick details:

        1. What follower size are you targeting?
           (Nano=<10k  Micro=10k–50k  Mid=50k–100k)
           → Nano | Micro | Mid | Macro | No preference

        2. What is your max CPV budget (₹ per view)?
           → ₹0.10 | ₹0.15 | ₹0.20 | ₹0.25 | No limit
    """
    lines = ["To find the best creators for you, I need a few quick details:\n"]

    for i, q in enumerate(questions, start=1):
        opts_str = " | ".join(q.options)
        lines.append(f"{i}. {q.label}")
        if q.hint:
            lines.append(f"   ({q.hint})")
        lines.append(f"   → {opts_str}\n")

    return "\n".join(lines)


def mcq_questions_to_dict(questions: list[MCQQuestion]) -> list[dict]:
    """
    Serialise MCQQuestion objects to plain dicts.
    Useful for storing in LangGraph state (which must be JSON-serialisable).
    """
    return [
        {
            "field":   q.field,
            "label":   q.label,
            "options": q.options,
            "hint":    q.hint,
        }
        for q in questions
    ]


# ─────────────────────────────────────────────────────────────────
# SMOKE TEST — run this file directly to test without the full app
#
#   python -m api.intent_parser data/raw/creator_profiles.csv
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from pathlib import Path
    from api.modules.profile_filter import load_and_normalise, filter_profiles

    # Load the creator profiles DataFrame
    csv_path  = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/raw/creator_profiles.csv")
    df_global = load_and_normalise(csv_path)

    # Test queries — ranging from crystal clear to very vague
    # Format: (role, query, simulated_mcq_answers_if_needed)
    test_cases = [
        ("brand",   "find a creator for my new protein brand",
                    {"tiers": "Micro", "cpv_budget_max": "₹0.25"}),

        ("brand",   "I want someone who vibes with gym culture",
                    {"tiers": "Mid"}),

        ("brand",   "meme pages for a food delivery app",
                    {}),

        ("brand",   "top 10 fitness micro-influencers under 0.25 CPV",
                    {}),

        ("creator", "show me peer fitness creators around 80k followers",
                    {}),

        ("brand",   "saree brand targeting Tier 2 cities",
                    {"tiers": "Micro", "cpv_budget_max": "₹0.20"}),
    ]

    for role, query, sim_answers in test_cases:
        print("\n" + "=" * 65)
        print(f"ROLE : {role.upper()}")
        print(f"QUERY: {query}")
        print("-" * 65)

        # Run the full intent parser
        result = run_intent_parser(query, df_global, role)

        print(f"needs_mcq  : {result.needs_mcq}")
        print(f"confidence : {result.confidence}")
        print(f"reasoning  : {result.reasoning}")

        if result.needs_mcq:
            print("\n--- MCQ Questions ---")
            print(format_mcq_for_chat(result.mcq_questions))

            if sim_answers:
                print(f"--- Simulated answers: {sim_answers} ---")
                spec = finalize_with_mcq_answers(result, sim_answers)
            else:
                # No simulated answers — build spec with whatever Gemini extracted
                spec = stage3_merge(result.extracted, {})
        else:
            spec = result.filter_spec

        # Run Module 1 with the complete spec
        print("\n--- Module 1 Filter Results ---")
        top_df, meta = filter_profiles(df_global, spec)
        print(f"Total matches : {meta['total_matches']:,}")
        print(f"Returned      : {meta['returned']}")
        print(f"Confidence    : {meta['confidence']}")

        if not top_df.empty:
            print("\nTop 3 creators:")
            for _, row in top_df.head(3).iterrows():
                print(f"  {row['name']:<25} @{row['ig_handle']:<25} "
                      f"{row['followers']:>10,} followers  {row['tier']:<10} "
                      f"₹{row['cpv_min']}–{row['cpv_max']} CPV")