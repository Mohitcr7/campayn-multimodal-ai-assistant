"""
graph.py
========
Campayn AI Assistant — LangGraph Orchestration

WHAT THIS FILE DOES
--------------------
Defines the full LangGraph pipeline as a stateful graph.
Each node is one step in the pipeline. Edges connect them.
The graph runs every time a user sends a message.

UPDATED GRAPH FLOW — PARALLEL MODULE EXECUTION
------------------------------------------------

                    ┌─────────────────┐
                    │   User query    │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  intent_parser  │  ← Gemini extracts FilterSpec
                    └────────┬────────┘
                             │
              ┌──────────────┼─────────────┐
              │  Send()      │  Send()     │  Send() ← LangGraph fan-out
              ▼              ▼             ▼
        ┌──────────┐  ┌──────────┐  ┌──────────┐
        │ module1  │  │ module2  │  │ module3  │  ← ALL run in PARALLEL
        │(profiles)│  │  (RAG)   │  │(vision)  │  ← Module3 skipped if
        └────┬─────┘  └────┬─────┘  └────┬─────┘    no image present
             │             │             │
             └─────────────┼─────────────┘
                           │  fan-in: prompt_builder waits for all 3
                  ┌────────▼────────┐
                  │  prompt_builder │  ← assembles final prompt
                  └────────┬────────┘
                           │
                  ┌────────▼────────┐
                  │   generator     │  ← Gemini final response
                  └────────┬────────┘
                           │
                  ┌────────▼────────┐
                  │  save_memory    │  ← Redis conversation history
                  └────────┬────────┘
                           │
                  ┌────────▼────────┐
                  │    Response     │
                  └─────────────────┘

WHY PARALLEL? (Key design change from sequential)
--------------------------------------------------
In sequential mode, total latency = sum of all module times:
  Module 1 (pandas)  ~150ms
  Module 2 (RAG)     ~650ms
  Module 3 (vision)  ~900ms  ← only if image present
  Sequential total   ~1700ms for all three

In parallel mode, total latency = max(module times):
  All three run simultaneously after intent parsing
  Parallel total     ~900ms  ← bottleneck is the slowest module
  Saving             ~800ms per image query, ~500ms per text query

This works because Modules 1, 2, 3 are completely independent:
  - None of them read from each other's output
  - They all just need filter_spec from intent_parser
  - They all write to different keys in state (no conflicts)

HOW LANGGRAPH PARALLELISM WORKS
---------------------------------
LangGraph's Send() API dispatches multiple nodes simultaneously.
When a conditional edge function returns a LIST of Send() objects
instead of a single string, LangGraph treats this as parallel dispatch:
  - All nodes in the list start at the same time
  - Each node writes to its own keys in state independently
  - The fan-in node (prompt_builder) automatically waits for ALL
    parallel nodes to complete before it starts
  - You write zero synchronisation code — LangGraph handles it

KEY DESIGN DECISIONS
---------------------
1. Modules 1, 2, 3 run in PARALLEL via Send() fan-out
   — ~800ms latency saving on image queries

2. Module 3 is conditional — only dispatched if image_b64 is present
   — no wasted Gemini vision API call when there is no image

3. MCQ interrupt — if intent parser needs more info from the user,
   the graph pauses and returns MCQ questions before running modules

4. Conversation memory in Redis per session_id
   — loaded at graph start, saved at graph end

Author : Mohit Chauhan — Campayn Multimodal AI Assistant
"""

from __future__ import annotations

import json
import os
import time
import uuid
from functools import lru_cache
from typing import Annotated, Any, Optional

# ── LangGraph core ────────────────────────────────────────────────
# StateGraph  = the container we add nodes and edges into
# END         = special marker meaning "pipeline is done, stop here"
# Send        = dispatches a node in parallel (the fan-out primitive)
from langgraph.graph  import StateGraph, END
from langgraph.types  import Send                    # ← NEW: parallel dispatch
from langgraph.checkpoint.memory import MemorySaver  # saves state for MCQ resume

# ── TypedDict for state schema ────────────────────────────────────
from typing_extensions import TypedDict

# ── Gemini (final generation only) ───────────────────────────────
from google      import genai
from google.genai import types

# ── Our modules — imported as plain Python functions ──────────────
# LangGraph doesn't care what's inside each node function.
# It just calls them with state and merges what they return.
from api.modules.profile_filter import load_and_normalise, run_module1
from api.modules.rag            import run_module2
from api.modules.vision         import run_module3
from api.intent_parser          import (
    run_intent_parser,
    finalize_with_mcq_answers,
    format_mcq_for_chat,
    IntentParserResult,
)
from api.prompt_constructor import (
    PromptInput,
    build_prompt,
    parse_response,
    BRAND_SYSTEM_PROMPT,
    CREATOR_SYSTEM_PROMPT,
)
from api.llm_fallback import generate_text


# ─────────────────────────────────────────────────────────────────
# CONFIGURATION
# All values read from .env file via environment variables.
# Never hardcode API keys here.
# ─────────────────────────────────────────────────────────────────

GEMINI_API_KEY    = os.environ.get("GEMINI_API_KEY",             "")
GEMINI_MODEL      = os.environ.get("GEMINI_MODEL",               "gemini-2.0-flash")
REDIS_URL         = os.environ.get("REDIS_URL",                  "redis://localhost:6379")
HISTORY_TTL       = int(os.environ.get("CONVERSATION_TTL_SECONDS","3600"))
MAX_HISTORY_TURNS = 3


# ─────────────────────────────────────────────────────────────────
# SHARED RESOURCES
# These are initialised once at startup and reused for every request.
# Loading 56k profiles or creating a Gemini client takes ~2 seconds.
# We do it once, not on every user message.
# ─────────────────────────────────────────────────────────────────

# Global DataFrame — None until first call to get_dataframe()
_df = None

def get_dataframe():
    """
    Lazy-load the creator profiles DataFrame.
    First call: reads CSV, normalises it, stores in _df.
    Every subsequent call: returns the already-loaded _df instantly.
    This pattern is called a "singleton" — one shared instance.
    """
    global _df
    if _df is None:
        csv_path = os.environ.get(
            "CREATOR_PROFILES_PATH", "data/raw/creator_profiles.csv"
        )
        print(f"[Graph] Loading creator profiles: {csv_path}...")
        _df = load_and_normalise(csv_path)
        print(f"[Graph] {len(_df):,} creator profiles loaded into memory")
    return _df


@lru_cache(maxsize=1)
def get_gemini_client() -> genai.Client:
    """
    Create Gemini client once, cache forever.
    @lru_cache(maxsize=1) = "run this function once, return the cached
    result on every subsequent call". Same as the _df singleton above
    but using Python's built-in caching decorator.
    """
    return genai.Client(api_key=GEMINI_API_KEY)


# ─────────────────────────────────────────────────────────────────
# STATE — the shared whiteboard every node reads from and writes to
#
# CampaynState is a TypedDict — a Python dict with declared types.
# Every node in the graph receives the FULL state and returns only
# the keys it changed. LangGraph merges the return dict into state.
#
# Think of it as a relay baton that accumulates data at every step.
# ─────────────────────────────────────────────────────────────────

def _merge_latency(left: dict, right: dict) -> dict:
    """
    Reducer for node_latency. Modules 1/2/3 run in PARALLEL and each writes
    node_latency in the same super-step. Without a reducer LangGraph rejects
    the concurrent writes ("can receive only one value per step"). Merging
    the dicts lets every parallel node contribute its own timing key.
    """
    return {**(left or {}), **(right or {})}


class CampaynState(TypedDict):
    # ── What the user sent ────────────────────────────────────────
    query:           str              # the user's message
    role:            str              # "brand" or "creator"
    session_id:      str              # unique ID for this conversation
    image_b64:       Optional[str]    # base64 image, or None if no image

    # ── Intent parser output ──────────────────────────────────────
    filter_spec:     Optional[dict]   # pandas FilterSpec as a JSON-safe dict
    needs_mcq:       bool             # True = pipeline paused, awaiting user input
    mcq_questions:   Optional[list]   # the MCQ questions to show the user
    mcq_answers:     Optional[dict]   # user's answers {field_name: selected_option}
    intent_result:   Optional[dict]   # raw Gemini extraction (saved for MCQ resume)

    # ── Module outputs (written in parallel) ──────────────────────
    # Each module writes to its own keys — no conflicts in parallel
    module1_block:   Optional[str]    # formatted creator profile list
    module1_meta:    Optional[dict]   # match count, confidence, filter summary
    module2_block:   Optional[str]    # formatted RAG knowledge chunks
    module2_meta:    Optional[dict]   # chunk count, similarity scores
    module3_block:   Optional[str]    # formatted vision analysis
    module3_meta:    Optional[dict]   # scores, brand safety, model used

    # ── Generation ────────────────────────────────────────────────
    prompt:          Optional[str]    # assembled final prompt (all modules merged)
    response:        Optional[str]    # raw LLM text response
    parsed_response: Optional[dict]   # cleaned response for the API layer

    # ── Conversation memory ───────────────────────────────────────
    history:         list[dict]       # [{role: "user", content: "..."}, ...]

    # ── Observability ─────────────────────────────────────────────
    trace_id:        Optional[str]    # LangSmith run ID
    node_latency:    Annotated[dict, _merge_latency]  # {node_name: ms}; parallel-safe


# ─────────────────────────────────────────────────────────────────
# REDIS MEMORY HELPERS
# Redis is a fast key-value store used to persist conversation history
# between API calls. Without it, the assistant forgets what was said.
# ─────────────────────────────────────────────────────────────────

def _load_history(session_id: str) -> list[dict]:
    """
    Load this session's conversation history from Redis.
    Returns an empty list if Redis is unavailable or session is new.
    Fails silently — conversation works even without Redis (just no memory).
    """
    try:
        import redis
        r   = redis.from_url(REDIS_URL, decode_responses=True)
        raw = r.get(f"campayn:history:{session_id}")
        return json.loads(raw) if raw else []
    except Exception as e:
        print(f"[Memory] Redis load failed: {e} — starting with empty history")
        return []


def _save_history(session_id: str, history: list[dict]) -> None:
    """
    Save conversation history to Redis with a TTL (expiry time).
    HISTORY_TTL = 3600 seconds = 1 hour by default.
    After TTL expires, old sessions are automatically cleaned up.
    """
    try:
        import redis
        r = redis.from_url(REDIS_URL, decode_responses=True)
        r.setex(
            f"campayn:history:{session_id}",   # key
            HISTORY_TTL,                        # seconds until expiry
            json.dumps(history),                # value
        )
    except Exception as e:
        print(f"[Memory] Redis save failed: {e}")


# ─────────────────────────────────────────────────────────────────
# NODE 1 — INTENT PARSER
#
# First node to run. Sends the user query to Gemini and extracts
# a structured FilterSpec. If Gemini can't infer all required fields,
# it sets needs_mcq=True — the graph will stop and ask the user.
#
# Signature rule: every node takes (state: CampaynState) → dict
# The dict contains ONLY the keys this node changed.
# ─────────────────────────────────────────────────────────────────

def node_intent_parser(state: CampaynState) -> dict:
    """
    Runs Gemini to parse the query into a structured FilterSpec.

    TWO paths through this node:
    Path A — MCQ answers just arrived (resuming after MCQ pause):
        Merge the MCQ answers with the previously extracted spec.
        Return the complete FilterSpec and set needs_mcq=False.

    Path B — Fresh query (no prior MCQ answers):
        Call Gemini to extract what it can.
        If complete → return FilterSpec, needs_mcq=False.
        If incomplete → return MCQ questions, needs_mcq=True.
        The graph will route to END and pause.
    """
    t0 = time.time()

    # Always load history first — needed for context-aware responses
    history = _load_history(state["session_id"])

    df = get_dataframe()

    # ── Path A: MCQ answers just came in — finalise the spec ─────────
    # Trigger on mcq_answers alone. run_pipeline rebuilds a fresh state on
    # every call (intent_result=None), so the originally-extracted spec does
    # NOT survive the client round-trip. If it happens to be present (e.g.
    # restored by the checkpointer) reuse it; otherwise re-extract what the
    # model can infer. Either way finalize_with_mcq_answers() fills the gaps
    # from the user's selections, so the spec comes out complete and we skip
    # re-asking (needs_mcq=False). Without this, missing intent_result made
    # the resume condition never fire → the same MCQ was asked forever.
    if state.get("mcq_answers"):
        import dataclasses
        stored = state.get("intent_result")
        if stored:
            extracted = stored["extracted"]
        else:
            reparse   = run_intent_parser(query=state["query"], df=df, role=state["role"])
            extracted = reparse.extracted

        spec = finalize_with_mcq_answers(
            # Minimal duck-typed object — only needs .extracted attribute
            type("R", (), {"extracted": extracted})(),
            state["mcq_answers"],
        )
        return {
            "filter_spec":   dataclasses.asdict(spec),  # dict so state stays JSON-safe
            "needs_mcq":     False,
            "mcq_questions": None,
            "intent_result": None,
            "history":       history,
            "node_latency": {
                **state.get("node_latency", {}),
                "intent_parser": round((time.time() - t0) * 1000, 1),
            },
        }

    # ── Path B: Fresh query — run Gemini intent parser ────────────────
    result: IntentParserResult = run_intent_parser(
        query = state["query"],
        df    = df,
        role  = state["role"],
    )

    if result.needs_mcq:
        # Gemini couldn't infer enough — pause and ask the user
        # needs_mcq=True triggers the conditional edge → END
        print(f"[IntentParser] MCQ needed for fields: "
              f"{[q.field for q in result.mcq_questions]}")
        return {
            "needs_mcq":     True,
            "mcq_questions": [
                {
                    "field":   q.field,
                    "label":   q.label,
                    "options": q.options,
                    "hint":    q.hint,
                }
                for q in result.mcq_questions
            ],
            # Save extracted so we can merge it when MCQ answers arrive
            "intent_result": {
                "extracted":  result.extracted,
                "confidence": result.confidence,
                "reasoning":  result.reasoning,
            },
            "history":      history,
            "node_latency": {
                **state.get("node_latency", {}),
                "intent_parser": round((time.time() - t0) * 1000, 1),
            },
        }

    # Gemini got everything — spec is complete, continue to modules
    import dataclasses
    return {
        "filter_spec":  dataclasses.asdict(result.filter_spec),
        "needs_mcq":    False,
        "mcq_questions":None,
        "intent_result":None,
        "history":      history,
        "node_latency": {
            **state.get("node_latency", {}),
            "intent_parser": round((time.time() - t0) * 1000, 1),
        },
    }


# ─────────────────────────────────────────────────────────────────
# NODE 2 — MODULE 1: CREATOR PROFILE FILTER
#
# Runs pandas filtering on the 56k creator profiles DataFrame.
# Fast (~150ms) because it's pure in-memory computation — no API calls.
# Runs in PARALLEL with Module 2 and Module 3.
# ─────────────────────────────────────────────────────────────────

def node_module1(state: CampaynState) -> dict:
    """
    Filters 56k creator profiles using the FilterSpec from intent_parser.
    Returns the top-N matching profiles as a formatted string block.

    Why it's safe to run in parallel:
    - Reads only: query, role, filter_spec from state
    - Writes only: module1_block, module1_meta
    - No overlap with Module 2 (reads module2_block) or Module 3 (module3_block)
    """
    t0 = time.time()

    from api.modules.profile_filter import FilterSpec

    # Reconstruct the FilterSpec dataclass from its JSON-safe dict form
    # (State stores dicts, not dataclasses, to stay JSON-serialisable)
    spec_dict = state.get("filter_spec") or {}
    spec = FilterSpec(**{
        k: v for k, v in spec_dict.items()
        if k in FilterSpec.__dataclass_fields__
    })

    df = get_dataframe()

    # run_module1 returns (prompt_block, metadata, top_df)
    # prompt_block = formatted string ready for the Prompt Constructor
    block, meta, _ = run_module1(
        query        = state["query"],
        df           = df,
        role         = state["role"],
        spec_override= spec,  # use the parsed spec, not NL re-parsing
    )

    return {
        "module1_block": block,
        "module1_meta":  meta,
        "node_latency":  {
            **state.get("node_latency", {}),
            "module1": round((time.time() - t0) * 1000, 1),
        },
    }


# ─────────────────────────────────────────────────────────────────
# NODE 3 — MODULE 2: RAG KNOWLEDGE RETRIEVAL
#
# Embeds the query with Gemini Embedding 2, searches ChromaDB,
# returns the most relevant knowledge base chunks.
# Runs in PARALLEL with Module 1 and Module 3.
# ─────────────────────────────────────────────────────────────────

def node_module2(state: CampaynState) -> dict:
    """
    Retrieves relevant knowledge chunks from ChromaDB using semantic search.
    Returns a formatted block of the top-k most relevant sections.

    Why it's safe to run in parallel:
    - Reads only: query, role from state
    - Writes only: module2_block, module2_meta
    - Independent of what Module 1 or Module 3 are doing
    """
    t0 = time.time()

    # run_module2 handles embedding + ChromaDB search + re-ranking
    block, meta = run_module2(
        query = state["query"],
        role  = state["role"],
    )

    return {
        "module2_block": block,
        "module2_meta":  meta,
        "node_latency":  {
            **state.get("node_latency", {}),
            "module2": round((time.time() - t0) * 1000, 1),
        },
    }


# ─────────────────────────────────────────────────────────────────
# NODE 4 — MODULE 3: VISION ANALYSIS
#
# Sends the uploaded image to Gemini 2.0 Flash for multimodal analysis.
# Runs in PARALLEL with Module 1 and Module 2.
# Only dispatched by fan_out_to_modules() if image_b64 is present.
# ─────────────────────────────────────────────────────────────────

def node_module3(state: CampaynState) -> dict:
    """
    Analyses the uploaded image using Gemini 2.0 Flash vision.
    Returns structured feedback on hook strength, lighting, brand safety, CTA.

    This node is only dispatched when image_b64 is present in state.
    The fan_out_to_modules() function checks this before sending.
    If no image → node is never called → module3_block stays None.

    Why it's safe to run in parallel:
    - Reads only: image_b64, role from state
    - Writes only: module3_block, module3_meta
    - Completely independent of Modules 1 and 2
    """
    t0 = time.time()

    image_b64 = state.get("image_b64")

    # Defensive check — should never reach here without an image
    # because fan_out_to_modules() guards this, but safe to double-check
    if not image_b64:
        return {
            "module3_block": None,
            "module3_meta":  {"vision_used": False, "error": "No image provided"},
            "node_latency":  {**state.get("node_latency", {}), "module3": 0},
        }

    block, meta = run_module3(
        image_input = image_b64,
        role        = state["role"],
    )

    return {
        "module3_block": block,
        "module3_meta":  meta,
        "node_latency":  {
            **state.get("node_latency", {}),
            "module3": round((time.time() - t0) * 1000, 1),
        },
    }


# ─────────────────────────────────────────────────────────────────
# NODE 5 — PROMPT BUILDER
#
# FAN-IN POINT: this node runs only after ALL parallel modules finish.
# Assembles the outputs of Modules 1, 2, 3 into one final LLM prompt.
# ─────────────────────────────────────────────────────────────────

def node_prompt_builder(state: CampaynState) -> dict:
    """
    Assembles all module outputs into the final prompt string.

    This is the fan-in node — LangGraph automatically waits for ALL
    parallel nodes (module1, module2, module3) to complete before
    running this node. No synchronisation code needed.

    By the time this runs, state has:
      module1_block → creator profiles (or None if no matches)
      module2_block → knowledge chunks (or None if low confidence)
      module3_block → vision analysis  (or None if no image)
    """
    t0 = time.time()

    # PromptInput packages all module outputs for build_prompt()
    prompt_input = PromptInput(
        query          = state["query"],
        role           = state["role"],
        module1_block  = state.get("module1_block"),
        module2_block  = state.get("module2_block"),
        module3_block  = state.get("module3_block"),
        module1_meta   = state.get("module1_meta"),
        module2_meta   = state.get("module2_meta"),
        module3_meta   = state.get("module3_meta"),
        history        = state.get("history", []),
        max_history_turns = MAX_HISTORY_TURNS,
    )

    # build_prompt() assembles: system prompt + RAG chunks + profiles
    #                           + vision analysis + history + query
    prompt = build_prompt(prompt_input)

    return {
        "prompt":      prompt,
        "node_latency":{
            **state.get("node_latency", {}),
            "prompt_builder": round((time.time() - t0) * 1000, 1),
        },
    }


# ─────────────────────────────────────────────────────────────────
# NODE 6 — GENERATOR
#
# Sends the assembled prompt to Gemini and gets the final response.
# This is the only node that calls the generation model.
# ─────────────────────────────────────────────────────────────────

def node_generator(state: CampaynState) -> dict:
    """
    Sends the final assembled prompt to Gemini 2.0 Flash and returns
    the raw text response. Falls back to Anthropic Claude if Gemini fails
    (see api/llm_fallback.py).

    temperature=0.4 → slight creativity for natural, varied responses
    max_output_tokens=1500 → enough for detailed campaign advice
    """
    t0 = time.time()

    prompt = state["prompt"]
    raw_text, _provider_used = generate_text(prompt, temperature=0.4, max_tokens=1500)
    parsed = parse_response(raw_text, state["role"])

    return {
        "response":        raw_text,
        "parsed_response": parsed,
        "node_latency":    {
            **state.get("node_latency", {}),
            "generator": round((time.time() - t0) * 1000, 1),
        },
    }


# ─────────────────────────────────────────────────────────────────
# NODE 7 — SAVE MEMORY
#
# Appends this turn to Redis conversation history.
# Always the last node before END.
# ─────────────────────────────────────────────────────────────────

def node_save_memory(state: CampaynState) -> dict:
    """
    Saves this conversation turn to Redis so future messages have context.

    We keep only the last 20 messages (10 turns) to prevent the history
    from growing unboundedly and blowing up the prompt size.
    """
    history = list(state.get("history") or [])

    # Append this turn — user message first, then assistant response
    history.append({"role": "user",      "content": state["query"]})
    history.append({"role": "assistant", "content": state.get("response", "")})

    # Cap at 20 messages = 10 turns
    if len(history) > 20:
        history = history[-20:]

    _save_history(state["session_id"], history)

    return {"history": history}


# ─────────────────────────────────────────────────────────────────
# ROUTING FUNCTIONS — the brain of the edge logic
#
# These functions are called by LangGraph to decide where to go next.
# They read from state and return either:
#   A string → name of the next node
#   A list of Send() objects → dispatch multiple nodes in parallel
# ─────────────────────────────────────────────────────────────────

def route_after_intent(state: CampaynState):
    """
    Called after node_intent_parser completes.
    Decides: pause for MCQ, or fan out to the modules in parallel?

    Returns either:
      - END          → pause the graph, return the MCQ to the user
      - list[Send]   → dispatch modules 1/2/(3) simultaneously

    LangGraph dispatches a returned list of Send() objects as a parallel
    fan-out. (Path-map targets must be node names / END, so the Send list
    has to come straight out of the routing function — not from a path map.)
    """
    if state.get("needs_mcq"):
        return END                       # pause — graph stops, returns MCQ to user
    return fan_out_to_modules(state)     # continue — parallel fan-out


def fan_out_to_modules(state: CampaynState) -> list[Send]:
    """
    ★ THE KEY PARALLEL EXECUTION FUNCTION ★

    Called when route_after_intent() returns "run_modules".
    Returns a LIST of Send() objects — one per module to run.

    When LangGraph sees a list of Send() objects, it dispatches
    ALL of them simultaneously. Each Send() says:
      "run this node with this state"

    Module 3 is only included if there's an image in state.
    This avoids a wasted Gemini vision API call when there's nothing to analyse.

    Why this is safe (no race conditions):
    - Module 1 writes only to: module1_block, module1_meta
    - Module 2 writes only to: module2_block, module2_meta
    - Module 3 writes only to: module3_block, module3_meta
    - No two modules write to the same key → no conflicts
    """
    sends = [
        # Always run Module 1 (pandas — fast, no API call)
        Send("node_module1", state),

        # Always run Module 2 (RAG — Gemini embed + ChromaDB)
        Send("node_module2", state),
    ]

    # Conditionally add Module 3 only if the user uploaded an image
    if state.get("image_b64"):
        sends.append(Send("node_module3", state))
        print("[Graph] Dispatching Modules 1, 2, 3 in parallel (image detected)")
    else:
        print("[Graph] Dispatching Modules 1, 2 in parallel (no image)")

    # Returning a list tells LangGraph: run all of these RIGHT NOW, simultaneously
    return sends


# ─────────────────────────────────────────────────────────────────
# GRAPH CONSTRUCTION
#
# This function wires everything together:
#   add_node()               → register a function under a name
#   set_entry_point()        → which node runs first
#   add_conditional_edges()  → routing with a function
#   add_edge()               → fixed, unconditional connection
#   compile()                → lock the graph, enable checkpointing
# ─────────────────────────────────────────────────────────────────

def build_graph() -> Any:
    """
    Assemble and compile the full LangGraph pipeline.

    PARALLEL EXECUTION PATTERN EXPLAINED
    ──────────────────────────────────────
    Sequential (old):
      intent_parser → module1 → module2 → module3 → prompt_builder

    Parallel (new):
      intent_parser → [module1, module2, module3] → prompt_builder
                       ↑ all three run at the same time ↑

    The fan-out is achieved by:
      add_conditional_edges("intent_parser", fan_out_to_modules)
      where fan_out_to_modules() returns a list of Send() objects.

    The fan-in is automatic:
      When module1, module2, module3 all have edges to prompt_builder,
      LangGraph waits for ALL of them before starting prompt_builder.
      This is built into LangGraph — no code needed.
    """
    builder = StateGraph(CampaynState)

    # ── Step 1: Register all nodes ────────────────────────────────
    # add_node(name, function) — name is used in edges to refer to this node
    builder.add_node("intent_parser",  node_intent_parser)
    builder.add_node("node_module1",   node_module1)
    builder.add_node("node_module2",   node_module2)
    builder.add_node("node_module3",   node_module3)
    builder.add_node("prompt_builder", node_prompt_builder)
    builder.add_node("generator",      node_generator)
    builder.add_node("save_memory",    node_save_memory)

    # ── Step 2: Set entry point ────────────────────────────────────
    # The first node LangGraph runs when graph.invoke() is called
    builder.set_entry_point("intent_parser")

    # ── Step 3: Conditional edge after intent_parser ───────────────
    # route_after_intent() returns either END (MCQ pause) or a list of
    # Send() objects (parallel fan-out to the module nodes). No path_map:
    # the destinations are carried by the Send() objects themselves, and
    # END is a built-in target.
    builder.add_conditional_edges(
        "intent_parser",       # from this node
        route_after_intent,    # returns END or list[Send]
    )

    # ── Step 4: Fan-in edges to prompt_builder ─────────────────────
    # All three module nodes connect to prompt_builder.
    # LangGraph's rule: a node only starts when ALL its incoming
    # edges have completed. So prompt_builder waits for module1 AND
    # module2 AND module3 — automatic synchronisation, no code needed.
    builder.add_edge("node_module1", "prompt_builder")
    builder.add_edge("node_module2", "prompt_builder")
    builder.add_edge("node_module3", "prompt_builder")
    # Note: if module3 was not dispatched (no image), LangGraph still
    # proceeds — it only waits for nodes that were actually dispatched.

    # ── Step 5: Fixed linear pipeline after modules ────────────────
    # These always run in this order, no branching needed
    builder.add_edge("prompt_builder", "generator")
    builder.add_edge("generator",      "save_memory")
    builder.add_edge("save_memory",    END)   # graph complete

    # ── Step 6: Compile the graph ──────────────────────────────────
    # compile() locks the graph structure — no more add_node/add_edge after this.
    # MemorySaver checkpointer = saves state between runs for the MCQ
    # pause-and-resume pattern. Without it, graph.invoke() can't resume
    # a paused conversation.
    checkpointer = MemorySaver()
    graph = builder.compile(checkpointer=checkpointer)

    print("[Graph] ✅ Pipeline compiled — parallel module execution enabled")
    return graph


# ─────────────────────────────────────────────────────────────────
# SINGLETONS — compiled once, reused for every request
# ─────────────────────────────────────────────────────────────────

_graph = None   # compiled graph, initialised on first call

def get_graph():
    """
    Lazy-compile the graph singleton.
    build_graph() is expensive (~1s). We call it once and cache the result.
    Thread-safe for single-worker deployments (our Docker setup).
    """
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


# ─────────────────────────────────────────────────────────────────
# PUBLIC API — the single function FastAPI routers call
# ─────────────────────────────────────────────────────────────────

def run_pipeline(
    query:       str,
    role:        str,
    session_id:  Optional[str] = None,
    image_b64:   Optional[str] = None,
    mcq_answers: Optional[dict] = None,
) -> dict:
    """
    Main entry point called by FastAPI for every user message.

    Handles two scenarios:
      1. Fresh query       → runs full pipeline from intent_parser
      2. MCQ continuation  → resumes with mcq_answers, skips re-parsing

    Args:
        query       : the user's message text
        role        : "brand" or "creator"
        session_id  : conversation ID — generate on first call, send back
                      with every subsequent call to maintain memory
        image_b64   : base64-encoded image string (optional)
        mcq_answers : {field_name: selected_option} from the MCQ UI

    Returns:
        {
          response      : assistant's text (or MCQ question text)
          session_id    : send this back on every subsequent request
          needs_mcq     : True if pipeline paused for user input
          mcq_questions : list of questions if needs_mcq=True
          metadata      : latency, module stats, cost estimate
        }
    """
    # Generate session_id on first message — client stores and returns it
    if not session_id:
        session_id = str(uuid.uuid4())

    graph = get_graph()

    # Build the initial state — every key in CampaynState must be present
    # Keys not yet known are initialised to None / [] / {}
    initial_state: CampaynState = {
        "query":           query,
        "role":            role,
        "session_id":      session_id,
        "image_b64":       image_b64,
        "filter_spec":     None,
        "needs_mcq":       False,
        "mcq_questions":   None,
        "mcq_answers":     mcq_answers,   # None on first call, filled on MCQ resume
        "intent_result":   None,
        "module1_block":   None,
        "module1_meta":    None,
        "module2_block":   None,
        "module2_meta":    None,
        "module3_block":   None,
        "module3_meta":    None,
        "prompt":          None,
        "response":        None,
        "parsed_response": None,
        "history":         [],
        "trace_id":        str(uuid.uuid4()),
        "node_latency":    {},
    }

    # thread_id ties this run to a conversation session.
    # MemorySaver uses it to save/restore state between calls.
    # Same session_id = same conversation = state is remembered.
    config = {"configurable": {"thread_id": session_id}}

    # graph.invoke() runs the full pipeline synchronously.
    # It blocks until END is reached and returns the final state.
    t0 = time.time()
    final_state = graph.invoke(initial_state, config=config)
    total_ms    = round((time.time() - t0) * 1000, 1)

    # ── MCQ interrupt path ────────────────────────────────────────
    # Graph stopped early — needs_mcq=True means the user must answer
    # questions before the modules can run. Return formatted MCQ to UI.
    if final_state.get("needs_mcq"):
        from api.intent_parser import format_mcq_for_chat, MCQQuestion
        questions = [
            MCQQuestion(
                field   = q["field"],
                label   = q["label"],
                options = q["options"],
                hint    = q.get("hint", ""),
            )
            for q in (final_state.get("mcq_questions") or [])
        ]
        return {
            "response":      format_mcq_for_chat(questions),
            "session_id":    session_id,
            "needs_mcq":     True,
            "mcq_questions": final_state.get("mcq_questions"),
            "metadata": {
                "total_ms":     total_ms,
                "node_latency": final_state.get("node_latency", {}),
            },
        }

    # ── Normal response path ──────────────────────────────────────
    parsed = final_state.get("parsed_response") or {}
    m1     = final_state.get("module1_meta") or {}
    m2     = final_state.get("module2_meta") or {}
    m3     = final_state.get("module3_meta") or {}
    lat    = final_state.get("node_latency", {})

    return {
        "response":      parsed.get("text", final_state.get("response", "")),
        "session_id":    session_id,
        "needs_mcq":     False,
        "mcq_questions": None,
        "metadata": {
            "total_ms":        total_ms,
            "node_latency":    lat,
            # Module stats for LangSmith + API response
            "module1_matches": m1.get("total_matches",  0),
            "module2_chunks":  m2.get("chunks_returned", 0),
            "rag_confidence":  m2.get("confidence",     "unknown"),
            "vision_used":     m3.get("vision_used",    False),
            "sources":         parsed.get("sources",    []),
            # Parallel execution summary
            "parallel_savings_ms": max(0, (
                lat.get("module1", 0) +
                lat.get("module2", 0) +
                lat.get("module3", 0)
            ) - max(
                lat.get("module1", 0),
                lat.get("module2", 0),
                lat.get("module3", 0),
            )),
        },
    }


# ─────────────────────────────────────────────────────────────────
# SMOKE TEST
#
# Run this file directly to test the full pipeline end-to-end:
#   python graph.py
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_cases = [
        {
            "query":      "Find top fitness micro-influencers for a protein supplement launch",
            "role":       "brand",
            "session_id": "test-brand-001",
        },
        {
            "query":      "How do I grow my Instagram from 10k to 50k followers?",
            "role":       "creator",
            "session_id": "test-creator-001",
        },
        {
            "query":      "I want someone who vibes with gym culture",
            "role":       "brand",
            "session_id": "test-brand-002",
        },
    ]

    print("=" * 65)
    print("  CAMPAYN PIPELINE — PARALLEL EXECUTION SMOKE TEST")
    print("=" * 65)

    for tc in test_cases:
        print(f"\nROLE : {tc['role'].upper()}")
        print(f"QUERY: {tc['query']}")
        print("-" * 65)

        result = run_pipeline(**tc)

        if result["needs_mcq"]:
            print("MCQ triggered:")
            print(result["response"])
        else:
            print(f"Response (first 300 chars):")
            print(result["response"][:300] + "...")
            meta = result["metadata"]
            print(f"\nLatency breakdown:")
            for node, ms in meta.get("node_latency", {}).items():
                print(f"  {node:<20} {ms}ms")
            print(f"  {'TOTAL':<20} {meta['total_ms']}ms")
            savings = meta.get("parallel_savings_ms", 0)
            if savings > 0:
                print(f"  Parallel saved    ~{savings}ms vs sequential")