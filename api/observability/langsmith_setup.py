"""
langsmith_setup.py
==================
Campayn AI Assistant — LangSmith Observability

WHAT THIS FILE DOES
--------------------
Sets up LangSmith tracing so every pipeline run is logged with:
  - Per-node latency (how long each step took)
  - Token usage and estimated cost per run
  - Input/output for every node (for debugging)
  - Evaluation datasets for RAG quality testing
  - Custom feedback scores (thumbs up/down from users)

WHY LANGSMITH
-------------
Without observability you're flying blind. LangSmith gives you:
  - A trace for every single run → see exactly what happened
  - Cost tracking → know how much each query costs
  - Eval datasets → measure RAG quality over time
  - A/B prompt testing → compare prompt versions

HOW IT WORKS
------------
LangSmith works through the LANGCHAIN_TRACING_V2 environment variable.
When it's set to "true", LangChain and LangGraph automatically send
traces to LangSmith without any code changes.

For custom spans (non-LangChain code like our Gemini calls), we use
the @traceable decorator to manually instrument functions.

SETUP
-----
1. Create a free account at https://smith.langchain.com
2. Get your API key from Settings
3. Add to .env:
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_API_KEY=ls__...
   LANGCHAIN_PROJECT=campayn-ai-assistant

Author : Mohit Chauhan — Campayn Multimodal AI Assistant
"""

from __future__ import annotations

import os
import time
from functools import wraps
from typing import Any, Callable, Optional

# ─────────────────────────────────────────────────────────────────
# ENVIRONMENT SETUP
# LangSmith reads these env vars automatically
# Set them here as a safety net in case .env wasn't loaded yet
# ─────────────────────────────────────────────────────────────────

def setup_langsmith() -> bool:
    """
    Configure LangSmith environment variables and verify the connection.

    Call this once at app startup (in api/main.py).

    Returns True if LangSmith is enabled, False if disabled or misconfigured.
    """
    api_key = os.environ.get("LANGCHAIN_API_KEY", "")
    project = os.environ.get("LANGCHAIN_PROJECT", "campayn-ai-assistant")
    tracing = os.environ.get("LANGCHAIN_TRACING_V2", "false").lower()

    if not api_key:
        print("[LangSmith] ⚠ LANGCHAIN_API_KEY not set — tracing disabled")
        print("[LangSmith]   Add to .env: LANGCHAIN_API_KEY=ls__...")
        return False

    if tracing != "true":
        print("[LangSmith] ⚠ LANGCHAIN_TRACING_V2 != true — tracing disabled")
        return False

    # These are read by LangChain/LangGraph automatically
    os.environ["LANGCHAIN_TRACING_V2"]  = "true"
    os.environ["LANGCHAIN_PROJECT"]     = project
    os.environ["LANGCHAIN_API_KEY"]     = api_key
    os.environ["LANGCHAIN_ENDPOINT"]    = os.environ.get(
        "LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com"
    )

    print(f"[LangSmith] ✅ Tracing enabled → project: '{project}'")
    print(f"[LangSmith]    View traces: https://smith.langchain.com/projects/{project}")
    return True


# ─────────────────────────────────────────────────────────────────
# CUSTOM TRACE METADATA
# Attach structured metadata to each run for filtering in LangSmith
# ─────────────────────────────────────────────────────────────────

def build_run_metadata(
    role:       str,
    session_id: str,
    query:      str,
    has_image:  bool = False,
    mcq_used:   bool = False,
) -> dict:
    """
    Build a metadata dict attached to every LangGraph run.
    Visible in LangSmith UI — use for filtering and analysis.

    Example use in graph.py:
        config = {
            "configurable": {"thread_id": session_id},
            "metadata": build_run_metadata(role, session_id, query)
        }
    """
    return {
        "role":        role,          # "brand" or "creator"
        "session_id":  session_id,    # for grouping a conversation
        "query_len":   len(query),    # rough complexity signal
        "has_image":   has_image,     # did user upload an image?
        "mcq_used":    mcq_used,      # did MCQ interrupt fire?
        "app_version": "0.1.0",
        "environment": os.environ.get("ENVIRONMENT", "development"),
    }


def build_run_tags(role: str, has_image: bool = False) -> list[str]:
    """
    Build tags for LangSmith run filtering.
    Tags show up as filter chips in the LangSmith UI.
    """
    tags = [f"role:{role}", "campayn-v1"]
    if has_image:
        tags.append("multimodal")
    if os.environ.get("ENVIRONMENT") == "production":
        tags.append("production")
    else:
        tags.append("development")
    return tags


# ─────────────────────────────────────────────────────────────────
# @traceable DECORATOR — instrument non-LangChain functions
#
# LangGraph nodes are traced automatically by LangSmith because
# LangGraph is a LangChain product.
#
# But our Gemini API calls (in intent_parser.py, api/modules/vision.py)
# are NOT LangChain — they use the raw google.genai SDK.
# We need to manually wrap them to appear in traces.
# ─────────────────────────────────────────────────────────────────

def make_traceable(name: str, run_type: str = "llm"):
    """
    Decorator factory that wraps a function in a LangSmith trace span.

    Args:
        name     : span name shown in LangSmith (e.g. "gemini_intent_parse")
        run_type : "llm" | "chain" | "retriever" | "tool"

    Usage:
        @make_traceable("gemini_embed", run_type="retriever")
        def embed_query(text: str) -> list[float]:
            ...

    When LangSmith is disabled (no API key), the decorator is a no-op
    and the function runs normally — no performance penalty.
    """
    def decorator(fn: Callable) -> Callable:
        # Check if langsmith is available
        try:
            from langsmith import traceable
            # langsmith.traceable wraps the function in a trace span
            return traceable(name=name, run_type=run_type)(fn)
        except ImportError:
            # langsmith not installed — return function unchanged
            return fn

    return decorator


# ─────────────────────────────────────────────────────────────────
# COST ESTIMATION
# Gemini pricing as of 2025 — update if pricing changes
# ─────────────────────────────────────────────────────────────────

# Cost per 1000 tokens in USD (approximate)
GEMINI_COSTS = {
    "gemini-2.0-flash": {
        "input":  0.000075,   # $0.075 per 1M input tokens
        "output": 0.000300,   # $0.300 per 1M output tokens
    },
    "models/gemini-embedding-2": {
        "input":  0.000010,   # embeddings are very cheap
        "output": 0.0,
    },
}


def estimate_cost(
    model:         str,
    input_tokens:  int,
    output_tokens: int = 0,
) -> dict:
    """
    Estimate the USD cost of a Gemini API call.

    Returns a dict with input_cost, output_cost, total_cost.
    Attach this to LangSmith run metadata for cost tracking.

    Note: Token counts are rough estimates (1 token ≈ 4 chars).
    Use actual token counts from the API response when available.
    """
    pricing = GEMINI_COSTS.get(model, {"input": 0.0001, "output": 0.0003})

    input_cost  = (input_tokens  / 1000) * pricing["input"]
    output_cost = (output_tokens / 1000) * pricing["output"]

    return {
        "model":        model,
        "input_tokens": input_tokens,
        "output_tokens":output_tokens,
        "input_cost":   round(input_cost,  8),
        "output_cost":  round(output_cost, 8),
        "total_cost":   round(input_cost + output_cost, 8),
        "currency":     "USD",
    }


def estimate_tokens(text: str) -> int:
    """
    Rough token count estimate: 1 token ≈ 4 characters for English.
    Use this when the API doesn't return actual token counts.
    """
    return max(1, len(text) // 4)


# ─────────────────────────────────────────────────────────────────
# FEEDBACK LOGGING
# Record user thumbs up/down feedback to LangSmith
# Enables eval dataset creation from real user signals
# ─────────────────────────────────────────────────────────────────

def log_feedback(
    run_id:  str,
    score:   float,          # 1.0 = positive, 0.0 = negative
    comment: str = "",
    key:     str = "user_feedback",
) -> bool:
    """
    Log user feedback (thumbs up/down) to LangSmith.

    This turns user feedback into evaluation data — over time,
    you can use this to build eval datasets and measure quality.

    Args:
        run_id  : the LangSmith run ID from the pipeline response
        score   : 1.0 = thumbs up, 0.0 = thumbs down
        comment : optional user comment
        key     : feedback category (shown in LangSmith)

    Returns True if feedback was logged successfully.
    """
    try:
        from langsmith import Client
        client = Client()
        client.create_feedback(
            run_id   = run_id,
            key      = key,
            score    = score,
            comment  = comment,
        )
        print(f"[LangSmith] Feedback logged: run={run_id[:8]}... score={score}")
        return True
    except Exception as e:
        print(f"[LangSmith] Feedback log failed: {e}")
        return False


# ─────────────────────────────────────────────────────────────────
# RAG EVALUATION HELPERS
# Create eval datasets in LangSmith for measuring retrieval quality
# ─────────────────────────────────────────────────────────────────

# Sample Q&A pairs for RAG evaluation
# These are ground-truth examples the RAG system should get right
RAG_EVAL_DATASET = [
    {
        "question": "What is CPV and how does it differ from CPM?",
        "expected_parts": ["1"],   # Part 1 should be retrieved
        "expected_keywords": ["cost-per-view", "impressions", "views"],
    },
    {
        "question": "What CPV should I target for a beauty campaign?",
        "expected_parts": ["1", "2"],
        "expected_keywords": ["0.10", "0.20", "beauty", "skincare"],
    },
    {
        "question": "How do I grow from 10k to 50k followers on Instagram?",
        "expected_parts": ["6"],
        "expected_keywords": ["algorithm", "niche", "consistency", "hook"],
    },
    {
        "question": "When do creators get paid their Creator Coins?",
        "expected_parts": ["1", "8"],
        "expected_keywords": ["creator coins", "KYC", "payout", "UPI"],
    },
    {
        "question": "What meme pages work for food delivery brands?",
        "expected_parts": ["3"],
        "expected_keywords": ["meme", "food delivery", "CPV", "0.08"],
    },
]


def create_eval_dataset(dataset_name: str = "campayn-rag-eval") -> bool:
    """
    Create a LangSmith evaluation dataset from our ground-truth Q&A pairs.

    Run this once after setup. The dataset appears in LangSmith under
    'Datasets & Testing' and can be used to run automated evals.

    Usage:
        python langsmith_setup.py --create-dataset
    """
    try:
        from langsmith import Client
        client = Client()

        # Check if dataset already exists
        datasets = list(client.list_datasets(dataset_name=dataset_name))
        if datasets:
            print(f"[LangSmith] Dataset '{dataset_name}' already exists")
            return True

        # Create dataset
        dataset = client.create_dataset(
            dataset_name = dataset_name,
            description  = "Ground-truth Q&A pairs for Campayn RAG evaluation",
        )

        # Add examples
        for item in RAG_EVAL_DATASET:
            client.create_example(
                inputs   = {"question": item["question"]},
                outputs  = {
                    "expected_parts":    item["expected_parts"],
                    "expected_keywords": item["expected_keywords"],
                },
                dataset_id = dataset.id,
            )

        print(f"[LangSmith] ✅ Created dataset '{dataset_name}' "
              f"with {len(RAG_EVAL_DATASET)} examples")
        return True

    except Exception as e:
        print(f"[LangSmith] Dataset creation failed: {e}")
        return False


# ─────────────────────────────────────────────────────────────────
# PIPELINE METRICS — attach to every run response
# ─────────────────────────────────────────────────────────────────

def compute_pipeline_metrics(pipeline_result: dict) -> dict:
    """
    Extract and compute summary metrics from a pipeline result dict.
    These are logged to LangSmith and returned in the API response.

    Args:
        pipeline_result : the dict returned by run_pipeline() in graph.py

    Returns a flat metrics dict suitable for LangSmith metadata.
    """
    meta     = pipeline_result.get("metadata", {})
    latency  = meta.get("node_latency", {})

    # Total pipeline latency
    total_ms = meta.get("total_ms", 0)

    # Estimate cost from the final generation call
    response_text = pipeline_result.get("response", "")
    prompt_text   = ""  # we don't store the full prompt in the result

    gen_cost = estimate_cost(
        model         = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash"),
        input_tokens  = estimate_tokens(prompt_text),
        output_tokens = estimate_tokens(response_text),
    )

    return {
        # Latency breakdown
        "total_ms":          total_ms,
        "intent_parser_ms":  latency.get("intent_parser", 0),
        "module1_ms":        latency.get("module1", 0),
        "module2_ms":        latency.get("module2", 0),
        "module3_ms":        latency.get("module3", 0),
        "prompt_builder_ms": latency.get("prompt_builder", 0),
        "generator_ms":      latency.get("generator", 0),

        # Retrieval quality
        "rag_confidence":    meta.get("rag_confidence", "unknown"),
        "module1_matches":   meta.get("module1_matches", 0),
        "module2_chunks":    meta.get("module2_chunks", 0),
        "vision_used":       meta.get("vision_used", False),

        # Cost
        "estimated_cost_usd": gen_cost["total_cost"],
        "output_tokens_est":  gen_cost["output_tokens"],
    }


# ─────────────────────────────────────────────────────────────────
# CLI — run this file directly to set up LangSmith
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv
    load_dotenv()

    print("=" * 55)
    print("  LANGSMITH SETUP")
    print("=" * 55)

    enabled = setup_langsmith()

    if "--create-dataset" in sys.argv and enabled:
        print("\nCreating RAG evaluation dataset...")
        create_eval_dataset()

    print("\nDone. Start the API with: uvicorn api.main:app --reload")