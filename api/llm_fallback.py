"""
llm_fallback.py
================
Campayn AI Assistant — Multi-Provider Automatic Failover

WHAT THIS MODULE DOES
----------------------
Every LLM call in this project talks to Gemini first. If Gemini raises
(bad key, quota exhausted, outage, network error), callers retry the
same request down a chain of backups so a single-provider incident
doesn't take the whole assistant down.

TEXT FALLBACK CHAIN (in order):
  1. Google Gemini              — primary
  2. NVIDIA Nemotron 3 Ultra    — via OpenRouter (free tier)
  3. Anthropic Claude           — last resort (low credit balance)

VISION FALLBACK CHAIN:
  1. Google Gemini → 2. Anthropic Claude
  (Nemotron is text-only, so it is skipped for image analysis.)

Public functions:
  generate_text()         — Gemini → Nemotron → Claude. Returns (text, provider).
                            Used by api/graph.py's node_generator.
  fallback_text()         — Nemotron → Claude (no Gemini attempt). Used by
                            callers (api/intent_parser.py) that already tried
                            their own Gemini path and just need a backup.
  call_openrouter_text()  — OpenRouter/Nemotron only.
  call_anthropic_text()   — Anthropic only.
  call_anthropic_vision() — Anthropic only, image + text. Used by
                            api/modules/vision.py's except block.

Any provider whose API key is not configured is skipped automatically.
This only covers text/vision generation. RAG embeddings (api/modules/rag.py)
stay Gemini-only — none of the fallbacks expose an embeddings API here.

Author : Mohit Chauhan — Campayn Multimodal AI Assistant
"""

from __future__ import annotations

import base64
import os

# ── Provider config (all read from the environment) ───────────────
ANTHROPIC_API_KEY  = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL    = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL   = os.environ.get("OPENROUTER_MODEL", "nvidia/nemotron-3-ultra-550b-a55b:free")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

_JSON_ONLY_INSTRUCTION = (
    "\n\nRespond with ONLY a single valid JSON object. "
    "No markdown fences, no commentary before or after."
)


def _anthropic_client():
    import anthropic
    return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


def call_openrouter_text(
    prompt: str,
    *,
    temperature: float = 0.4,
    max_tokens: int = 1500,
    json_mode: bool = False,
) -> str:
    """
    Direct OpenRouter call — NVIDIA Nemotron 3 Ultra (free tier).
    OpenRouter is OpenAI-compatible, so we reuse the already-installed
    `openai` SDK and just point it at the OpenRouter base URL.
    """
    from openai import OpenAI

    client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=OPENROUTER_API_KEY)
    or_prompt = prompt + (_JSON_ONLY_INSTRUCTION if json_mode else "")
    response = client.chat.completions.create(
        model       = OPENROUTER_MODEL,
        max_tokens  = max_tokens,
        temperature = temperature,
        messages    = [{"role": "user", "content": or_prompt}],
        # Optional attribution headers OpenRouter recommends (harmless if unused).
        extra_headers = {
            "HTTP-Referer": "https://github.com/Mohitcr7/campayn-multimodal-ai-assistant",
            "X-Title":      "Campayn AI Assistant",
        },
    )
    return response.choices[0].message.content


def call_anthropic_text(
    prompt: str,
    *,
    temperature: float = 0.4,
    max_tokens: int = 1500,
    json_mode: bool = False,
) -> str:
    """
    Direct Anthropic text call — no Gemini/OpenRouter attempt.
    """
    client = _anthropic_client()
    anthropic_prompt = prompt + (_JSON_ONLY_INSTRUCTION if json_mode else "")
    response = client.messages.create(
        model       = ANTHROPIC_MODEL,
        max_tokens  = max_tokens,
        temperature = temperature,
        messages    = [{"role": "user", "content": anthropic_prompt}],
    )
    return response.content[0].text


def fallback_text(
    prompt: str,
    *,
    temperature: float = 0.4,
    max_tokens: int = 1500,
    json_mode: bool = False,
) -> tuple[str, str]:
    """
    The non-Gemini text chain: NVIDIA Nemotron (OpenRouter) → Anthropic Claude.

    Used by generate_text()'s fallback path and directly by callers that
    already tried Gemini themselves (e.g. api/intent_parser.py). A provider
    with no API key configured is skipped. Returns (text, provider_used).
    Raises the last provider's exception if every configured provider fails.
    """
    # 2nd choice — NVIDIA Nemotron via OpenRouter (free tier)
    if OPENROUTER_API_KEY:
        try:
            text = call_openrouter_text(prompt, temperature=temperature, max_tokens=max_tokens, json_mode=json_mode)
            print(f"[LLM Fallback] OpenRouter ({OPENROUTER_MODEL}) served the request")
            return text, "openrouter"
        except Exception as or_error:
            print(f"[LLM Fallback] OpenRouter/Nemotron failed ({or_error}) — falling back to Anthropic...")
    else:
        print("[LLM Fallback] OPENROUTER_API_KEY not set — skipping Nemotron, trying Anthropic...")

    # 3rd choice — Anthropic Claude (last resort)
    text = call_anthropic_text(prompt, temperature=temperature, max_tokens=max_tokens, json_mode=json_mode)
    print(f"[LLM Fallback] Anthropic ({ANTHROPIC_MODEL}) served the request")
    return text, "anthropic"


def generate_text(
    prompt: str,
    *,
    temperature: float = 0.4,
    max_tokens: int = 1500,
    json_mode: bool = False,
) -> tuple[str, str]:
    """
    Generate text with automatic failover: Gemini → Nemotron → Claude.

    Returns (text, provider_used). Raises if every configured provider fails.
    """
    from api.graph import GEMINI_MODEL, get_gemini_client
    from google.genai import types

    try:
        client = get_gemini_client()
        response = client.models.generate_content(
            model    = GEMINI_MODEL,
            contents = prompt,
            config   = types.GenerateContentConfig(
                temperature       = temperature,
                max_output_tokens = max_tokens,
                **({"response_mime_type": "application/json"} if json_mode else {}),
            ),
        )
        return response.text, "gemini"

    except Exception as gemini_error:
        print(f"[LLM Fallback] Gemini text generation failed ({gemini_error}) — trying fallback chain...")
        return fallback_text(prompt, temperature=temperature, max_tokens=max_tokens, json_mode=json_mode)


def call_anthropic_vision(
    image_bytes: bytes,
    mime_type: str,
    prompt: str,
    *,
    temperature: float = 0.1,
    max_tokens: int = 1500,
    json_mode: bool = True,
) -> str:
    """
    Direct Anthropic vision call — no Gemini attempt. Claude's vision
    models accept a base64 image content block alongside the text prompt.
    """
    client = _anthropic_client()
    anthropic_prompt = prompt + (_JSON_ONLY_INSTRUCTION if json_mode else "")
    response = client.messages.create(
        model       = ANTHROPIC_MODEL,
        max_tokens  = max_tokens,
        temperature = temperature,
        messages    = [{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type":       "base64",
                        "media_type": mime_type,
                        "data":       base64.b64encode(image_bytes).decode("utf-8"),
                    },
                },
                {"type": "text", "text": anthropic_prompt},
            ],
        }],
    )
    return response.content[0].text
