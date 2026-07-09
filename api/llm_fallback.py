"""
llm_fallback.py
================
Campayn AI Assistant — Gemini → Anthropic Automatic Failover

WHAT THIS MODULE DOES
----------------------
Every LLM call in this project talks to Gemini first. If Gemini raises
(bad key, quota exhausted, outage, network error), callers retry the
same request against Anthropic Claude instead, so a Gemini incident
doesn't take the whole assistant down.

generate_text()        — tries Gemini, falls back to Anthropic on failure.
                          Used by api/graph.py's node_generator.
call_anthropic_text()  — Anthropic only, no Gemini attempt. Used both
                          internally by generate_text() and directly by
                          callers (api/intent_parser.py) that already
                          tried their own Gemini path and just need the
                          Anthropic completion.
call_anthropic_vision() — same idea for image + text prompts. Used by
                          api/modules/vision.py's except block.

This only covers text/vision generation. RAG embeddings (api/modules/rag.py)
stay Gemini-only — Anthropic has no embeddings API.

Author : Mohit Chauhan — Campayn Multimodal AI Assistant
"""

from __future__ import annotations

import base64
import os

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL   = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")

_JSON_ONLY_INSTRUCTION = (
    "\n\nRespond with ONLY a single valid JSON object. "
    "No markdown fences, no commentary before or after."
)


def _anthropic_client():
    import anthropic
    return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


def call_anthropic_text(
    prompt: str,
    *,
    temperature: float = 0.4,
    max_tokens: int = 1500,
    json_mode: bool = False,
) -> str:
    """
    Direct Anthropic text call — no Gemini attempt. Used both as the
    fallback path inside generate_text() and by callers (like
    api/intent_parser.py) that already tried their own Gemini path
    and just need the Anthropic completion.
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


def generate_text(
    prompt: str,
    *,
    temperature: float = 0.4,
    max_tokens: int = 1500,
    json_mode: bool = False,
) -> tuple[str, str]:
    """
    Generate text, preferring Gemini and falling back to Anthropic.

    Returns (text, provider_used) where provider_used is "gemini" or "anthropic".
    Raises the Anthropic exception if BOTH providers fail.
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
        print(f"[LLM Fallback] Gemini text generation failed ({gemini_error}) — retrying with Anthropic...")
        text = call_anthropic_text(prompt, temperature=temperature, max_tokens=max_tokens, json_mode=json_mode)
        print(f"[LLM Fallback] Anthropic ({ANTHROPIC_MODEL}) served the request")
        return text, "anthropic"


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
