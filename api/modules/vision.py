"""
vision.py
=========
Campayn AI Assistant — Module 3: Vision Analysis

WHAT THIS MODULE DOES
----------------------
Accepts an image from the user (as base64, URL, or file path),
sends it to Gemini 2.0 Flash with a role-aware analysis prompt,
and returns structured feedback that the Prompt Constructor
injects into the final LLM response.

TWO ANALYSIS MODES (driven by role)
-------------------------------------
BRAND MODE — evaluating a creator's content for campaign fit:
  • Visual style consistency with a category
  • Production quality (lighting, framing, focus)
  • On-screen text legibility
  • Brand safety signals (nothing problematic in frame)
  • Aesthetic alignment with the brand's positioning tier

CREATOR MODE — giving feedback on your own content:
  • Hook strength (does the first frame stop a scroll?)
  • Lighting and framing quality
  • On-screen text readability on a small phone screen
  • CTA clarity and placement
  • Specific, actionable improvement suggestions

THREE INPUT TYPES SUPPORTED
-----------------------------
  1. base64 string  — image uploaded through Streamlit / FastAPI
  2. URL string     — direct link to image (Instagram CDN, etc.)
  3. File path      — local file path (for testing / scripts)

OUTPUT
------
Returns a structured dict + formatted string for the Prompt Constructor.
The LLM never sees the raw image directly — it sees Module 3's
analysis as text context, keeping the final prompt efficient.

SDK
---
Uses google.genai (v1.x) — same SDK as intent_parser.py.
Model: gemini-2.0-flash (multimodal — handles text + image in one call).

Author : Mohit Chauhan — Campayn Multimodal AI Assistant
"""

# ─────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────

import base64          # decode base64 image strings from the UI
import json            # parse structured JSON from Gemini response
import mimetypes       # detect image type from file extension
import os
import re
import time
import urllib.request  # fetch images from URLs without extra dependencies
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional

from google import genai                   # pip install google-genai
from google.genai import types             # types.Part, types.Content etc.

# ─────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────

GEMINI_API_KEY    = os.environ.get("GEMINI_API_KEY", "")

# gemini-2.0-flash: fast, cheap, multimodal — handles image + text in one call
# Same model used by intent_parser.py for text generation
VISION_MODEL      = os.environ.get("GEMINI_VISION_MODEL", "gemini-2.0-flash")

# Max image size we'll process (bytes) — 10 MB
# Gemini's limit is 20 MB; we use 10 MB to stay safe
MAX_IMAGE_BYTES   = 10 * 1024 * 1024

# Supported image MIME types
SUPPORTED_TYPES   = {"image/jpeg", "image/png", "image/webp", "image/gif"}

# Score labels for structured output
SCORE_LABELS      = {5: "Excellent", 4: "Good", 3: "Average", 2: "Weak", 1: "Poor"}


# ─────────────────────────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────────────────────────

@dataclass
class VisionAnalysis:
    """
    Structured output from a single image analysis.

    Scores are 1–5 integers.
    Flags are booleans for quick programmatic checks.
    All text fields are plain English, beginner-readable.
    """
    # ── Shared fields (both roles) ────────────────────────────────
    role:             str              # "brand" or "creator"
    overall_score:    int              # 1–5 overall quality score
    summary:          str              # 1-sentence summary of the image
    top_strength:     str              # the single best thing about this content
    top_weakness:     str              # the single most important thing to fix

    # ── Production quality (both roles) ──────────────────────────
    lighting_score:   int              # 1–5
    lighting_note:    str              # specific observation
    framing_score:    int              # 1–5
    framing_note:     str

    # ── Text readability (both roles) ─────────────────────────────
    text_present:     bool             # is there on-screen text in the image?
    text_score:       int              # 1–5, or 0 if no text present
    text_note:        str

    # ── Hook strength (both roles, critical for Reels) ────────────
    hook_score:       int              # 1–5 — would this stop a scroll?
    hook_note:        str              # why / why not

    # ── BRAND-specific fields ─────────────────────────────────────
    brand_safety:     bool             # True = safe, False = potential issue
    brand_safety_note:str              # what was flagged (empty if safe)
    aesthetic_fit:    int              # 1–5 — fits brand category visually
    aesthetic_note:   str

    # ── CREATOR-specific fields ───────────────────────────────────
    cta_present:      bool             # is there a clear call to action?
    cta_score:        int              # 1–5, or 0 if no CTA
    cta_note:         str

    # ── Actionable suggestions (both roles) ───────────────────────
    suggestions:      list[str] = field(default_factory=list)  # top 3 specific fixes

    # ── Metadata ─────────────────────────────────────────────────
    model_used:       str = VISION_MODEL
    analysis_ms:      float = 0.0
    confidence:       str = "high"     # "high" / "medium" / "low"
    error:            str = ""         # non-empty if analysis failed


# ─────────────────────────────────────────────────────────────────
# GEMINI CLIENT — cached, initialised once
# ─────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_client() -> genai.Client:
    """
    Create the Gemini client. Cached after first call.
    Same pattern as api/modules/rag.py.
    """
    client = genai.Client(api_key=GEMINI_API_KEY)
    print(f"[Module 3] Gemini client ready — vision model: {VISION_MODEL}")
    return client


# ─────────────────────────────────────────────────────────────────
# IMAGE LOADING
# Accepts three input types: base64 string, URL, or file path
# Returns (raw_bytes, mime_type) ready for the Gemini API
# ─────────────────────────────────────────────────────────────────

def _load_image(image_input: str) -> tuple[bytes, str]:
    """
    Load image bytes from any of the three supported input types.

    Args:
        image_input : one of:
            - base64 string (may include data URI prefix)
            - URL starting with http:// or https://
            - local file path

    Returns:
        (image_bytes, mime_type)

    Raises:
        ValueError if the input type can't be determined or image is too large
    """

    # ── Type 1: Base64 string ─────────────────────────────────────────────
    # Streamlit and FastAPI send uploaded images as base64
    # Sometimes includes a data URI prefix: "data:image/jpeg;base64,/9j/..."
    if image_input.startswith("data:"):
        # Strip the "data:image/jpeg;base64," prefix
        header, encoded = image_input.split(",", 1)
        mime_type = header.split(":")[1].split(";")[0]  # "image/jpeg"
        image_bytes = base64.b64decode(encoded)

    elif _looks_like_base64(image_input):
        # Raw base64 without the data URI prefix
        # We don't know the mime type — assume JPEG (most common)
        image_bytes = base64.b64decode(image_input)
        mime_type = "image/jpeg"

    # ── Type 2: URL ───────────────────────────────────────────────────────
    elif image_input.startswith(("http://", "https://")):
        image_bytes, mime_type = _fetch_url(image_input)

    # ── Type 3: Local file path ───────────────────────────────────────────
    else:
        path = Path(image_input)
        if not path.exists():
            raise ValueError(f"File not found: {image_input}")

        image_bytes = path.read_bytes()
        # Detect MIME type from file extension
        mime_type, _ = mimetypes.guess_type(str(path))
        mime_type = mime_type or "image/jpeg"

    # ── Size guard ────────────────────────────────────────────────────────
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise ValueError(
            f"Image too large: {len(image_bytes) / 1024 / 1024:.1f} MB "
            f"(max {MAX_IMAGE_BYTES // 1024 // 1024} MB)"
        )

    # ── Type guard ────────────────────────────────────────────────────────
    if mime_type not in SUPPORTED_TYPES:
        raise ValueError(
            f"Unsupported image type: {mime_type}. "
            f"Supported: {', '.join(SUPPORTED_TYPES)}"
        )

    return image_bytes, mime_type


def _looks_like_base64(s: str) -> bool:
    """
    Heuristic check: is this string a raw base64 payload?
    Base64 strings are long, use only A-Z a-z 0-9 + / = chars,
    and don't look like URLs or file paths.
    """
    if len(s) < 100:
        return False
    if "/" in s[:10] or "\\" in s[:10]:
        return False   # looks like a path
    return bool(re.match(r"^[A-Za-z0-9+/=]+$", s[:100]))


def _fetch_url(url: str) -> tuple[bytes, str]:
    """
    Download an image from a URL using only stdlib urllib.
    No requests library needed.
    """
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0"},   # some CDNs block blank user-agents
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            content_type = resp.headers.get("Content-Type", "image/jpeg")
            # Content-Type may be "image/jpeg; charset=utf-8" — take first part
            mime_type = content_type.split(";")[0].strip()
            image_bytes = resp.read()
        return image_bytes, mime_type
    except Exception as e:
        raise ValueError(f"Failed to fetch image from URL: {e}")


# ─────────────────────────────────────────────────────────────────
# ANALYSIS PROMPTS — role-specific instructions for Gemini
# ─────────────────────────────────────────────────────────────────

BRAND_ANALYSIS_PROMPT = """
You are a senior creator marketing strategist at Campayn, an Indian D2C influencer platform.
A brand has shared this image — either a creator's post or profile screenshot — to evaluate
whether this creator is a good fit for their campaign.

Analyse the image across these dimensions and return ONLY valid JSON. No markdown, no preamble.

{
  "overall_score": <1-5 integer>,
  "summary": "<one sentence describing what is in the image>",
  "top_strength": "<the single best thing about this content for brand marketing>",
  "top_weakness": "<the single most important concern for a brand considering this creator>",

  "lighting_score": <1-5>,
  "lighting_note": "<specific observation about lighting quality>",

  "framing_score": <1-5>,
  "framing_note": "<specific observation about composition and framing>",

  "text_present": <true|false>,
  "text_score": <1-5 or 0 if no text>,
  "text_note": "<is on-screen text readable on a small phone screen?>",

  "hook_score": <1-5>,
  "hook_note": "<would this first frame stop a scroll? why or why not?>",

  "brand_safety": <true if safe, false if any concern>,
  "brand_safety_note": "<describe any brand safety issues, or 'No issues detected'>",

  "aesthetic_fit": <1-5>,
  "aesthetic_note": "<does the visual style suit a quality D2C brand campaign?>",

  "cta_present": <true|false>,
  "cta_score": <1-5 or 0 if no CTA>,
  "cta_note": "<is the call to action clear and well-placed?>",

  "suggestions": [
    "<specific actionable suggestion 1 for the brand evaluating this creator>",
    "<specific actionable suggestion 2>",
    "<specific actionable suggestion 3>"
  ],

  "confidence": "<high|medium|low based on image clarity and content visibility>"
}

Scoring guide:
  5 = Excellent — professional quality, no issues
  4 = Good — minor issues, still effective
  3 = Average — noticeable issues that reduce effectiveness
  2 = Weak — significant issues limiting brand value
  1 = Poor — would damage brand reputation or campaign

Be specific and constructive. Reference actual elements visible in the image.
Avoid generic comments like "good lighting" — say WHY it is good or what specifically to fix.
"""


CREATOR_ANALYSIS_PROMPT = """
You are the Campayn Creator Coach — a supportive but honest content strategist
helping an Indian Instagram creator improve their content quality.

The creator has shared one of their posts or a screenshot of their content.
Analyse it across these dimensions and return ONLY valid JSON. No markdown, no preamble.

{
  "overall_score": <1-5 integer>,
  "summary": "<one sentence describing what is in the image>",
  "top_strength": "<the single best thing the creator is doing well>",
  "top_weakness": "<the single most important thing to fix to improve performance>",

  "lighting_score": <1-5>,
  "lighting_note": "<specific observation — what exactly is good or needs fixing>",

  "framing_score": <1-5>,
  "framing_note": "<specific observation about composition, subject placement, background>",

  "text_present": <true|false>,
  "text_score": <1-5 or 0 if no text>,
  "text_note": "<is the on-screen text readable on a small phone screen? font size, contrast, placement>",

  "hook_score": <1-5>,
  "hook_note": "<would this first frame stop someone scrolling? what specifically does or doesn't work?>",

  "brand_safety": <true if content is appropriate for brand partnerships, false if concerns>,
  "brand_safety_note": "<any content that would prevent brand partnerships, or 'No issues'>",

  "aesthetic_fit": <1-5>,
  "aesthetic_note": "<does the overall aesthetic feel professional and consistent with a clear niche?>",

  "cta_present": <true|false>,
  "cta_score": <1-5 or 0 if no CTA visible>,
  "cta_note": "<is there a clear call to action? is it specific, friction-light, well-placed?>",

  "suggestions": [
    "<specific actionable improvement 1 — be precise, not generic>",
    "<specific actionable improvement 2>",
    "<specific actionable improvement 3>"
  ],

  "confidence": "<high|medium|low based on image clarity>"
}

Tone: honest and supportive. Not harsh, not falsely positive.
Be specific — reference actual elements visible in the image.
Bad suggestion: "improve your lighting"
Good suggestion: "move the ring light 30cm closer and slightly to the left to eliminate the shadow under your chin"
"""


def _get_prompt(role: str, context: str = "") -> str:
    """
    Return the correct analysis prompt for the given role.
    Optionally append extra context (e.g. brand category, creator niche).
    """
    base = BRAND_ANALYSIS_PROMPT if role == "brand" else CREATOR_ANALYSIS_PROMPT
    if context:
        return base + f"\n\nADDITIONAL CONTEXT PROVIDED:\n{context}"
    return base


# ─────────────────────────────────────────────────────────────────
# CORE ANALYSIS FUNCTION
# ─────────────────────────────────────────────────────────────────

def analyse_image(
    image_input: str,
    role:        str = "brand",
    context:     str = "",
) -> VisionAnalysis:
    """
    Main function — analyse an image and return structured feedback.

    Args:
        image_input : base64 string, URL, or local file path
        role        : "brand" or "creator"
        context     : optional extra context injected into the prompt
                      e.g. "Brand category: fitness supplements, target: Micro tier"

    Returns:
        VisionAnalysis dataclass with all scores and notes populated
    """
    t0 = time.time()

    # ── Step 1: Load the image ────────────────────────────────────────────
    try:
        image_bytes, mime_type = _load_image(image_input)
    except ValueError as e:
        # Return a graceful error analysis instead of crashing
        return VisionAnalysis(
            role=role, overall_score=0, summary="", top_strength="",
            top_weakness="", lighting_score=0, lighting_note="",
            framing_score=0, framing_note="", text_present=False,
            text_score=0, text_note="", hook_score=0, hook_note="",
            brand_safety=True, brand_safety_note="", aesthetic_fit=0,
            aesthetic_note="", cta_present=False, cta_score=0, cta_note="",
            suggestions=[], confidence="low", error=str(e),
        )

    # ── Step 2: Build the multimodal request ─────────────────────────────
    # The new google.genai SDK sends image + text in one call.
    # types.Part.from_bytes() wraps raw bytes into the correct Part format.
    # We send [image_part, text_part] as the contents list.
    prompt = _get_prompt(role, context)

    image_part = types.Part.from_bytes(
        data      = image_bytes,
        mime_type = mime_type,
    )
    text_part = types.Part.from_text(text=prompt)

    # ── Step 3: Call Gemini vision ────────────────────────────────────────
    client = _get_client()

    try:
        response = client.models.generate_content(
            model    = VISION_MODEL,
            contents = types.Content(
                role  = "user",
                parts = [image_part, text_part],
            ),
            config = types.GenerateContentConfig(
                temperature       = 0.1,  # low temp = consistent structured output
                response_mime_type= "application/json",
            ),
        )
        raw_text = response.text.strip()

    except Exception as gemini_error:
        # Gemini vision failed — retry once against Anthropic Claude
        # before giving up (see api/llm_fallback.py).
        try:
            print(f"[Module 3] Gemini vision failed ({gemini_error}) — retrying with Anthropic...")
            from api.llm_fallback import call_anthropic_vision
            raw_text = call_anthropic_vision(
                image_bytes, mime_type, prompt, temperature=0.1, json_mode=True,
            ).strip()
        except Exception as anthropic_error:
            analysis_ms = round((time.time() - t0) * 1000, 1)
            return VisionAnalysis(
                role=role, overall_score=0, summary="", top_strength="",
                top_weakness="", lighting_score=0, lighting_note="",
                framing_score=0, framing_note="", text_present=False,
                text_score=0, text_note="", hook_score=0, hook_note="",
                brand_safety=True, brand_safety_note="", aesthetic_fit=0,
                aesthetic_note="", cta_present=False, cta_score=0, cta_note="",
                suggestions=[], confidence="low",
                analysis_ms=analysis_ms,
                error=f"Gemini API error: {gemini_error} | Anthropic fallback also failed: {anthropic_error}",
            )

    # ── Step 4: Parse JSON response ───────────────────────────────────────
    # Strip markdown fences if model adds them despite mime type setting
    if raw_text.startswith("```"):
        raw_text = raw_text.split("```")[1]
        if raw_text.startswith("json"):
            raw_text = raw_text[4:]
        raw_text = raw_text.strip()

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as e:
        analysis_ms = round((time.time() - t0) * 1000, 1)
        return VisionAnalysis(
            role=role, overall_score=0, summary=raw_text[:200], top_strength="",
            top_weakness="", lighting_score=0, lighting_note="",
            framing_score=0, framing_note="", text_present=False,
            text_score=0, text_note="", hook_score=0, hook_note="",
            brand_safety=True, brand_safety_note="", aesthetic_fit=0,
            aesthetic_note="", cta_present=False, cta_score=0, cta_note="",
            suggestions=[], confidence="low",
            analysis_ms=analysis_ms, error=f"JSON parse error: {e}",
        )

    # ── Step 5: Build VisionAnalysis from parsed JSON ─────────────────────
    analysis_ms = round((time.time() - t0) * 1000, 1)

    analysis = VisionAnalysis(
        role              = role,
        overall_score     = int(data.get("overall_score",   3)),
        summary           = data.get("summary",             ""),
        top_strength      = data.get("top_strength",        ""),
        top_weakness      = data.get("top_weakness",        ""),
        lighting_score    = int(data.get("lighting_score",  3)),
        lighting_note     = data.get("lighting_note",       ""),
        framing_score     = int(data.get("framing_score",   3)),
        framing_note      = data.get("framing_note",        ""),
        text_present      = bool(data.get("text_present",   False)),
        text_score        = int(data.get("text_score",      0)),
        text_note         = data.get("text_note",           ""),
        hook_score        = int(data.get("hook_score",      3)),
        hook_note         = data.get("hook_note",           ""),
        brand_safety      = bool(data.get("brand_safety",   True)),
        brand_safety_note = data.get("brand_safety_note",   "No issues detected"),
        aesthetic_fit     = int(data.get("aesthetic_fit",   3)),
        aesthetic_note    = data.get("aesthetic_note",      ""),
        cta_present       = bool(data.get("cta_present",    False)),
        cta_score         = int(data.get("cta_score",       0)),
        cta_note          = data.get("cta_note",            ""),
        suggestions       = data.get("suggestions",         []),
        confidence        = data.get("confidence",          "medium"),
        model_used        = VISION_MODEL,
        analysis_ms       = analysis_ms,
        error             = "",
    )

    print(f"[Module 3] Analysis complete: overall={analysis.overall_score}/5 "
          f"role={role} time={analysis_ms}ms confidence={analysis.confidence}")

    return analysis


# ─────────────────────────────────────────────────────────────────
# OUTPUT FORMATTER — for the Prompt Constructor
# ─────────────────────────────────────────────────────────────────

def format_for_prompt(analysis: VisionAnalysis) -> str:
    """
    Convert VisionAnalysis into a compact, token-efficient string
    for injection into the LLM prompt by the Prompt Constructor.

    The LLM reads this text block to inform its response —
    it never sees the raw image directly.
    """
    if analysis.error:
        return (
            f"[Module 3 — Vision Analysis]\n"
            f"⚠ Analysis failed: {analysis.error}\n"
            f"Proceeding without image context."
        )

    def score_label(s: int) -> str:
        return SCORE_LABELS.get(s, "N/A")

    lines = [
        "[Module 3 — Vision Analysis]",
        f"Model: {analysis.model_used}  |  "
        f"Confidence: {analysis.confidence}  |  "
        f"Time: {analysis.analysis_ms}ms",
        "",
        f"Summary     : {analysis.summary}",
        f"Overall     : {analysis.overall_score}/5 — {score_label(analysis.overall_score)}",
        f"Top strength: {analysis.top_strength}",
        f"Top fix     : {analysis.top_weakness}",
        "",
        "── Production Quality ────────────────────────────────────",
        f"Lighting : {analysis.lighting_score}/5 — {analysis.lighting_note}",
        f"Framing  : {analysis.framing_score}/5 — {analysis.framing_note}",
    ]

    # On-screen text
    if analysis.text_present:
        lines.append(f"Text     : {analysis.text_score}/5 — {analysis.text_note}")
    else:
        lines.append("Text     : None detected in frame")

    lines += [
        "",
        "── Engagement Signals ────────────────────────────────────",
        f"Hook     : {analysis.hook_score}/5 — {analysis.hook_note}",
    ]

    # CTA
    if analysis.cta_present:
        lines.append(f"CTA      : {analysis.cta_score}/5 — {analysis.cta_note}")
    else:
        lines.append("CTA      : Not visible in frame")

    lines += [
        "",
        "── Brand Signals ─────────────────────────────────────────",
        f"Brand safety : {'✅ Safe' if analysis.brand_safety else '⚠ Concern'} — {analysis.brand_safety_note}",
        f"Aesthetic fit: {analysis.aesthetic_fit}/5 — {analysis.aesthetic_note}",
        "",
        "── Top 3 Suggestions ─────────────────────────────────────",
    ]
    for i, suggestion in enumerate(analysis.suggestions[:3], start=1):
        lines.append(f"  {i}. {suggestion}")

    return "\n".join(lines)


def format_metadata(analysis: VisionAnalysis) -> dict:
    """
    Return a JSON-serialisable metadata dict for LangSmith tracing
    and LangGraph state storage.
    """
    return {
        "role":           analysis.role,
        "overall_score":  analysis.overall_score,
        "hook_score":     analysis.hook_score,
        "brand_safety":   analysis.brand_safety,
        "confidence":     analysis.confidence,
        "model_used":     analysis.model_used,
        "analysis_ms":    analysis.analysis_ms,
        "error":          analysis.error,
        "vision_used":    analysis.error == "",
    }


# ─────────────────────────────────────────────────────────────────
# PUBLIC API — called by the LangGraph orchestrator node
# ─────────────────────────────────────────────────────────────────

def run_module3(
    image_input: str,
    role:        str = "brand",
    context:     str = "",
) -> tuple[str, dict]:
    """
    Single entry point for the LangGraph node.

    Args:
        image_input : base64 string, URL, or local file path
        role        : "brand" or "creator"
        context     : optional extra context for the analysis prompt
                      e.g. "Brand category: beauty. Target tier: Micro."

    Returns:
        prompt_block : formatted string for the Prompt Constructor
        metadata     : dict for LangSmith tracing / LangGraph state
    """
    analysis     = analyse_image(image_input, role, context)
    prompt_block = format_for_prompt(analysis)
    metadata     = format_metadata(analysis)

    return prompt_block, metadata


# ─────────────────────────────────────────────────────────────────
# SMOKE TEST
# python -m api.modules.vision <path_to_image.jpg> brand
# python -m api.modules.vision <path_to_image.jpg> creator
# python -m api.modules.vision https://example.com/image.jpg brand
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    # if len(sys.argv) < 2:
    #     print("Usage: python -m api.modules.vision <data/mohit.jpeg> [brand|creator]")
    #     print("Examples:")
    #     print("  python -m api.modules.vision test_image.jpg brand")
    #     print("  python -m api.modules.vision test_image.jpg creator")
    #     print("  python -m api.modules.vision https://example.com/post.jpg brand")
    #     sys.exit(1)

    image_input = "data/mohit.jpeg"
    role        = "brand"
    context     = sys.argv[3] if len(sys.argv) > 3 else ""

    print("=" * 60)
    print(f"  MODULE 3 — VISION ANALYSIS")
    print(f"  Model : {VISION_MODEL}")
    print(f"  Role  : {role}")
    print(f"  Input : {image_input[:60]}...")
    print("=" * 60)

    prompt_block, metadata = run_module3(image_input, role, context)

    print("\n--- Prompt Block (injected into LLM prompt) ---")
    print(prompt_block)
    print("\n--- Metadata (for LangSmith tracing) ---")
    for k, v in metadata.items():
        print(f"  {k}: {v}")