"""
check_gemini_models.py
======================
Lists all Gemini embedding models available to your API key.
Uses the NEW google.genai SDK (v1.x) — the only supported SDK as of 2025.

Usage:
    pip install google-genai
    python check_gemini_models.py
"""

import os
from dotenv import load_dotenv
from google import genai

load_dotenv()   # read GEMINI_API_KEY from .env when run standalone
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
client = genai.Client(api_key=GEMINI_API_KEY)

print("=== ALL GEMINI MODELS SUPPORTING EMBEDDING ===\n")

embedding_models = []

for model in client.models.list():
    # supported_actions tells us what this model can do
    supported = getattr(model, "supported_actions", []) or []
    name      = model.name.lower()

    if "embedcontent" in [a.lower() for a in supported] \
       or "embed" in name \
       or "embedding" in name:
        embedding_models.append(model.name)
        print(f"  ✅ {model.name}")
        print(f"     Display name     : {getattr(model, 'display_name', 'N/A')}")
        print(f"     Supported actions: {supported}")
        print()

# If nothing matched the filter, print everything so we can inspect manually
if not embedding_models:
    print("No models matched embedding filter.\n")
    print("=== ALL AVAILABLE MODELS (for manual inspection) ===\n")
    for model in client.models.list():
        print(f"  {model.name}")
        print(f"     actions : {getattr(model, 'supported_actions', 'N/A')}")
        print()