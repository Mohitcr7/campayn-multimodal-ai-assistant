"""
main.py
=======
Campayn AI Assistant — FastAPI Application Entry Point

WHAT THIS FILE DOES
--------------------
Creates the FastAPI app, registers all routers, adds middleware,
and handles startup/shutdown events (loading models, connecting to DB).

This is the file uvicorn runs:
    uvicorn api.main:app --reload        # development
    uvicorn api.main:app --host 0.0.0.0  # production

HOW FASTAPI WORKS (beginner explanation)
-----------------------------------------
FastAPI is a Python web framework. It:
  1. Receives HTTP requests (POST /brand/chat with a JSON body)
  2. Validates the request body using Pydantic schemas
  3. Calls the matching route handler function
  4. Returns the result as a JSON response

The @app.on_event("startup") decorator runs code ONCE when the
server starts — we use it to pre-load the DataFrame and ChromaDB
so the first request doesn't take 3 extra seconds.

Author : Mohit Chauhan — Campayn Multimodal AI Assistant
"""

from __future__ import annotations

import os
import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

# Load .env file before anything else
# This makes GEMINI_API_KEY, LANGCHAIN_API_KEY etc. available.
# override=True makes the .env file authoritative in local dev, so a stale
# or empty shell variable can't silently shadow the value in .env. In Docker
# there is no .env in the image (see .dockerignore), so this is a no-op there
# and the container's injected env vars are used as-is.
load_dotenv(override=True)

# ─────────────────────────────────────────────────────────────────
# APP CREATION
# ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "Campayn AI Assistant API",
    description = (
        "Multimodal AI assistant for the Campayn influencer marketing platform. "
        "Dual-role: brand-side campaign planning and creator-side content coaching. "
        "Built with LangGraph, Gemini, ChromaDB RAG, and LangSmith observability."
    ),
    version     = "0.1.0",
    docs_url    = "/docs",       # Swagger UI at /docs
    redoc_url   = "/redoc",      # ReDoc at /redoc
)


# ─────────────────────────────────────────────────────────────────
# CORS MIDDLEWARE
# Allows the Streamlit UI (localhost:8501) and any frontend
# to call this API from a browser without CORS errors
# ─────────────────────────────────────────────────────────────────

allowed_origins = os.environ.get(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,http://localhost:8501,http://127.0.0.1:8501"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins     = allowed_origins,
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ─────────────────────────────────────────────────────────────────
# REQUEST TIMING MIDDLEWARE
# Adds X-Process-Time header to every response
# Visible in browser dev tools and LangSmith
# ─────────────────────────────────────────────────────────────────

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """
    Middleware runs for EVERY request before and after the handler.
    We use it to measure total request time and add it as a header.
    """
    start = time.time()
    response = await call_next(request)
    process_ms = round((time.time() - start) * 1000, 1)
    response.headers["X-Process-Time-Ms"] = str(process_ms)
    return response


# ─────────────────────────────────────────────────────────────────
# GLOBAL EXCEPTION HANDLER
# Returns clean JSON errors instead of HTML error pages
# ─────────────────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Catch any unhandled exception and return a structured JSON response.
    In production this should also alert via Slack/PagerDuty.
    """
    print(f"[API] Unhandled exception on {request.url}: {exc}")
    return JSONResponse(
        status_code = 500,
        content     = {
            "error":   "Internal server error",
            "detail":  str(exc)[:200],
            "path":    str(request.url),
        },
    )


# ─────────────────────────────────────────────────────────────────
# STARTUP EVENT
# Pre-load expensive resources so the first request is fast
# ─────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    """
    Runs once when the server starts.

    We pre-load:
      1. LangSmith — enables tracing for all subsequent requests
      2. Creator profiles DataFrame — 56k rows, takes ~2s to load
      3. ChromaDB collection — connects to the vector store
      4. LangGraph pipeline — compiles the graph

    Without pre-loading, the first user request takes 5-10 extra seconds.
    After startup, these are cached in memory and reused.
    """
    print("\n" + "=" * 55)
    print("  CAMPAYN AI ASSISTANT — STARTING UP")
    print("=" * 55)

    # 1. LangSmith
    from api.observability.langsmith_setup import setup_langsmith
    setup_langsmith()

    # 2. Creator profiles (heavy — load now, not on first request)
    try:
        from api.graph import get_dataframe
        df = get_dataframe()
        print(f"[Startup] Creator profiles: {len(df):,} rows loaded")
    except Exception as e:
        print(f"[Startup] ⚠ Creator profiles failed: {e}")

    # 3. ChromaDB (connect and verify)
    try:
        import chromadb
        persist_dir  = os.environ.get("CHROMA_PERSIST_DIR", "data/vector_store")
        collection   = os.environ.get("CHROMA_COLLECTION_NAME", "campayn_knowledge_base")
        client       = chromadb.PersistentClient(path=persist_dir)
        col          = client.get_collection(collection)
        print(f"[Startup] ChromaDB: {col.count()} chunks ready")
    except Exception as e:
        print(f"[Startup] ⚠ ChromaDB failed: {e}")
        print(f"[Startup]   Run: python scripts/ingest_knowledge_base.py --reset")

    # 4. Compile LangGraph pipeline
    try:
        from api.graph import get_graph
        get_graph()
        print("[Startup] LangGraph pipeline: compiled ✅")
    except Exception as e:
        print(f"[Startup] ⚠ LangGraph compile failed: {e}")

    print("=" * 55)
    print("  READY — http://localhost:8000/docs")
    print("=" * 55 + "\n")


# ─────────────────────────────────────────────────────────────────
# SHUTDOWN EVENT
# ─────────────────────────────────────────────────────────────────

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    print("[Shutdown] Campayn AI Assistant shutting down...")


# ─────────────────────────────────────────────────────────────────
# REGISTER ROUTERS
# Each router adds its endpoints to the app
# ─────────────────────────────────────────────────────────────────

from api.routers import brand_router, creator_router, health_router, feedback_router

app.include_router(brand_router)
app.include_router(creator_router)
app.include_router(health_router)
app.include_router(feedback_router)


# ─────────────────────────────────────────────────────────────────
# ROOT ENDPOINT
# ─────────────────────────────────────────────────────────────────

@app.get("/", tags=["System"])
async def root():
    """Root endpoint — API info and links."""
    return {
        "name":        "Campayn AI Assistant API",
        "version":     "0.1.0",
        "docs":        "/docs",
        "health":      "/health",
        "endpoints": {
            "brand_chat":   "POST /brand/chat",
            "creator_chat": "POST /creator/chat",
            "feedback":     "POST /feedback",
        },
    }


# ─────────────────────────────────────────────────────────────────
# RUN DIRECTLY (for development without uvicorn CLI)
# python main.py
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host    = os.environ.get("API_HOST", "0.0.0.0"),
        port    = int(os.environ.get("API_PORT", "8000")),
        reload  = os.environ.get("ENVIRONMENT", "development") == "development",
        workers = 1,   # single worker for development
    )