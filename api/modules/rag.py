"""
rag.py
======
Campayn AI Assistant — Module 2: Knowledge Base RAG Retrieval

WHAT THIS MODULE DOES
----------------------
Given a user query, embeds it using the SAME model used during ingestion
(models/gemini-embedding-2), searches ChromaDB, and returns the most
semantically relevant chunks from the Campayn knowledge base.

CRITICAL RULE — ONE MODEL FOR EVERYTHING
-----------------------------------------
Documents were embedded with models/gemini-embedding-2 during ingestion.
Queries MUST also be embedded with models/gemini-embedding-2 here.
Using a different model for queries would put them in a different
mathematical space — similarity search would return garbage results.

SDK
---
Uses the NEW google.genai SDK (v1.x).
The old google-generativeai SDK is deprecated. Do NOT import it.

Author : Mohit Chauhan — Campayn Multimodal AI Assistant
"""

# ─────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────

import os
import time
from dataclasses import dataclass
from functools import lru_cache       # cache expensive connections — init once
from typing import Optional

import chromadb                       # pip install chromadb
from google import genai              # pip install google-genai  ← NEW SDK only
from google.genai import types        # types.EmbedContentConfig

# ─────────────────────────────────────────────────────────────────
# CONFIGURATION — must match ingest_knowledge_base.py exactly
# ─────────────────────────────────────────────────────────────────

CHROMA_PERSIST_DIR  = os.environ.get("CHROMA_PERSIST_DIR",     "data/vector_store")
CHROMA_COLLECTION   = os.environ.get("CHROMA_COLLECTION_NAME", "campayn_knowledge_base")
GEMINI_API_KEY      = os.environ.get("GEMINI_API_KEY",         "")

# ── MUST match the model used in ingest_knowledge_base.py ────────
GEMINI_EMBED_MODEL  = "models/gemini-embedding-2"

# How many final chunks to return to the Prompt Constructor
DEFAULT_TOP_K       = int(os.environ.get("MAX_RAG_CHUNKS", "5"))

# We fetch (top_k × 3) from ChromaDB then re-rank down to top_k.
# Over-fetching gives the role-aware re-ranker more candidates to work with,
# which improves the quality of the final top_k set.
RETRIEVAL_MULTIPLIER = 3

# Chunks with similarity below this score are dropped entirely.
# 0 = completely unrelated, 1 = identical.
# 0.30 means "must share at least some meaningful content with the query"
MIN_SIMILARITY      = 0.30


# ─────────────────────────────────────────────────────────────────
# DATA CLASS — one retrieved chunk with its scores
# ─────────────────────────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    """
    One chunk returned by the vector search, with its metadata and scores.

    similarity  : raw cosine similarity from ChromaDB (0–1, higher = more relevant)
    final_score : after role-aware re-ranking (may differ from similarity)
    """
    chunk_id:      str
    text:          str
    section_title: str
    part_number:   str
    part_name:     str
    role_tags:     list[str]
    similarity:    float
    final_score:   float = 0.0


# ─────────────────────────────────────────────────────────────────
# CACHED CONNECTIONS
# @lru_cache(maxsize=1) means the decorated function runs only ONCE
# and returns the cached object on every subsequent call.
# This avoids reconnecting to ChromaDB or re-loading the client
# on every single query — critical for low-latency production use.
# ─────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_collection() -> chromadb.Collection:
    """Connect to the ChromaDB collection on disk. Cached after first call."""
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    try:
        collection = client.get_collection(CHROMA_COLLECTION)
        print(f"[Module 2] ChromaDB: {collection.count()} chunks loaded")
        return collection
    except Exception as e:
        raise RuntimeError(
            f"ChromaDB collection \'{CHROMA_COLLECTION}\' not found at \'{CHROMA_PERSIST_DIR}\'\n"
            f"Run: python ingest_knowledge_base.py --reset\n"
            f"Error: {e}"
        )


@lru_cache(maxsize=1)
def _get_gemini_client() -> genai.Client:
    """
    Create the google.genai Client. Cached after first call.

    This is the new-style client from the google.genai SDK (v1.x).
    It replaces ALL of the old google-generativeai patterns:
      Old: genai.configure(api_key=...)  → gone
      Old: genai.embed_content(...)      → replaced by client.models.embed_content(...)
    """
    client = genai.Client(api_key=GEMINI_API_KEY)
    print(f"[Module 2] Gemini client ready — embed model: {GEMINI_EMBED_MODEL}")
    return client


# ─────────────────────────────────────────────────────────────────
# CORE RETRIEVAL
# ─────────────────────────────────────────────────────────────────

def retrieve_chunks(
    query:       str,
    role:        str = "brand",
    top_k:       int = DEFAULT_TOP_K,
    part_filter: Optional[list[str]] = None,
) -> tuple[list[RetrievedChunk], dict]:
    """
    Main retrieval function — the heart of Module 2.

    Steps:
      1. Embed the query with gemini-embedding-2 (RETRIEVAL_QUERY task)
      2. Search ChromaDB for the closest chunk vectors
      3. Drop chunks below MIN_SIMILARITY threshold
      4. Apply soft role-aware re-ranking
      5. Return top_k chunks sorted by final_score

    Args:
        query       : user's natural language question
        role        : "brand" or "creator" — used for soft re-ranking
        top_k       : number of chunks to return
        part_filter : optional list of part numbers to restrict search
                      e.g. ["1", "2"] → only search CPV + niche atlas

    Returns:
        chunks   : list[RetrievedChunk], sorted by final_score desc
        metadata : dict of retrieval stats for LangSmith tracing
    """

    # ── Step 1: Embed the query ───────────────────────────────────────────
    # task_type="RETRIEVAL_QUERY" is the correct type for search-time queries.
    # task_type="RETRIEVAL_DOCUMENT" was used during ingestion.
    # Same model, different task_type = asymmetric embedding = better accuracy.
    client = _get_gemini_client()

    t0 = time.time()
    embed_response = client.models.embed_content(
        model    = GEMINI_EMBED_MODEL,
        contents = query,   # single string — same pattern as ingestion fix
        config   = types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    query_vector = embed_response.embeddings[0].values
    embed_ms = round((time.time() - t0) * 1000, 1)

    # ── Step 2: Search ChromaDB ───────────────────────────────────────────
    collection = _get_collection()

    # Over-fetch to give the re-ranker more candidates
    fetch_k = min(top_k * RETRIEVAL_MULTIPLIER, collection.count())

    # Optional where clause to restrict search to specific parts
    where = {"part_number": {"$in": part_filter}} if part_filter else None

    t1 = time.time()
    results = collection.query(
        query_embeddings = [query_vector],
        n_results        = fetch_k,
        where            = where,
        include          = ["documents", "metadatas", "distances"],
    )
    search_ms = round((time.time() - t1) * 1000, 1)

    # ── Step 3: Parse results and apply similarity threshold ──────────────
    raw_chunks: list[RetrievedChunk] = []

    for doc, meta, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        # ChromaDB returns cosine DISTANCE (0=identical, 2=opposite for cosine)
        # Convert to similarity: similarity = 1 - distance
        # For normalised cosine: distance range is [0, 2], similarity range [−1, 1]
        # In practice for text embeddings, similarity is always positive
        similarity = round(1.0 - distance, 4)

        if similarity < MIN_SIMILARITY:
            continue   # drop irrelevant chunks

        role_tags = [t.strip() for t in meta.get("role_tags", "brand,creator").split(",")]

        raw_chunks.append(RetrievedChunk(
            chunk_id      = meta.get("chunk_id", "unknown"),
            text          = doc,
            section_title = meta.get("section_title", ""),
            part_number   = meta.get("part_number",   "?"),
            part_name     = meta.get("part_name",     ""),
            role_tags     = role_tags,
            similarity    = similarity,
            final_score   = similarity,   # updated by re-ranker below
        ))

    # ── Step 4: Role-aware re-ranking ─────────────────────────────────────
    reranked = _rerank_by_role(raw_chunks, role)
    reranked.sort(key=lambda c: c.final_score, reverse=True)
    top_chunks = reranked[:top_k]

    # ── Step 5: Build metadata for observability ──────────────────────────
    metadata = {
        "query":           query,
        "role":            role,
        "embed_model":     GEMINI_EMBED_MODEL,
        "chunks_fetched":  len(raw_chunks),
        "chunks_returned": len(top_chunks),
        "embed_ms":        embed_ms,
        "search_ms":       search_ms,
        "top_similarity":  top_chunks[0].similarity if top_chunks else 0.0,
        "parts_retrieved": list({c.part_number for c in top_chunks}),
        "confidence":      _confidence(top_chunks),
    }

    print(f"[Module 2] {len(top_chunks)} chunks returned "
          f"(embed={embed_ms}ms  search={search_ms}ms  "
          f"top_sim={metadata['top_similarity']})")

    return top_chunks, metadata


# ─────────────────────────────────────────────────────────────────
# ROLE-AWARE RE-RANKER
# ─────────────────────────────────────────────────────────────────

def _rerank_by_role(chunks: list[RetrievedChunk], role: str) -> list[RetrievedChunk]:
    """
    Apply a soft role-relevance multiplier to each chunk's similarity score.

    SOFT (not hard) filtering — we never fully block chunks tagged for
    the other role. If a brand asks about creator growth mechanics, the
    answer lives in Part 6 (creator-tagged) and we still want to return it,
    just at a slightly lower rank than a brand-tagged chunk with equal similarity.

    Multipliers:
      Role match only    → ×1.15  (15% boost — clearly relevant to this role)
      Both roles         → ×1.00  (neutral — shared platform knowledge)
      Other role only    → ×0.90  (10% penalty — still retrievable if very relevant)
    """
    for chunk in chunks:
        if len(chunk.role_tags) > 1:          # tagged for both roles
            mult = 1.00
        elif role in chunk.role_tags:          # tagged for this role specifically
            mult = 1.15
        else:                                  # tagged for the other role
            mult = 0.90
        chunk.final_score = round(chunk.similarity * mult, 4)
    return chunks


def _confidence(chunks: list[RetrievedChunk]) -> str:
    """
    Translate top similarity score into a human-readable confidence label.
    Surfaced in the API response and LangSmith traces.
    """
    if not chunks:
        return "none"
    top = chunks[0].similarity
    if top >= 0.75: return "high"
    if top >= 0.50: return "medium"
    if top >= 0.30: return "low"
    return "none"


# ─────────────────────────────────────────────────────────────────
# PROMPT FORMATTER
# ─────────────────────────────────────────────────────────────────

def format_for_prompt(chunks: list[RetrievedChunk], metadata: dict) -> str:
    """
    Convert retrieved chunks into a compact string for the Prompt Constructor.

    Format injected into the LLM prompt:
        [Module 2 — Knowledge Base Context]
        Confidence: high  |  Chunks: 5  |  Parts: 1, 3

        ── [Part 1] The CPV Model Explained ─────────────
        CPV stands for cost-per-view...
        (similarity: 0.89)

        ── [Part 3] Meme Marketing for Dating Apps ──────
        Dating apps in India...
        (similarity: 0.71)
    """
    parts_str = ", ".join(sorted(metadata.get("parts_retrieved", [])))
    lines = [
        "[Module 2 — Knowledge Base Context]",
        f"Confidence: {metadata.get('confidence')}  |  "
        f"Chunks: {len(chunks)}  |  Parts: {parts_str}",
        "",
    ]

    for chunk in chunks:
        header = f"── [Part {chunk.part_number}] {chunk.section_title} "
        header = header + "─" * max(0, 60 - len(header))
        lines.append(header)
        lines.append(chunk.text)
        lines.append(f"(similarity: {chunk.similarity})")
        lines.append("")

    if metadata.get("confidence") in ("low", "none"):
        lines.append("⚠ Low confidence — knowledge base may not cover this topic specifically.")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────
# PUBLIC API — called by LangGraph node
# ─────────────────────────────────────────────────────────────────

def run_module2(
    query:       str,
    role:        str = "brand",
    top_k:       int = DEFAULT_TOP_K,
    part_filter: Optional[list[str]] = None,
) -> tuple[str, dict]:
    """
    Single entry point for the LangGraph orchestrator node.

    Returns:
        prompt_block : formatted string ready for Prompt Constructor
        metadata     : retrieval stats for LangSmith tracing
    """
    chunks, metadata = retrieve_chunks(query, role, top_k, part_filter)
    return format_for_prompt(chunks, metadata), metadata


# ─────────────────────────────────────────────────────────────────
# SMOKE TEST
# python -m api.modules.rag "how does CPV work" brand
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    test_queries = [
        ("brand",   "how does CPV work and what budget should I set"),
        ("brand",   "which meme pages work for food delivery campaigns"),
        ("creator", "how do I grow my Instagram followers faster"),
        ("creator", "when do I get my Creator Coin payout"),
        ("brand",   "what CPV should I expect for fitness creators"),
        ("creator", "my reel hook is weak how do I fix it"),
    ]

    print("=" * 65)
    print("  MODULE 2 — RAG RETRIEVAL SMOKE TEST")
    print(f"  Embedding model: {GEMINI_EMBED_MODEL}")
    print("=" * 65)

    for role, query in test_queries:
        print(f"\nROLE : {role.upper()}")
        print(f"QUERY: {query}")
        print("-" * 65)

        chunks, meta = retrieve_chunks(query, role)

        print(f"Confidence : {meta['confidence']}")
        print(f"Parts hit  : {meta['parts_retrieved']}")
        print(f"Latency    : embed={meta['embed_ms']}ms  search={meta['search_ms']}ms")
        print("Top chunks:")
        for i, c in enumerate(chunks, 1):
            preview = c.text.replace("\n", " ")[:100]
            print(f"  {i}. [Part {c.part_number}] {c.section_title} "
                  f"(sim={c.similarity}  final={c.final_score})")
            print(f"     {preview}...")