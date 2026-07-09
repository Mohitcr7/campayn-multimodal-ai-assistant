"""
ingest_knowledge_base.py
========================
Campayn AI Assistant — Knowledge Base Ingestion Script

WHAT THIS SCRIPT DOES
----------------------
Reads the Campayn knowledge base text file, splits it into chunks,
embeds each chunk using Gemini Embedding 2, and stores everything
in a ChromaDB vector database on disk.

Run this script ONCE before starting the app.
Re-run it with --reset whenever the knowledge base content changes.

EMBEDDING MODEL
---------------
Uses models/gemini-embedding-2 via the NEW google.genai SDK (v1.x).
The old google-generativeai SDK is deprecated — do NOT use it.

USAGE
-----
  python ingest_knowledge_base.py
  python ingest_knowledge_base.py --kb path/to/knowledge_base.txt
  python ingest_knowledge_base.py --reset   # wipe and rebuild from scratch

Author : Mohit Chauhan — Campayn Multimodal AI Assistant
"""

# ─────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────

import argparse
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import chromadb                    # pip install chromadb
from google import genai           # pip install google-genai  ← NEW SDK
from google.genai import types     # types.EmbedContentConfig etc.

# ─────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────

CHROMA_PERSIST_DIR  = os.environ.get("CHROMA_PERSIST_DIR",       "data/vector_store")
CHROMA_COLLECTION   = os.environ.get("CHROMA_COLLECTION_NAME",   "campayn_knowledge_base")
DEFAULT_KB_PATH     = os.environ.get("KNOWLEDGE_BASE_PATH",      "data/raw/knowledge_base.txt")
GEMINI_API_KEY      = os.environ.get("GEMINI_API_KEY",           "")

# ── EMBEDDING MODEL ───────────────────────────────────────────────
# models/gemini-embedding-2  — latest stable Gemini embedding model
# Confirmed available via client.models.list() on your API key.
# MUST be the same model used in api/modules/rag.py for queries.
# Using different models for documents vs queries BREAKS retrieval.
GEMINI_EMBED_MODEL  = "models/gemini-embedding-2"

# Gemini-embedding-2 output dimension
EMBEDDING_DIM       = 3072

# Batch size — Gemini allows up to 100 texts per embed_content call
# We use 50 to stay safely within token limits per batch
EMBED_BATCH_SIZE    = 50

# Pause between batches to respect API rate limits
BATCH_DELAY_SECONDS = 0.5

# ─────────────────────────────────────────────────────────────────
# PART METADATA
# ─────────────────────────────────────────────────────────────────

PART_NAMES = {
    "1": "Platform Foundations and CPV Economics",
    "2": "The India Niche Atlas",
    "3": "The Meme Marketing Playbook",
    "4": "Strategic Marketing Frameworks",
    "5": "Category Strategy Modules",
    "6": "Creator Growth and Monetization",
    "7": "Diagnostic Frameworks",
    "8": "Operations Content Craft and Compliance",
    "9": "AI Assistant Behavior Attribution and Closing",
}

# Which parts are most relevant for each role.
# Stored as metadata on every chunk so Module 2 can soft-boost by role.
PART_ROLE_RELEVANCE = {
    "1": ["brand", "creator"],   # CPV basics — both sides need this
    "2": ["brand"],              # niche atlas — brand campaign planning
    "3": ["brand"],              # meme playbook — brand strategy
    "4": ["brand", "creator"],   # marketing frameworks — both benefit
    "5": ["brand"],              # category modules — brand planning
    "6": ["creator"],            # creator growth — creator-side only
    "7": ["brand", "creator"],   # diagnostics — both sides
    "8": ["brand", "creator"],   # operations — both sides
    "9": ["brand", "creator"],   # AI behaviour — both sides
}


# ─────────────────────────────────────────────────────────────────
# DATA CLASS
# ─────────────────────────────────────────────────────────────────

@dataclass
class KBChunk:
    """One parsed section from the knowledge base — the unit we embed and store."""
    chunk_id:      str        # "chunk_0042"
    text:          str        # full text embedded into ChromaDB
    section_title: str        # first line of the block (topic heading)
    part_number:   str        # "1" – "9"
    part_name:     str        # human-readable part name
    role_tags:     list[str]  # ["brand"] | ["creator"] | ["brand", "creator"]
    char_count:    int        # length for diagnostics


# ─────────────────────────────────────────────────────────────────
# STEP 1 — LOAD THE KNOWLEDGE BASE FILE
# ─────────────────────────────────────────────────────────────────

def load_knowledge_base(kb_path: Path) -> str:
    """
    Read the knowledge base file from disk.
    Supports .txt (plain text export) and .docx (Word document).

    We use utf-8-sig encoding for .txt to automatically strip the
    BOM character (\ufeff / \xef\xbb\xbf) that PDF-to-text exporters
    sometimes add at the very start of the file.
    """
    if not kb_path.exists():
        print(f"❌ File not found: {kb_path}")
        sys.exit(1)

    if kb_path.suffix.lower() == ".txt":
        text = kb_path.read_text(encoding="utf-8-sig")
        print(f"✅ Loaded text file: {kb_path} ({len(text):,} characters)")
        return text

    else:
        print(f"❌ Unsupported file type: {kb_path.suffix}. Use .txt")
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────
# STEP 2 — PARSE INTO CHUNKS
# ─────────────────────────────────────────────────────────────────

def parse_into_chunks(raw_text: str) -> list[KBChunk]:
    """
    Split the knowledge base into semantic chunks.

    ACTUAL FILE STRUCTURE (plain text, no markdown)
    ────────────────────────────────────────────────
    PART 1 — PLATFORM FOUNDATIONS AND CPV ECONOMICS   ← part divider
                                                       ← blank line
    Campayn Platform Overview                          ← section TITLE
    Campayn is an AI-powered...                        ← section BODY line 1
    brands. Campayn connects...                        ← section BODY line 2
                                                       ← blank line = chunk boundary
    What Makes Campayn Different                       ← next section TITLE
    Most creator marketing platforms...                ← next section BODY
    ...

    Page 4 of 60          ← NOISE — strip these
    Campayn Knowledge Base ← NOISE — strip these

    KEY INSIGHT: The PDF export wraps long lines at ~80 chars and injects
    page-break noise mid-paragraph. We strip the noise first, then
    collapse the resulting blank clusters, then split on blank lines.
    Each resulting block = title (line 1) + body (remaining lines).
    """

    # ── 2a. Strip noise lines ─────────────────────────────────────────────
    noise_patterns = [
        r"^Page \d+ of \d+$",                           # "Page 4 of 60"
        r"^Campayn Knowledge Base$",                     # repeated page header
        r"^CAMPAYN$",                                    # title page artefact
        r"^Knowledge Base for the Multimodal AI Assistant$",
        r"^The Strategist Edition$",
        r"^CPV economics, niche atlas.*$",
        r"^Prepared for$",
        r"^Mohit Chauhan$",
        r"^Multimodal AI Chatbot Project$",
    ]
    cleaned = []
    for line in raw_text.split("\n"):
        stripped = line.strip()
        if not any(re.match(p, stripped) for p in noise_patterns):
            cleaned.append(stripped)

    # ── 2b. Collapse consecutive blank lines into one ─────────────────────
    collapsed, prev_blank = [], False
    for line in cleaned:
        is_blank = (line == "")
        if is_blank and prev_blank:
            continue
        collapsed.append(line)
        prev_blank = is_blank

    # ── 2c. Split on blank lines → raw blocks ────────────────────────────
    blocks = re.split(r"\n\n+", "\n".join(collapsed))

    # ── 2d. Parse each block ──────────────────────────────────────────────
    chunks: list[KBChunk] = []
    current_part: Optional[str] = None   # skip preamble until first PART
    chunk_idx = 0

    PREAMBLE_SKIP = {
        "How To Use This Document",
        "Document Structure",
        "Document Notes for Mohit",
    }

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        # PART header → update current part, don't create a chunk
        part_m = re.match(r"PART\s+(\d+)\s*[—\-]+\s*(.+)", block, re.IGNORECASE)
        if part_m:
            current_part = part_m.group(1)
            continue

        # Everything before the first PART header is preamble — skip
        if current_part is None:
            continue

        lines = [l.strip() for l in block.split("\n") if l.strip()]
        if not lines:
            continue

        title = lines[0]
        body  = " ".join(lines[1:])   # rejoin wrapped PDF lines into one sentence flow

        # Skip structural noise
        if title in PREAMBLE_SKIP:
            continue
        if re.match(r"Part \d+\s*[—\-]", title):
            continue
        if not body:
            continue
        if len(title) > 120 or title.endswith("."):
            # Real section titles are short and don't end with a period
            continue

        full_text = f"{title}\n\n{body}"
        chunks.append(KBChunk(
            chunk_id      = f"chunk_{chunk_idx:04d}",
            text          = full_text,
            section_title = title,
            part_number   = current_part,
            part_name     = PART_NAMES.get(current_part, "Unknown"),
            role_tags     = PART_ROLE_RELEVANCE.get(current_part, ["brand", "creator"]),
            char_count    = len(full_text),
        ))
        chunk_idx += 1

    if not chunks:
        print("❌ No chunks parsed — check PART N — headers exist in the file.")
        return []

    avg = sum(c.char_count for c in chunks) // len(chunks)
    print(f"✅ Parsed {len(chunks)} chunks from {len(PART_NAMES)} parts")
    print(f"   Avg chunk size: {avg:,} characters")
    return chunks


# ─────────────────────────────────────────────────────────────────
# STEP 3 — EMBED WITH GEMINI (new google.genai SDK)
# ─────────────────────────────────────────────────────────────────

def generate_embeddings(chunks: list[KBChunk]) -> list[list[float]]:
    """
    Embed all chunks using models/gemini-embedding-2 via the new google.genai SDK.

    WHY ONE MODEL ONLY
    ──────────────────
    This is the ONLY embedding function — no fallbacks to sentence-transformers.
    Using two different models (one for docs, one for queries) means the
    vectors live in different mathematical spaces and similarity search breaks
    completely. We enforce one model here and in api/modules/rag.py.

    HOW THE NEW SDK WORKS
    ──────────────────────
    Old SDK (deprecated):  genai.embed_content(model=..., content=..., task_type=...)
    New SDK (current):     client.models.embed_content(model=..., contents=..., config=...)

    task_type="RETRIEVAL_DOCUMENT" tells the model these are documents being
    stored for later search (vs RETRIEVAL_QUERY for search-time queries).
    This asymmetric embedding significantly improves retrieval accuracy.
    """
    # Initialise the new-style Gemini client once
    client = genai.Client(api_key=GEMINI_API_KEY)

    texts       = [chunk.text for chunk in chunks]
    all_vectors: list[list[float]] = []

    total_batches = (len(texts) + EMBED_BATCH_SIZE - 1) // EMBED_BATCH_SIZE

    print(f"\n📐 Embedding {len(texts)} chunks with {GEMINI_EMBED_MODEL}...")

    for batch_num, batch_start in enumerate(range(0, len(texts), EMBED_BATCH_SIZE), start=1):
        batch = texts[batch_start : batch_start + EMBED_BATCH_SIZE]

        print(f"   Batch {batch_num}/{total_batches} — {len(batch)} chunks...")

        # ── WHY WE LOOP CHUNK BY CHUNK ────────────────────────────────────
        # The google.genai SDK's embed_content() treats contents as a SINGLE
        # content item when passed a list — it returns ONE embedding for the
        # entire list, not one per string.
        #
        # To get one embedding per chunk we must call embed_content() once
        # per string. We group these calls into batches (EMBED_BATCH_SIZE)
        # and sleep between batches to respect the API rate limit.
        #
        # Future option: use asyncBatchEmbedContent() for true parallel
        # batching — the model supports it (shown in check_gemini_models.py).
        for text in batch:
            response = client.models.embed_content(
                model    = GEMINI_EMBED_MODEL,
                contents = text,             # ONE string at a time
                config   = types.EmbedContentConfig(
                    task_type = "RETRIEVAL_DOCUMENT",
                    # RETRIEVAL_DOCUMENT = these chunks are being stored for search
                    # RETRIEVAL_QUERY    = used at query time in api/modules/rag.py
                    # Using correct task_type per context improves retrieval accuracy
                ),
            )
            # response.embeddings[0].values = the vector for this single chunk
            all_vectors.append(response.embeddings[0].values)

        print(f"   ✓ Batch {batch_num} done — {len(all_vectors)} total vectors so far")

        # Pause between batches to respect API rate limits
        if batch_start + EMBED_BATCH_SIZE < len(texts):
            time.sleep(BATCH_DELAY_SECONDS)

    print(f"✅ Embeddings generated: {len(all_vectors)} vectors, "
          f"{len(all_vectors[0])} dimensions each")
    return all_vectors


# ─────────────────────────────────────────────────────────────────
# STEP 4 — STORE IN CHROMADB
# ─────────────────────────────────────────────────────────────────

def get_chroma_collection(reset: bool = False) -> chromadb.Collection:
    """
    Connect to (or create) the ChromaDB collection on disk.

    WHY WE DELETE THE DIRECTORY ON RESET (not just the collection)
    ───────────────────────────────────────────────────────────────
    ChromaDB's Rust backend stores HNSW index files on disk alongside
    the collection metadata. When the embedding dimension changes
    (e.g. from 384 → 3072 after switching models), the old index files
    remain on disk even after delete_collection() is called.

    The PersistentClient reads the on-disk HNSW index on startup and
    locks the dimension to whatever it finds there. So calling
    delete_collection() + get_or_create_collection() still inherits
    the old 384-dim constraint from the surviving index files.

    Fix: on reset, delete the entire persist directory so ChromaDB
    starts with a completely clean slate and no leftover dimension lock.
    """
    import shutil

    persist_path = Path(CHROMA_PERSIST_DIR)

    if reset:
        if persist_path.exists():
            shutil.rmtree(persist_path)   # wipe ALL ChromaDB files, not just collection
            print(f"🗑  Wiped vector store directory: {CHROMA_PERSIST_DIR}")
        persist_path.mkdir(parents=True, exist_ok=True)
        print(f"📁 Recreated empty directory: {CHROMA_PERSIST_DIR}")

    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

    collection = client.get_or_create_collection(
        name     = CHROMA_COLLECTION,
        metadata = {"hnsw:space": "cosine"},   # cosine similarity = best for text
    )
    print(f"✅ Collection ready: \'{CHROMA_COLLECTION}\' ({collection.count()} existing chunks)")
    return collection


def store_in_chromadb(
    chunks:     list[KBChunk],
    embeddings: list[list[float]],
    collection: chromadb.Collection,
) -> None:
    """
    Insert chunks + embeddings into ChromaDB in batches.

    ChromaDB.add() takes parallel lists — ids, documents, embeddings, metadatas
    must all have the same length and correspond index-by-index.

    We store role_tags as a comma-separated string because ChromaDB
    metadata values must be strings, ints, or floats — not lists.
    """
    print(f"\n💾 Storing {len(chunks)} chunks in \'{CHROMA_COLLECTION}\'...")

    # ── Dimension guard ───────────────────────────────────────────────────
    # If the collection already has chunks, verify the stored dimension
    # matches our embeddings BEFORE attempting to add anything.
    # This catches model mismatch early with a clear error message
    # instead of a cryptic ChromaDB Rust panic deep in the stack.
    if collection.count() > 0:
        existing = collection.peek(limit=1)
        existing_dim = len(existing["embeddings"][0])
        new_dim      = len(embeddings[0])
        if existing_dim != new_dim:
            raise ValueError(
                f"Dimension mismatch: collection has {existing_dim}-dim vectors "
                f"but new embeddings are {new_dim}-dim.\n"
                f"Run with --reset to wipe the old index and rebuild from scratch."
            )

    ids       = [c.chunk_id for c in chunks]
    documents = [c.text     for c in chunks]
    metadatas = [
        {
            "section_title": c.section_title,
            "part_number":   c.part_number,
            "part_name":     c.part_name,
            "role_tags":     ",".join(c.role_tags),
            "char_count":    c.char_count,
            "chunk_index":   int(c.chunk_id.split("_")[1]),
            "embed_model":   GEMINI_EMBED_MODEL,   # store which model was used
        }
        for c in chunks
    ]

    batch_size = 100
    for start in range(0, len(chunks), batch_size):
        end = min(start + batch_size, len(chunks))
        collection.add(
            ids        = ids[start:end],
            documents  = documents[start:end],
            embeddings = embeddings[start:end],
            metadatas  = metadatas[start:end],
        )
        print(f"   Stored chunks {start}–{end - 1}")

    print(f"✅ ChromaDB ready: {collection.count()} chunks stored")


# ─────────────────────────────────────────────────────────────────
# STEP 5 — VERIFY WITH A TEST QUERY
# ─────────────────────────────────────────────────────────────────

def verify_index(collection: chromadb.Collection) -> None:
    """
    Embed a test query with the SAME model used for documents,
    then search ChromaDB to confirm retrieval works end-to-end.

    This is the canonical check that documents and queries are
    in the same vector space — if the top result is relevant,
    the index is consistent.
    """
    test_query = "what is CPV and how does it work"
    print(f"\n🔍 Verification query: \"{test_query}\"")

    client = genai.Client(api_key=GEMINI_API_KEY)

    # IMPORTANT: use RETRIEVAL_QUERY here (not RETRIEVAL_DOCUMENT)
    # Query and document use the same MODEL but different task_types
    # This is the correct asymmetric embedding pattern
    response = client.models.embed_content(
        model    = GEMINI_EMBED_MODEL,
        contents = test_query,   # single string — consistent with ingestion fix
        config   = types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    query_vector = response.embeddings[0].values

    results = collection.query(
        query_embeddings = [query_vector],
        n_results        = 3,
        include          = ["documents", "metadatas", "distances"],
    )

    print("   Top 3 results:")
    for i, (doc, meta, dist) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ), start=1):
        sim   = round(1 - dist, 3)   # cosine distance → similarity
        title = meta.get("section_title", "?")
        part  = meta.get("part_number",   "?")
        print(f"   {i}. [Part {part}] {title}  (similarity={sim})")
        print(f"      {doc[:100].replace(chr(10), ' ')}...")

    print()


# ─────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────

def main(kb_path: Path, reset: bool = False) -> None:
    print("=" * 60)
    print("  CAMPAYN KNOWLEDGE BASE INGESTION")
    print(f"  Embedding model : {GEMINI_EMBED_MODEL}")
    print("=" * 60)

    Path(CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True)

    raw_text   = load_knowledge_base(kb_path)
    chunks     = parse_into_chunks(raw_text)

    if not chunks:
        sys.exit(1)

    embeddings = generate_embeddings(chunks)

    assert len(embeddings) == len(chunks), (
        f"Mismatch: {len(embeddings)} embeddings for {len(chunks)} chunks"
    )

    collection = get_chroma_collection(reset=reset)
    store_in_chromadb(chunks, embeddings, collection)
    verify_index(collection)

    print("=" * 60)
    print("  INGESTION COMPLETE ✅")
    print(f"  Model      : {GEMINI_EMBED_MODEL}")
    print(f"  Chunks     : {len(chunks)}")
    print(f"  Location   : {CHROMA_PERSIST_DIR}")
    print("  Next step  : python -m api.modules.rag")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest the Campayn knowledge base into ChromaDB"
    )
    parser.add_argument(
        "--kb",
        type    = str,
        default = DEFAULT_KB_PATH,
        help    = "Path to knowledge_base.txt or .docx",
    )
    parser.add_argument(
        "--reset",
        action  = "store_true",
        help    = "Wipe and rebuild the ChromaDB collection from scratch",
    )
    args = parser.parse_args()
    main(kb_path=Path(args.kb), reset=args.reset)