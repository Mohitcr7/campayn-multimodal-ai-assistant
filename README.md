# Campayn — Multimodal Agentic AI Assistant

A dual-role (brand + creator) influencer-marketing assistant that combines **structured retrieval over 56K+ creator profiles**, **Retrieval-Augmented Generation (RAG)** over a domain knowledge base, and **multimodal vision analysis** behind a single **LangGraph** agentic pipeline.

**Stack:** Python · LangGraph · LangChain · ChromaDB · Google Gemini · Anthropic Claude · FastAPI · Pydantic · Redis · Streamlit · Docker · LangSmith

---

## Architecture

```
                        User query (+ optional image)
                                   │
                          ┌────────▼────────┐
                          │  intent_parser  │  extract structured FilterSpec;
                          │   (LLM + MCQ)   │  ask clarifying questions if
                          └────────┬────────┘  confidence is low (human-in-the-loop,
                                   │            graph paused/resumed via checkpointer)
                 ┌─────────────────┼─────────────────┐   ← LangGraph Send() fan-out
        ┌────────▼───────┐ ┌───────▼────────┐ ┌───────▼────────┐
        │  Module 1      │ │  Module 2      │ │  Module 3      │
        │  Profile       │ │  Knowledge     │ │  Vision        │   run in PARALLEL
        │  Filter        │ │  Base RAG      │ │  Analysis      │
        │  (pandas, 56K) │ │  (ChromaDB)    │ │  (Gemini VLM)  │
        └────────┬───────┘ └───────┬────────┘ └───────┬────────┘
                 └─────────────────┼─────────────────┘
                          ┌────────▼────────┐
                          │ prompt_builder  │  merge module outputs + history
                          └────────┬────────┘
                          ┌────────▼────────┐
                          │   generator     │  Gemini → Anthropic Claude failover
                          └────────┬────────┘
                          ┌────────▼────────┐
                          │  save_memory    │  Redis conversation history
                          └─────────────────┘
```

- **7-node stateful graph** orchestrated with LangGraph; the three retrieval/analysis modules are dispatched **concurrently** via the `Send()` fan-out API (wall-clock ≈ slowest module, not the sum).
- **Automatic LLM failover:** every text/vision generation call tries Gemini first and transparently falls back to Anthropic Claude on error (rate limit / outage / bad key). See [api/llm_fallback.py](api/llm_fallback.py). RAG embeddings stay Gemini-only (no Anthropic embeddings API).

## Modules

| Module | File | What it does |
|--------|------|--------------|
| Profile Filter | [api/modules/profile_filter.py](api/modules/profile_filter.py) | pandas-based normalization + typed filtering over 56K+ creator profiles |
| Knowledge Base RAG | [api/modules/rag.py](api/modules/rag.py) | ChromaDB semantic search over 3,072-dim Gemini embeddings with role-aware re-ranking |
| Vision Analysis | [api/modules/vision.py](api/modules/vision.py) | Gemini VLM image analysis returning a 35-field structured JSON schema |

## Project structure

```
api/                    FastAPI backend
  main.py               app entry point (uvicorn api.main:app)
  routers.py            /brand, /creator, /feedback, /health endpoints
  schemas.py            Pydantic request/response models
  graph.py              LangGraph pipeline (7 nodes)
  intent_parser.py      LLM intent extraction + MCQ clarification
  prompt_constructor.py role-aware prompt assembly
  llm_fallback.py       Gemini → Anthropic failover
  modules/              profile_filter · rag · vision
  observability/        langsmith_setup
frontend/               Streamlit chat UI
scripts/                knowledge base ingestion, model discovery
data/raw/               creator_profiles.csv, knowledge_base.txt
```

## Setup

```bash
cp .env.example .env          # fill in GEMINI_API_KEY, ANTHROPIC_API_KEY, etc.
pip install -r requirements.txt
python scripts/ingest_knowledge_base.py --reset   # build the ChromaDB vector store
```

## Run

```bash
# Backend (FastAPI)
uvicorn api.main:app --reload            # http://localhost:8000/docs

# Frontend (Streamlit)
streamlit run frontend/streamlit_app.py  # http://localhost:8501

# Or the full stack (API + UI + Redis)
docker-compose up --build
```

## Configuration

All configuration is via environment variables — see [.env.example](.env.example). Key settings: `GEMINI_API_KEY`, `GEMINI_MODEL`, `ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL`, `CHROMA_PERSIST_DIR`, `REDIS_URL`, `LANGCHAIN_API_KEY`.
