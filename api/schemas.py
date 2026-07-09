"""
schemas.py
==========
Campayn AI Assistant — FastAPI Pydantic Schemas

All request and response models for the API.
Pydantic validates every incoming request automatically —
if the user sends wrong types or missing fields, FastAPI
returns a 422 error with a clear message before our code runs.

Author : Mohit Chauhan — Campayn Multimodal AI Assistant
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field, field_validator


# ─────────────────────────────────────────────────────────────────
# REQUEST MODELS — what the client sends
# ─────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    """
    Request body for POST /brand/chat and POST /creator/chat.

    The client sends:
      - query      : the user's question
      - session_id : conversation ID (client generates on first call,
                     then sends it back on every subsequent call)
      - image_b64  : optional base64 image for multimodal queries
      - mcq_answers: if the previous response had needs_mcq=True,
                     the client sends the user's MCQ selections here
    """
    query:       str  = Field(..., min_length=1, max_length=2000,
                              description="User's question or message")
    session_id:  Optional[str] = Field(None,
                              description="Conversation session ID. "
                                          "Omit on first message — server generates one.")
    image_b64:   Optional[str] = Field(None,
                              description="Base64-encoded image for multimodal queries. "
                                          "Include data URI prefix: data:image/jpeg;base64,...")
    mcq_answers: Optional[dict] = Field(None,
                              description="MCQ answers from previous turn. "
                                          "Format: {field_name: selected_option_string}")

    @field_validator("query")
    @classmethod
    def strip_query(cls, v: str) -> str:
        """Strip leading/trailing whitespace from query."""
        stripped = v.strip()
        if not stripped:
            raise ValueError("Query cannot be empty or whitespace only")
        return stripped

    @field_validator("image_b64")
    @classmethod
    def validate_image(cls, v: Optional[str]) -> Optional[str]:
        """Basic validation for base64 image string."""
        if v is None:
            return v
        # Must start with data URI or look like base64
        if not (v.startswith("data:image/") or len(v) > 100):
            raise ValueError("image_b64 must be a base64-encoded image string")
        return v

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "Find top fitness micro-influencers for a protein brand",
                    "session_id": None,
                    "image_b64": None,
                    "mcq_answers": None,
                }
            ]
        }
    }


class FeedbackRequest(BaseModel):
    """
    Request body for POST /feedback.
    Logs user thumbs up/down to LangSmith.
    """
    run_id:  str   = Field(..., description="LangSmith run ID from the chat response")
    score:   float = Field(..., ge=0.0, le=1.0,
                           description="1.0 = thumbs up, 0.0 = thumbs down")
    comment: Optional[str] = Field(None, max_length=500,
                                   description="Optional user comment")

    @field_validator("score")
    @classmethod
    def round_score(cls, v: float) -> float:
        """Round to avoid floating point artifacts."""
        return round(v, 2)


# ─────────────────────────────────────────────────────────────────
# RESPONSE MODELS — what the server sends back
# ─────────────────────────────────────────────────────────────────

class MCQQuestion(BaseModel):
    """One MCQ question shown to the user when the pipeline needs more info."""
    field:   str        # which FilterSpec field this fills
    label:   str        # human-readable question text
    options: list[str]  # clickable answer choices
    hint:    str        # short context shown below question


class PipelineMetrics(BaseModel):
    """Per-run performance metrics returned with every response."""
    total_ms:           float = 0.0
    intent_parser_ms:   float = 0.0
    module1_ms:         float = 0.0
    module2_ms:         float = 0.0
    module3_ms:         float = 0.0
    generator_ms:       float = 0.0
    rag_confidence:     str   = "unknown"
    module1_matches:    int   = 0
    module2_chunks:     int   = 0
    vision_used:        bool  = False
    estimated_cost_usd: float = 0.0


class ChatResponse(BaseModel):
    """
    Response body for POST /brand/chat and POST /creator/chat.

    Two possible states:

    State A — normal response (needs_mcq=False):
        response      = the assistant's answer text
        session_id    = send this back with every subsequent message
        needs_mcq     = False
        mcq_questions = None

    State B — MCQ needed (needs_mcq=True):
        response      = formatted MCQ question text (show this to user)
        session_id    = send this back with mcq_answers in next request
        needs_mcq     = True
        mcq_questions = list of MCQQuestion objects for the UI to render
    """
    response:      str
    session_id:    str
    role:          str
    needs_mcq:     bool              = False
    mcq_questions: Optional[list[MCQQuestion]] = None
    sources:       list[str]         = Field(default_factory=list,
                                             description="Knowledge base parts cited")
    run_id:        Optional[str]     = None   # LangSmith run ID for feedback
    metrics:       Optional[PipelineMetrics] = None

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "response": "Based on the Campayn knowledge base...",
                    "session_id": "550e8400-e29b-41d4-a716-446655440000",
                    "role": "brand",
                    "needs_mcq": False,
                    "mcq_questions": None,
                    "sources": ["Part 1", "Part 2"],
                    "run_id": "abc123",
                    "metrics": {
                        "total_ms": 2340.5,
                        "rag_confidence": "high",
                    }
                }
            ]
        }
    }


class HealthResponse(BaseModel):
    """Response for GET /health."""
    status:       str
    version:      str  = "0.1.0"
    environment:  str
    langsmith_on: bool = False
    modules: dict = Field(default_factory=dict,
                          description="Status of each loaded module")


class FeedbackResponse(BaseModel):
    """Response for POST /feedback."""
    success: bool
    message: str