"""
streamlit_app.py
================
Campayn AI Assistant — Streamlit Chat Interface

WHAT THIS FILE DOES
--------------------
A dual-role chat UI that calls the FastAPI backend.
Brand side: campaign planning, influencer matching, brief writing.
Creator side: content feedback, growth strategy, monetisation advice.

FEATURES
---------
- Role switcher (Brand / Creator)
- Chat history with message bubbles
- Image upload for multimodal queries (Module 3)
- MCQ rendering as clickable buttons (no typing needed)
- Session persistence across turns
- Latency + module stats shown per response

HOW IT CALLS THE BACKEND
--------------------------
Every message → POST http://localhost:8000/{role}/chat
Every image   → encoded to base64, sent in the same request body
MCQ answers   → sent as mcq_answers dict in the next request

USAGE
------
  streamlit run frontend/streamlit_app.py
  open http://localhost:8501

Author : Mohit Chauhan — Campayn Multimodal AI Assistant
"""

import base64
import time
import uuid
from io import BytesIO
from typing import Optional

import requests
import streamlit as st

# ─────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────

API_BASE_URL = "http://localhost:8000"   # FastAPI backend
PAGE_TITLE   = "Campayn AI Assistant"
PAGE_ICON    = "🎯"

# Role display config
ROLES = {
    "brand": {
        "label":       "Brand",
        "icon":        "🏢",
        "colour":      "#7F77DD",
        "placeholder": "Ask about campaign planning, influencer matching, CPV strategy...",
        "description": "Plan campaigns · Match creators · Set CPV targets · Write briefs",
    },
    "creator": {
        "label":       "Creator",
        "icon":        "🎨",
        "colour":      "#5DCAA5",
        "placeholder": "Ask about content improvement, growth strategy, Creator Coins...",
        "description": "Improve content · Grow audience · Maximise earnings · Pitch brands",
    },
}

# ─────────────────────────────────────────────────────────────────
# PAGE CONFIG — must be the first Streamlit call
# ─────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title = PAGE_TITLE,
    page_icon  = PAGE_ICON,
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ─────────────────────────────────────────────────────────────────
# CUSTOM CSS
# Streamlit's default styling is bland. We override key elements
# to match Campayn's brand feel — clean, minimal, professional.
# ─────────────────────────────────────────────────────────────────

st.markdown("""
<style>
  /* Hide Streamlit default hamburger and footer */
  #MainMenu, footer { visibility: hidden; }

  /* Chat message bubbles */
  .user-bubble {
    background: #EEEDFE;
    border-radius: 18px 18px 4px 18px;
    padding: 12px 16px;
    margin: 6px 0;
    font-size: 14px;
    color: #1a1a2e;
    max-width: 80%;
    margin-left: auto;
    border: 0.5px solid #AFA9EC;
  }
  .assistant-bubble {
    background: #F8F8F8;
    border-radius: 18px 18px 18px 4px;
    padding: 12px 16px;
    margin: 6px 0;
    font-size: 14px;
    color: #1a1a2e;
    max-width: 90%;
    border: 0.5px solid #E0E0E0;
    line-height: 1.6;
  }

  /* Metric cards */
  .metric-row {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin-top: 8px;
  }
  .metric-chip {
    font-size: 11px;
    padding: 3px 10px;
    border-radius: 20px;
    border: 0.5px solid #E0E0E0;
    background: #FAFAFA;
    color: #666;
  }
  .metric-chip.green { border-color: #97C459; background: #EAF3DE; color: #3B6D11; }
  .metric-chip.purple { border-color: #AFA9EC; background: #EEEDFE; color: #3C3489; }
  .metric-chip.amber { border-color: #EF9F27; background: #FAEEDA; color: #633806; }

  /* MCQ buttons */
  .mcq-container {
    background: #F8F9FF;
    border-radius: 12px;
    padding: 14px 16px;
    border: 0.5px solid #AFA9EC;
    margin: 8px 0;
  }
  .mcq-title {
    font-size: 13px;
    font-weight: 500;
    color: #3C3489;
    margin-bottom: 10px;
  }

  /* Sources citation */
  .sources-row {
    font-size: 11px;
    color: #888;
    margin-top: 6px;
  }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# SESSION STATE INITIALISATION
#
# Streamlit re-runs the entire script on every interaction.
# st.session_state persists data across re-runs — it's the only
# way to maintain state (chat history, session_id, role, etc.)
# ─────────────────────────────────────────────────────────────────

def init_session():
    """
    Initialise all session state variables on first load.
    st.session_state works like a dict that survives re-runs.
    """
    defaults = {
        "role":           "brand",      # current active role
        "session_id":     None,         # FastAPI session ID (None = first message)
        "messages":       [],           # list of {role, content, metadata} dicts
        "mcq_pending":    False,        # True = waiting for user to answer MCQ
        "mcq_questions":  [],           # list of MCQ question dicts
        "mcq_answers":    {},           # {field: selected_option} being built
        "mcq_submitted":  False,        # True = user clicked submit on MCQ
        "image_b64":      None,         # base64 image for current message
        "feedback_given": set(),        # set of message indices already rated
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session()


# ─────────────────────────────────────────────────────────────────
# API HELPERS
# All communication with FastAPI goes through these functions.
# We keep HTTP calls isolated here so the rest of the UI is clean.
# ─────────────────────────────────────────────────────────────────

def call_chat_api(
    query:       str,
    role:        str,
    session_id:  Optional[str] = None,
    image_b64:   Optional[str] = None,
    mcq_answers: Optional[dict] = None,
) -> dict:
    """
    POST to /brand/chat or /creator/chat and return the parsed response.
    Returns an error dict if the API is unreachable.
    """
    url  = f"{API_BASE_URL}/{role}/chat"
    body = {
        "query":       query,
        "session_id":  session_id,
        "image_b64":   image_b64,
        "mcq_answers": mcq_answers,
    }

    try:
        resp = requests.post(url, json=body, timeout=120)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        return {
            "error": (
                "Cannot connect to the API at `http://localhost:8000`. "
                "Make sure FastAPI is running: `uvicorn main:app --reload`"
            )
        }
    except requests.exceptions.Timeout:
        return {"error": "Request timed out after 120 seconds. The pipeline may be overloaded."}
    except Exception as e:
        return {"error": f"API error: {str(e)[:200]}"}


def call_feedback_api(run_id: str, score: float, comment: str = "") -> bool:
    """POST thumbs up/down to /feedback."""
    try:
        resp = requests.post(
            f"{API_BASE_URL}/feedback",
            json={"run_id": run_id, "score": score, "comment": comment},
            timeout=10,
        )
        return resp.status_code == 200
    except Exception:
        return False


def check_api_health() -> dict:
    """GET /health — check if all modules are loaded."""
    try:
        resp = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return resp.json()
    except Exception:
        return {"status": "unreachable"}


def image_to_base64(uploaded_file) -> str:
    """
    Convert a Streamlit UploadedFile to a base64 data URI string.
    FastAPI / Module 3 expect the data URI format:
      data:image/jpeg;base64,/9j/4AAQSkZJRgAB...
    """
    img_bytes = uploaded_file.read()
    mime_type = uploaded_file.type   # "image/jpeg", "image/png" etc.
    b64_str   = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{b64_str}"


# ─────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────

def render_sidebar():
    """
    Left sidebar contains:
    - Role switcher (Brand / Creator)
    - Image uploader
    - API health status
    - New conversation button
    - Session info
    """
    with st.sidebar:
        st.markdown(f"## {PAGE_ICON} {PAGE_TITLE}")
        st.markdown("---")

        # ── Role switcher ─────────────────────────────────────────
        st.markdown("### Your role")
        role_col1, role_col2 = st.columns(2)

        with role_col1:
            brand_active = st.session_state.role == "brand"
            if st.button(
                "🏢 Brand",
                use_container_width = True,
                type = "primary" if brand_active else "secondary",
            ):
                if not brand_active:
                    _switch_role("brand")

        with role_col2:
            creator_active = st.session_state.role == "creator"
            if st.button(
                "🎨 Creator",
                use_container_width = True,
                type = "primary" if creator_active else "secondary",
            ):
                if not creator_active:
                    _switch_role("creator")

        role_info = ROLES[st.session_state.role]
        st.caption(role_info["description"])

        st.markdown("---")

        # ── Image upload ──────────────────────────────────────────
        st.markdown("### Upload image")
        st.caption(
            "Brand: upload a creator's post to evaluate.\n"
            "Creator: upload your content for feedback."
        )

        uploaded = st.file_uploader(
            "Choose image",
            type      = ["jpg", "jpeg", "png", "webp"],
            label_visibility = "collapsed",
        )

        if uploaded:
            st.image(uploaded, caption="Attached to next message", use_column_width=True)
            st.session_state.image_b64 = image_to_base64(uploaded)
            st.success("Image ready to send")
        else:
            st.session_state.image_b64 = None

        st.markdown("---")

        # ── API health ────────────────────────────────────────────
        st.markdown("### System status")
        if st.button("Check", use_container_width=True):
            with st.spinner("Checking..."):
                health = check_api_health()
            if health.get("status") == "ok":
                st.success("API online")
                for module, status in health.get("modules", {}).items():
                    icon = "✅" if "ok" in status or "enabled" in status or "configured" in status else "⚠️"
                    st.caption(f"{icon} {module}: {status}")
            else:
                st.error(f"API: {health.get('status', 'unreachable')}")

        st.markdown("---")

        # ── New conversation ──────────────────────────────────────
        if st.button("🔄 New conversation", use_container_width=True):
            _reset_conversation()

        # ── Session info ──────────────────────────────────────────
        if st.session_state.session_id:
            st.caption(f"Session: `{st.session_state.session_id[:8]}...`")
        st.caption(f"Messages: {len(st.session_state.messages)}")


def _switch_role(new_role: str):
    """Switch role and reset conversation — different roles need separate sessions."""
    st.session_state.role       = new_role
    st.session_state.session_id = None
    st.session_state.messages   = []
    st.session_state.mcq_pending= False
    st.session_state.mcq_answers= {}
    st.rerun()


def _reset_conversation():
    """Clear chat history and start fresh."""
    st.session_state.session_id  = None
    st.session_state.messages    = []
    st.session_state.mcq_pending = False
    st.session_state.mcq_answers = {}
    st.session_state.feedback_given = set()
    st.rerun()


# ─────────────────────────────────────────────────────────────────
# CHAT MESSAGE RENDERING
# ─────────────────────────────────────────────────────────────────

def render_message(msg: dict, idx: int):
    """
    Render one chat message — either user or assistant.

    msg format:
        {
          "role":     "user" | "assistant",
          "content":  str,
          "metadata": dict (assistant messages only),
          "run_id":   str (for feedback, optional),
        }
    """
    role    = msg["role"]
    content = msg["content"]

    if role == "user":
        # Right-aligned purple bubble
        st.markdown(
            f'<div class="user-bubble">{content}</div>',
            unsafe_allow_html=True,
        )
        # Show attached image thumbnail if present
        if msg.get("has_image"):
            st.caption("📎 Image attached")

    else:
        # Left-aligned assistant response
        with st.container():
            st.markdown(
                f'<div class="assistant-bubble">{content}</div>',
                unsafe_allow_html=True,
            )

            # Metrics row
            meta = msg.get("metadata", {})
            if meta:
                _render_metrics(meta)

            # Sources cited
            sources = meta.get("sources", [])
            if sources:
                st.markdown(
                    f'<div class="sources-row">📚 Sources: {", ".join(sources)}</div>',
                    unsafe_allow_html=True,
                )

            # Thumbs up/down feedback (only for non-MCQ responses)
            run_id = msg.get("run_id")
            if run_id and idx not in st.session_state.feedback_given:
                _render_feedback(run_id, idx)


def _render_metrics(meta: dict):
    """Render latency and module stats as compact chips."""
    chips = []

    total_ms = meta.get("total_ms", 0)
    if total_ms:
        chips.append(f'<span class="metric-chip purple">⏱ {total_ms}ms</span>')

    rag_conf = meta.get("rag_confidence", "")
    if rag_conf:
        cls = "green" if rag_conf == "high" else "amber" if rag_conf == "medium" else ""
        chips.append(f'<span class="metric-chip {cls}">RAG: {rag_conf}</span>')

    m1 = meta.get("module1_matches", 0)
    if m1:
        chips.append(f'<span class="metric-chip">👤 {m1:,} creators matched</span>')

    m2 = meta.get("module2_chunks", 0)
    if m2:
        chips.append(f'<span class="metric-chip">📄 {m2} chunks retrieved</span>')

    if meta.get("vision_used"):
        chips.append('<span class="metric-chip purple">🖼 Vision analysed</span>')

    savings = meta.get("parallel_savings_ms", 0)
    if savings > 100:
        chips.append(f'<span class="metric-chip green">⚡ Saved ~{savings:.0f}ms (parallel)</span>')

    if chips:
        st.markdown(
            f'<div class="metric-row">{"".join(chips)}</div>',
            unsafe_allow_html=True,
        )


def _render_feedback(run_id: str, msg_idx: int):
    """Render thumbs up/down buttons for a specific message."""
    col1, col2, col3 = st.columns([1, 1, 8])
    with col1:
        if st.button("👍", key=f"up_{msg_idx}", help="Helpful response"):
            call_feedback_api(run_id, 1.0)
            st.session_state.feedback_given.add(msg_idx)
            st.rerun()
    with col2:
        if st.button("👎", key=f"dn_{msg_idx}", help="Unhelpful response"):
            call_feedback_api(run_id, 0.0)
            st.session_state.feedback_given.add(msg_idx)
            st.rerun()


# ─────────────────────────────────────────────────────────────────
# MCQ RENDERING
# When needs_mcq=True, we show clickable MCQ buttons instead of
# a text input. User clicks options → answers are submitted
# automatically as the next message to the API.
# ─────────────────────────────────────────────────────────────────

def render_mcq_form(questions: list) -> Optional[dict]:
    """
    Render MCQ questions as clickable selectboxes.
    Returns the answers dict when user clicks Submit, or None.

    This replaces the text chat input when MCQ is pending.
    The user never has to type — just click options.
    """
    st.markdown(
        '<div class="mcq-container">'
        '<div class="mcq-title">🎯 I need a few quick details to find the best match:</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    answers = {}

    for q in questions:
        field   = q.get("field", "")
        label   = q.get("label", "")
        options = q.get("options", [])
        hint    = q.get("hint", "")

        st.markdown(f"**{label}**")
        if hint:
            st.caption(hint)

        selected = st.selectbox(
            label           = label,
            options         = options,
            key             = f"mcq_{field}",
            label_visibility= "collapsed",
        )
        answers[field] = selected

    col1, col2 = st.columns([2, 8])
    with col1:
        submitted = st.button("Continue →", type="primary", use_container_width=True)

    if submitted:
        return answers
    return None


# ─────────────────────────────────────────────────────────────────
# MESSAGE SENDING
# ─────────────────────────────────────────────────────────────────

def send_message(query: str, mcq_answers: Optional[dict] = None):
    """
    Send a message to the FastAPI backend and handle the response.

    Handles three response types:
      1. Normal response → append to chat history
      2. MCQ interrupt   → set mcq_pending=True, store questions
      3. API error       → show error message
    """
    role      = st.session_state.role
    image_b64 = st.session_state.image_b64

    # Append user message to chat history
    st.session_state.messages.append({
        "role":      "user",
        "content":   query,
        "has_image": image_b64 is not None,
    })

    # Call the API
    with st.spinner("Thinking..."):
        result = call_chat_api(
            query       = query,
            role        = role,
            session_id  = st.session_state.session_id,
            image_b64   = image_b64,
            mcq_answers = mcq_answers,
        )

    # Clear image after sending
    st.session_state.image_b64 = None

    # Handle API error
    if "error" in result:
        st.session_state.messages.append({
            "role":    "assistant",
            "content": f"⚠️ {result['error']}",
        })
        return

    # Store session_id from first response
    if not st.session_state.session_id:
        st.session_state.session_id = result.get("session_id")

    # Handle MCQ interrupt
    if result.get("needs_mcq"):
        st.session_state.mcq_pending   = True
        st.session_state.mcq_questions = result.get("mcq_questions", [])
        # Show the MCQ prompt as an assistant message
        st.session_state.messages.append({
            "role":    "assistant",
            "content": result.get("response", "I need a few more details:"),
        })
        return

    # Normal response — clear any pending MCQ state
    st.session_state.mcq_pending   = False
    st.session_state.mcq_questions = []
    st.session_state.mcq_answers   = {}

    # Append assistant response to chat history
    st.session_state.messages.append({
        "role":     "assistant",
        "content":  result.get("response", ""),
        "metadata": result.get("metadata", {}),
        "run_id":   result.get("run_id"),
    })


# ─────────────────────────────────────────────────────────────────
# MAIN UI LAYOUT
# ─────────────────────────────────────────────────────────────────

def main():
    render_sidebar()

    # ── Header ────────────────────────────────────────────────────
    role_info = ROLES[st.session_state.role]
    st.markdown(
        f"## {role_info['icon']} Campayn — {role_info['label']} Assistant"
    )
    st.caption(role_info["description"])
    st.markdown("---")

    # ── Chat history ──────────────────────────────────────────────
    # Render all previous messages in order
    chat_container = st.container()
    with chat_container:
        if not st.session_state.messages:
            # Empty state — show example queries
            st.markdown("#### Get started")
            role = st.session_state.role
            if role == "brand":
                examples = [
                    "Find top fitness micro-influencers for a protein supplement launch",
                    "What CPV should I target for a beauty campaign with ₹2L budget?",
                    "Write a campaign brief for our new skincare line",
                    "Which meme pages work best for food delivery brands?",
                ]
            else:
                examples = [
                    "How do I grow from 10k to 50k Instagram followers?",
                    "Review my reel hook — I'll upload a screenshot",
                    "How do Creator Coins work and when do I get paid?",
                    "How do I pitch a brand campaign on Campayn?",
                ]

            cols = st.columns(2)
            for i, example in enumerate(examples):
                with cols[i % 2]:
                    if st.button(example, key=f"eg_{i}", use_container_width=True):
                        send_message(example)
                        st.rerun()
        else:
            for idx, msg in enumerate(st.session_state.messages):
                render_message(msg, idx)

    st.markdown("---")

    # ── MCQ form OR chat input ─────────────────────────────────────
    # If MCQ is pending, show the MCQ form instead of the text input.
    # Once answered, the answers are sent as the next API call.
    if st.session_state.mcq_pending:
        answers = render_mcq_form(st.session_state.mcq_questions)
        if answers:
            # User submitted MCQ — send their original query + answers
            original_query = st.session_state.messages[-2]["content"]  # last user msg
            st.session_state.mcq_pending = False
            send_message(original_query, mcq_answers=answers)
            st.rerun()

    else:
        # Normal chat input
        with st.form(key="chat_form", clear_on_submit=True):
            col1, col2 = st.columns([9, 1])
            with col1:
                user_input = st.text_input(
                    "Message",
                    placeholder     = role_info["placeholder"],
                    label_visibility= "collapsed",
                )
            with col2:
                submitted = st.form_submit_button("Send", use_container_width=True)

        if submitted and user_input.strip():
            send_message(user_input.strip())
            st.rerun()


# ─────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()