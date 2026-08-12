# frontend/app.py
import os
import sys
import uuid

import requests
import streamlit as st

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.core.config import settings
from frontend.components.property_card import render_live_availability, render_property_card


st.set_page_config(
    page_title=settings.app_name,
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stHeader"],
    [data-testid="stToolbar"] {
        background-color: #ffffff;
        color: #172033;
    }

    .block-container {
        max-width: 1120px;
        padding-top: 1.25rem;
    }

    .brand-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 1rem 0 0.5rem 0;
        text-align: center;
    }

    .brand-tagline {
        color: #516173;
        font-size: 0.95rem;
        letter-spacing: 0;
    }

    .stChatMessage {
        background-color: #f7f9fc !important;
        border: 1px solid #d8e0ea !important;
        border-radius: 12px !important;
        box-shadow: 0 2px 8px rgba(24, 37, 56, 0.04);
        margin-bottom: 1rem !important;
    }

    [data-testid="stBottom"],
    [data-testid="stBottomBlockContainer"],
    [data-testid="stChatInput"] {
        background: #ffffff !important;
    }

    [data-testid="stBottom"] > div,
    [data-testid="stBottomBlockContainer"] > div {
        background: #ffffff !important;
    }

    [data-testid="stChatInput"] {
        max-width: 1120px;
        margin: 0 auto;
        padding: 0.75rem 1rem 1rem 1rem;
    }

    [data-testid="stChatInput"] div[data-baseweb="textarea"] {
        background-color: #ffffff !important;
        border: 1px solid #c9d4e2 !important;
        border-radius: 12px !important;
        box-shadow: 0 2px 10px rgba(24, 37, 56, 0.08) !important;
    }

    [data-testid="stChatInput"] div[data-baseweb="textarea"]:focus-within {
        border-color: #19747e !important;
        box-shadow: 0 0 0 3px rgba(25, 116, 126, 0.16) !important;
    }

    [data-testid="stChatInput"] textarea,
    .stChatInput textarea,
    .stChatInput input {
        background-color: #ffffff !important;
        color: #172033 !important;
        border: 0 !important;
        box-shadow: none !important;
        caret-color: #172033 !important;
    }

    [data-testid="stChatInput"] textarea::placeholder,
    .stChatInput textarea::placeholder,
    .stChatInput input::placeholder {
        color: #6b7887 !important;
        opacity: 1 !important;
    }

    [data-testid="stChatInput"] button {
        color: #19747e !important;
    }

    .stButton > button {
        background: #ffffff !important;
        color: #19747e !important;
        border: 1px solid #9fc5cb !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        margin: -0.25rem 0 1rem 0 !important;
    }

    .stButton > button:hover {
        background: #eef8fa !important;
        border-color: #19747e !important;
        color: #125d66 !important;
    }

    h1, h2, h3, h4 {
        color: #172033 !important;
        font-family: 'Inter', sans-serif;
    }

    p, li, span {
        color: #263445;
    }

    .stSpinner {
        color: #19747e;
    }

    section[data-testid="stSidebar"] {
        background-color: #f7f9fc;
        border-right: 1px solid #d8e0ea;
    }
</style>
""",
    unsafe_allow_html=True,
)


def _reply_for_frontend(data: dict, raw_reply: str) -> str:
    """Keep recommendation turns readable; cards hold the detailed hotel copy."""
    stage = data.get("pipeline_stage_reached", "")
    properties = data.get("properties") or []
    if properties and stage in {"rag_recommendations", "show_more"}:
        return (
            "I found local hotel matches from the RAG database. "
            "Click a hotel option or type an option number when you want me to check live availability."
        )
    return raw_reply


def _send_chat_message(prompt: str) -> str | None:
    prompt = prompt.strip()
    if not prompt:
        return None

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.live_availability = None

    try:
        response = requests.post(
            f"{settings.api_base_url}/api/chat",
            json={
                "session_id": st.session_state.session_id,
                "user_message": prompt,
                "conversation_history": st.session_state.messages,
            },
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.ConnectionError:
        error = "Cannot reach the backend. Verify that Uvicorn is listening."
    except requests.exceptions.Timeout:
        error = "The hotel lookup timed out. Try again or choose another hotel."
    except requests.exceptions.HTTPError as exc:
        error = f"Gateway error [{exc.response.status_code}]: {exc.response.text}"
    except Exception as exc:
        error = f"Pipeline parsing failure: {exc}"
    else:
        raw_reply = data.get("assistant_message", "No response payload parsed.")
        reply = _reply_for_frontend(data, raw_reply)
        stage = data.get("pipeline_stage_reached", "unknown")
        new_properties = data.get("properties") or []

        if new_properties:
            st.session_state.properties = new_properties
        elif stage in {"rag_search_empty", "setup_error", "error"}:
            st.session_state.properties = []

        st.session_state.live_availability = data.get("live_availability")
        st.session_state.last_pipeline_stage = stage
        st.session_state.messages.append({"role": "assistant", "content": reply})
        return None

    st.session_state.messages.append({"role": "assistant", "content": error})
    return error


def _chat_placeholder() -> str:
    assistant_messages = [
        message.get("content", "")
        for message in st.session_state.get("messages", [])
        if message.get("role") == "assistant"
    ]
    latest = assistant_messages[-1].lower() if assistant_messages else ""

    if "check-in" in latest:
        return "Enter check-in date, e.g. 2026-08-01"
    if "check-out" in latest:
        return "Enter check-out date, e.g. 2026-08-03"
    if "how many adults" in latest:
        return "Enter adult count, e.g. 2"
    if "how many children" in latest:
        return "Enter children count, e.g. 0"
    if "children's ages" in latest:
        return "Enter child ages, e.g. 5, 8"
    if "how many rooms" in latest:
        return "Enter room count, e.g. 1"
    if "which hotel" in latest or "choose another hotel" in latest:
        return "Click a hotel option or type e.g. option 2"
    return "Where do you want to stay? (e.g., Beach resorts in Hurghada)"


if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "properties" not in st.session_state:
    st.session_state.properties = []

if "live_availability" not in st.session_state:
    st.session_state.live_availability = None

if "last_pipeline_stage" not in st.session_state:
    st.session_state.last_pipeline_stage = "start"


with st.sidebar:
    st.markdown("<div style='padding-top: 1.5rem;'></div>", unsafe_allow_html=True)
    st.markdown("### System Controls")
    st.markdown("---")
    st.markdown("**Try asking:**")
    st.markdown("- *Find me a luxury hotel in Cairo*")
    st.markdown("- *Beach resorts in Hurghada*")
    st.markdown("- *Family hotels in Sharm El Sheikh*")
    st.markdown("- *Show more*")
    st.markdown("- *Option 2, 2026-08-01 to 2026-08-03, 2 adults, 0 children, 1 room*")
    st.markdown("---")

    if st.button("Clear Chat Session", width="stretch"):
        st.session_state.messages = []
        st.session_state.properties = []
        st.session_state.live_availability = None
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()


logo_path = "resources/logo.png"

if os.path.exists(logo_path):
    left_spacer, center_canvas, right_spacer = st.columns([3, 6, 3])
    with center_canvas:
        st.image(logo_path, width="stretch")
else:
    left_spacer, center_canvas, right_spacer = st.columns([2, 8, 2])
    with center_canvas:
        st.markdown(
            "<h1 style='text-align: center;'>GuideMe Travel Discovery</h1>",
            unsafe_allow_html=True,
        )

st.markdown(
    """
<div class="brand-container">
    <div class="brand-tagline">
        Egypt hotel recommendations from local RAG, with live availability after selection.
    </div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown("<div style='margin-bottom: 2rem;'></div>", unsafe_allow_html=True)


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if st.session_state.properties:
    st.markdown("---")
    st.markdown("### Recommended Hotels")

    passed = [p for p in st.session_state.properties if p.get("passed_review", True)]
    failed = [p for p in st.session_state.properties if not p.get("passed_review", True)]
    selected_prompt = None

    if passed:
        for prop in passed:
            selected_prompt = render_property_card(prop) or selected_prompt

    if failed:
        with st.expander(f"{len(failed)} properties flagged during rule-matching logic constraints"):
            for prop in failed:
                selected_prompt = render_property_card(prop) or selected_prompt

    if selected_prompt:
        with st.spinner("Preparing live availability questions..."):
            _send_chat_message(selected_prompt)
        st.rerun()

render_live_availability(st.session_state.live_availability)


if prompt := st.chat_input(_chat_placeholder()):
    with st.spinner("Searching hotels..."):
        _send_chat_message(prompt)
    st.rerun()
