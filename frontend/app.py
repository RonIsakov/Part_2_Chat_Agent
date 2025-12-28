"""
Streamlit Chat UI for Medical Services Chatbot.

A user-friendly web interface for:
1. Collection Phase: Gathering user information
2. Q&A Phase: Answering medical services questions with RAG

Features:
- Chat interface with message history
- Sidebar with user profile display
- Language switcher (Hebrew/English)
- Sources display for Q&A answers
- Health check indicator
- Reset functionality
"""

import streamlit as st
import sys
import os
from pathlib import Path
from datetime import datetime

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.api_client import get_api_client


# Page configuration
st.set_page_config(
    page_title="Medical Services Chatbot",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better chat UI
st.markdown("""
<style>
    /* 1. Main App Background */
    .stApp {
        background-color: #212121;
    }

    /* 2. Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #1e1e2e;
    }
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] .element-container, 
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] .stMarkdown {
        text-align: center !important;
    }

    /* 3. Header Styling */
    header[data-testid="stHeader"] {
        background-color: #1e1e2e !important;
    }

    /* ---------------------------------------------------- */
    /* CHAT INPUT STYLING - THE FIX                         */
    /* ---------------------------------------------------- */

    /* A. Target the Sticky Bottom Container */
    /* This handles the large white strip at the bottom */
    .stBottom, 
    div[data-testid="stBottom"],
    div[data-testid="stBottom"] > div {
        background-color: transparent !important; /* Make it see-through */
        border-top: none !important;
    }

    /* B. The Chat Input Wrapper */
    /* Make this transparent so no white box sits behind the rounded pill */
    div[data-testid="stChatInput"] {
        background-color: transparent !important;
    }

    /* C. The Input "Pill" (The rounded shape) */
    /* We apply the dark color HERE, on the specific rounded element */
    div[data-testid="stChatInput"] > div {
        background-color: #2d2d3d !important; /* Dark Grey */
        color: #ffffff !important;
        border-color: #3d3d4d !important;
        border-radius: 20px !important; 
    }

    /* D. The Text Area inside the pill */
    div[data-testid="stChatInput"] textarea {
        background-color: transparent !important; /* Let the pill color show through */
        color: #ffffff !important;
    }

    /* E. The Placeholder Text */
    div[data-testid="stChatInput"] textarea::placeholder {
        color: #b0b0b0 !important;
    }
    
    /* F. The Send Button */
    div[data-testid="stChatInput"] button {
        color: #4CAF50 !important;
    }
    div[data-testid="stChatInput"] button:hover {
        color: #45a049 !important;
    }

    /* ---------------------------------------------------- */
    /* MESSAGE BUBBLES                                      */
    /* ---------------------------------------------------- */
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }

    /* Chat message text - white for readability on dark background */
    .stChatMessage p,
    .stChatMessage div,
    .stChatMessage span {
        color: #ffffff !important;
    }

    div[data-testid="chatAvatarIcon-user"] {
        background-color: #bbdefb !important;
    }

    /* Main chat title - white and centered */
    .main h1 {
        text-align: center !important;
        color: #ffffff !important;
    }

    /* Override light mode defaults */
    .main .block-container h1 {
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)


def apply_rtl_styling():
    """Apply RTL (right-to-left) styling for Hebrew language."""
    st.markdown("""
    <style>
        /* Main content area RTL */
        .main .block-container {
            direction: rtl;
            text-align: right;
        }

        /* Sidebar content RTL */
        [data-testid="stSidebar"] > div {
            direction: rtl;
            text-align: right;
        }

        /* Chat messages RTL */
        .stChatMessage {
            direction: rtl;
            text-align: right;
        }

        /* Chat input RTL */
        .stChatInputContainer textarea {
            direction: rtl;
            text-align: right;
        }

        /* Title and headers RTL */
        h1, h2, h3, h4, h5, h6 {
            direction: rtl;
            text-align: right;
        }

        /* Info boxes RTL */
        .stAlert {
            direction: rtl;
            text-align: right;
        }
    </style>
    """, unsafe_allow_html=True)


def initialize_session_state():
    """Initialize Streamlit session state variables."""

    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []

    if "user_data" not in st.session_state:
        st.session_state.user_data = {
            "name": None,
            "id": None,
            "gender": None,
            "age": None,
            "hmo": None,
            "hmo_card": None,
            "tier": None,
            "confirmed": False
        }

    if "phase" not in st.session_state:
        st.session_state.phase = "collection"

    if "language" not in st.session_state:
        st.session_state.language = None  # Not set initially

    if "language_selected" not in st.session_state:
        st.session_state.language_selected = False  # Track if language was chosen

    if "backend_url" not in st.session_state:
        st.session_state.backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")


def show_language_selection():
    """
    Show language selection dialog before chat starts.

    Returns:
        bool: True if language was selected, False otherwise
    """
    # Custom CSS for green buttons
    st.markdown("""
    <style>
        div.stButton > button {
            background-color: #4CAF50;
            color: white;
            font-size: 18px;
            font-weight: bold;
            padding: 1rem 2rem;
            border: none;
            border-radius: 8px;
        }
        div.stButton > button:hover {
            background-color: #45a049;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align: center; padding: 2rem;">
        <h1>Welcome to your Personal AI Medical Assistant</h1>
        <h5>Please select your preferred language</h5>
        <h5 style="direction: rtl;">אנא בחר את השפה המועדפת עליך</h5>
    </div>
    """, unsafe_allow_html=True)

    # Create three columns for layout (left spacing, content, right spacing)
    col1, col2, col3, col4, col5 = st.columns([1, 2, 0.5, 2, 1])

    # Hebrew button on the left
    with col2:
        if st.button("עברית", use_container_width=True):
            st.session_state.language = "he"
            st.session_state.language_selected = True
            st.rerun()

    # English button on the right
    with col4:
        if st.button("English", use_container_width=True):
            st.session_state.language = "en"
            st.session_state.language_selected = True
            st.rerun()

    # Yellow info box below both buttons
    st.write("")  # Spacing

    col_left, col_center, col_right = st.columns([1, 3, 1])
    with col_center:
        st.markdown("""
        <div style="background-color: #FFD700; padding: 1.5rem; border-radius: 8px; text-align: center; border: 3px solid #FFA500;">
            <p style="margin: 0; font-size: 16px; color: #000; font-weight: 900;">
                <strong>The entire conversation will be in your selected language</strong>
            </p>
            <p style="margin: 0.5rem 0 0 0; font-size: 16px; direction: rtl; color: #000; font-weight: 900;">
                <strong>השיחה כולה תתנהל בשפה שבחרת</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)

    return False


def check_backend_health(api_client):
    """
    Check if backend is healthy.

    Returns:
        tuple: (is_healthy: bool, status_text: str)
    """
    try:
        health_data = api_client.check_health()
        status = health_data.get("status", "unknown")

        if status == "healthy":
            return True, "✅ Backend Connected"
        else:
            return False, f"⚠️ Backend {status.title()}"

    except Exception as e:
        return False, "❌ Backend Offline"


def render_sidebar():
    """Render sidebar with user profile, settings, and health check."""

    with st.sidebar:
        st.title("Medical Assistant")
       # User profile display
        st.divider()
        st.subheader("👤 User Profile")

        user_data = st.session_state.user_data

        if st.session_state.phase == "collection":
            st.info("📝 Collecting information..." if st.session_state.language == "en" else "📝 אוסף מידע...")
        else:
            st.success("✅ Profile Complete" if st.session_state.language == "en" else "✅ פרופיל הושלם")

        # Display collected fields
        if user_data.get("name"):
            st.write(f"**Name:** {user_data['name']}")

        if user_data.get("id"):
            st.write(f"**ID:** {user_data['id']}")

        if user_data.get("gender"):
            st.write(f"**Gender:** {user_data['gender']}")

        if user_data.get("age"):
            st.write(f"**Age:** {user_data['age']}")

        if user_data.get("hmo"):
            hmo_display = user_data['hmo'].title()
            st.write(f"**HMO:** {hmo_display}")

        if user_data.get("hmo_card"):
            st.write(f"**HMO Card:** {user_data['hmo_card']}")

        if user_data.get("tier"):
            tier_display = user_data['tier'].title()
            st.write(f"**Tier:** {tier_display}")

        st.divider()
        # Display selected language (read-only)
        st.subheader("🌐 Language")
        lang_display = "עברית (Hebrew)" if st.session_state.language == "he" else "English"
        st.info(f"**{lang_display}**")
        st.caption("To change language,\n click 'Start Over'")

        st.divider()

        # Reset button
        if st.button("🔄 Restart Chat", type="secondary", use_container_width=True):
            st.session_state.conversation_history = []
            st.session_state.user_data = {
                "name": None,
                "id": None,
                "gender": None,
                "age": None,
                "hmo": None,
                "hmo_card": None,
                "tier": None,
                "confirmed": False
            }
            st.session_state.phase = "collection"
            st.session_state.language = None
            st.session_state.language_selected = False
            st.rerun()

        # Debug info (collapsible)
        with st.expander("🔧 Debug Info"):
            st.write(f"**Phase:** {st.session_state.phase}")
            st.write(f"**Messages:** {len(st.session_state.conversation_history)}")
            st.write(f"**Backend:** {st.session_state.backend_url}")


def render_chat_messages():
    """Render all chat messages from conversation history."""

    for msg in st.session_state.conversation_history:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        sources = msg.get("sources", [])

        # Display message
        with st.chat_message(role):
            st.write(content)

            # Display sources if present (from assistant messages in Q&A phase)
            if role == "assistant" and sources:
                with st.expander(f"📚 Sources ({len(sources)})"):
                    for i, source in enumerate(sources[:5], 1):  # Show top 5
                        st.markdown(
                            f"**{i}.** {source.get('service', source.get('category', 'N/A'))} | "
                            f"HMO: {source.get('hmo')} | "
                            f"Tier: {source.get('tier')} | "
                            f"Score: {source.get('relevance_score', 0):.2f}"
                        )


def send_message(user_message: str):
    """
    Send message to backend and update conversation history.

    Args:
        user_message: User's message text
    """

    # Add user message to history
    st.session_state.conversation_history.append({
        "role": "user",
        "content": user_message
    })

    # Display user message immediately
    with st.chat_message("user"):
        st.write(user_message)

    # Show thinking indicator
    with st.chat_message("assistant"):
        with st.spinner("Thinking..." if st.session_state.language == "en" else "חושב..."):
            try:
                # Get API client
                api_client = get_api_client(st.session_state.backend_url)

                # Convert conversation history to API format (exclude sources)
                api_history = [
                    {"role": msg["role"], "content": msg["content"]}
                    for msg in st.session_state.conversation_history[:-1]  # Exclude the message we just added
                ]

                # Send request to backend
                response_data = api_client.send_message(
                    message=user_message,
                    user_data=st.session_state.user_data,
                    conversation_history=api_history,
                    language=st.session_state.language
                )

                # Extract response
                bot_response = response_data.get("response", "")
                phase = response_data.get("phase", "collection")
                updated_user_data = response_data.get("user_data", {})
                sources = response_data.get("sources", [])

                # Update session state
                st.session_state.phase = phase
                st.session_state.user_data = updated_user_data

                # Add assistant message to history
                assistant_msg = {
                    "role": "assistant",
                    "content": bot_response,
                    "sources": sources
                }
                st.session_state.conversation_history.append(assistant_msg)

                # Display bot response
                st.write(bot_response)

                # Display sources if in Q&A phase
                if sources:
                    with st.expander(f"📚 Sources ({len(sources)})"):
                        for i, source in enumerate(sources[:5], 1):
                            st.markdown(
                                f"**{i}.** {source.get('service', source.get('category', 'N/A'))} | "
                                f"HMO: {source.get('hmo')} | "
                                f"Tier: {source.get('tier')} | "
                                f"Score: {source.get('relevance_score', 0):.2f}"
                            )

                # Force rerun to update sidebar
                st.rerun()

            except Exception as e:
                error_msg = str(e)
                st.error(f"Error: {error_msg}")

                # Remove the user message since we couldn't process it
                st.session_state.conversation_history.pop()


def main():
    """Main application entry point."""

    # Initialize session state
    initialize_session_state()

    # Show language selection if not selected yet
    if not st.session_state.language_selected:
        show_language_selection()
        return  # Don't render rest of UI until language is selected

    # Apply RTL styling if Hebrew is selected
    if st.session_state.language == "he":
        apply_rtl_styling()

    # Render sidebar
    render_sidebar()

    # Main chat area
    st.title("I am your Medical AI Assistant 👋")

    # Display conversation history
    render_chat_messages()

    # Chat input
    user_input = st.chat_input(
        "Type your message..." if st.session_state.language == "en" else "הקלד הודעה..."
    )

    if user_input:
        send_message(user_input)


if __name__ == "__main__":
    main()
