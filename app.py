import os
import streamlit as st
from google import genai
from google.genai import types
from streamlit_mic_recorder import mic_recorder


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Personal Finance & Maintenance AI",
    page_icon="⚙️",
    layout="wide"
)


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-2.0-flash"
)

API_KEY = os.getenv("GEMINI_API_KEY", "")


# ============================================================
# SYSTEM PROMPTS
# ============================================================

SYSTEM_PROMPTS = {

    "💰 Financial Budgeting": """
You are a helpful personal finance assistant.

Help users:
- Create budgets
- Track expenses
- Identify unnecessary spending
- Build savings plans
- Understand financial concepts
- Compare financial options

Important:
- Do not claim to be a licensed financial advisor.
- Do not guarantee investment returns.
- Clearly mention uncertainty and risk when discussing investments.
- Focus on educational and practical budgeting guidance.
""",

    "🔧 Home & Car Maintenance": """
You are an experienced home and car maintenance assistant.

Help users:
- Troubleshoot common maintenance problems
- Create maintenance schedules
- Estimate possible repair costs
- Explain DIY maintenance steps
- Identify when professional help is needed

Important:
- Prioritize safety.
- Warn users when electrical, gas, structural, or dangerous mechanical work
  should be handled by a qualified professional.
""",

    "📊 Expense Analyzer": """
You are an expense analysis assistant.

Help users:
- Categorize expenses
- Analyze spending patterns
- Find unnecessary costs
- Identify potential savings
- Create monthly budget recommendations

Present results clearly using:
- Categories
- Tables when useful
- Totals
- Savings recommendations
""",

    "🤖 General AI Assistant": """
You are a helpful, fast, friendly personal AI assistant.

You help with:
- Daily tasks
- Personal organization
- Expenses
- Budgeting
- Home maintenance
- Car maintenance
- Planning
- General questions

Be concise, practical, and clear.
"""
}


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "api_key" not in st.session_state:
    st.session_state.api_key = API_KEY

if "mode" not in st.session_state:
    st.session_state.mode = "🤖 General AI Assistant"

if "audio_processed" not in st.session_state:
    st.session_state.audio_processed = None


# ============================================================
# CACHED GEMINI CLIENT
# ============================================================

@st.cache_resource
def get_gemini_client(api_key):

    return genai.Client(
        api_key=api_key
    )


# ============================================================
# BUILD CHAT HISTORY
# ============================================================

def build_conversation():

    conversation = []

    for message in st.session_state.messages:

        role = message["role"]

        # Gemini expects user/model style roles
        if role == "assistant":
            role = "model"

        conversation.append(
            types.Content(
                role=role,
                parts=[
                    types.Part(
                        text=message["content"]
                    )
                ]
            )
        )

    return conversation


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ AI Controls")

    if not st.session_state.api_key:

        entered_key = st.text_input(
            "Gemini API Key",
            type="password",
            placeholder="Enter your Gemini API key"
        )

        if entered_key:
            st.session_state.api_key = entered_key

    else:

        st.success("API Key Connected")

    st.divider()

    st.subheader("🎯 Focus Area")

    st.session_state.mode = st.radio(
        "Choose AI Mode",
        list(SYSTEM_PROMPTS.keys()),
        index=list(SYSTEM_PROMPTS.keys()).index(
            st.session_state.mode
        )
    )

    st.divider()

    st.subheader("⚡ Model Settings")

    model_name = st.text_input(
        "Gemini Model",
        value=DEFAULT_MODEL
    )

    st.divider()

    if st.button(
        "🗑️ Clear Chat History",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()

    st.divider()

    st.caption(
        "⚡ Streaming enabled • "
        "💬 Conversation memory • "
        "🎙️ Voice input"
    )


# ============================================================
# MAIN HEADER
# ============================================================

st.title("⚙️ Personal Finance & Maintenance AI")

st.caption(
    "Your all-in-one AI assistant for budgeting, expenses, "
    "home maintenance, car maintenance, and everyday tasks."
)


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# ============================================================
# RESPONSE GENERATION FUNCTION
# ============================================================

def generate_ai_response(user_message):

    client = get_gemini_client(
        st.session_state.api_key
    )

    conversation = build_conversation()

    # Add latest user message
    conversation.append(
        types.Content(
            role="user",
            parts=[
                types.Part(
                    text=user_message
                )
            ]
        )
    )

    response = client.models.generate_content_stream(

        model=model_name,

        contents=conversation,

        config=types.GenerateContentConfig(

            system_instruction=
            SYSTEM_PROMPTS[
                st.session_state.mode
            ],

            temperature=0.7,

            max_output_tokens=2048
        )
    )

    return response


# ============================================================
# PROCESS USER MESSAGE
# ============================================================

def process_message(user_message):

    if not user_message:
        return

    if not st.session_state.api_key:

        st.error(
            "Please enter your Gemini API Key in the sidebar."
        )

        return


    # Add user message to history
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_message
        }
    )


    # Display user message
    with st.chat_message("user"):

        st.markdown(user_message)


    # AI Response
    with st.chat_message("assistant"):

        placeholder = st.empty()

        full_response = ""


        try:

            stream = generate_ai_response(
                user_message
            )


            for chunk in stream:

                if hasattr(chunk, "text") and chunk.text:

                    full_response += chunk.text

                    placeholder.markdown(
                        full_response + "▌"
                    )


            placeholder.markdown(
                full_response
            )


            # Save assistant response
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": full_response
                }
            )


        except Exception as error:

            st.error(
                f"AI Error: {error}"
            )


# ============================================================
# VOICE INPUT
# ============================================================

with st.expander("🎙️ Voice Input"):

    audio = mic_recorder(

        start_prompt="🎙️ Start Recording",

        stop_prompt="⏹️ Stop Recording",

        just_once=True,

        key="voice_recorder"
    )


    if audio:

        st.audio(
            audio["bytes"]
        )

        st.info(
            "Voice recording received. "
            "Audio-to-text processing can be connected here."
        )


# ============================================================
# SINGLE CHAT INPUT
# ============================================================

prompt = st.chat_input(
    "Ask about budgets, expenses, home maintenance, cars, or anything else..."
)


if prompt:

    process_message(prompt)
