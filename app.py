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
    "gemini-3.6-flash"
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
import os
import io
import sqlite3
import hashlib
from datetime import datetime, date

import pandas as pd
import plotly.express as px
import streamlit as st

from google import genai
from google.genai import types

from pypdf import PdfReader
from streamlit_mic_recorder import mic_recorder
from gtts import gTTS


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Personal AI Super App",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.6-flash"
)

API_KEY = os.getenv(
    "GEMINI_API_KEY",
    ""
)

DATABASE_FILE = "personal_ai.db"


# ============================================================
# DATABASE
# ============================================================

def get_database():

    connection = sqlite3.connect(
        DATABASE_FILE,
        check_same_thread=False
    )

    return connection


def initialize_database():

    connection = get_database()

    cursor = connection.cursor()

    # Expenses table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS expenses (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        expense_date TEXT,
        category TEXT,
        description TEXT,
        amount REAL
    )
    """)

    # Maintenance table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS maintenance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        item TEXT,
        category TEXT,
        description TEXT,
        next_date TEXT,
        status TEXT
    )
    """)

    # User memory
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS memories (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        memory_type TEXT,
        content TEXT,
        created_at TEXT
    )
    """)

    connection.commit()

    connection.close()


initialize_database()


# ============================================================
# GEMINI CLIENT
# ============================================================

@st.cache_resource
def get_gemini_client(api_key):

    return genai.Client(
        api_key=api_key
    )


# ============================================================
# CACHE
# ============================================================

@st.cache_data(ttl=300)
def cached_category(description):

    return categorize_expense(description)


# ============================================================
# EXPENSE CATEGORY FUNCTION
# ============================================================

def categorize_expense(description):

    text = str(description).lower()

    categories = {

        "Food": [
            "restaurant",
            "food",
            "pizza",
            "burger",
            "coffee",
            "cafe",
            "grocery",
            "supermarket"
        ],

        "Transport": [
            "fuel",
            "petrol",
            "diesel",
            "uber",
            "taxi",
            "bus",
            "train"
        ],

        "Shopping": [
            "amazon",
            "clothes",
            "shoes",
            "shopping"
        ],

        "Bills": [
            "electricity",
            "internet",
            "wifi",
            "mobile",
            "phone",
            "water"
        ],

        "Entertainment": [
            "netflix",
            "spotify",
            "movie",
            "game"
        ],

        "Health": [
            "doctor",
            "hospital",
            "medicine",
            "pharmacy"
        ]
    }

    for category, keywords in categories.items():

        for keyword in keywords:

            if keyword in text:

                return category

    return "Other"


# ============================================================
# SYSTEM PROMPTS
# ============================================================

SYSTEM_PROMPTS = {

    "🤖 General AI": """
You are a helpful personal AI assistant.

Help with:
- Daily tasks
- Planning
- Organization
- General questions

Be practical, fast and clear.
""",

    "💰 Finance": """
You are a personal finance assistant.

Help users:
- Create budgets
- Analyze expenses
- Build savings plans
- Understand financial concepts

Do not guarantee financial returns.
Explain risks and uncertainty clearly.
Provide educational guidance.
""",

    "🔧 Maintenance": """
You are a home and car maintenance assistant.

Help users:
- Troubleshoot problems
- Create maintenance schedules
- Plan preventive maintenance
- Estimate possible repair costs

Always prioritize safety.

For dangerous electrical, gas, structural,
or serious mechanical problems,
recommend contacting a qualified professional.
""",

    "📊 Expense Analyzer": """
You are an expense analysis AI.

Analyze spending data.

Provide:
- Categories
- Spending patterns
- Monthly summaries
- Potential savings
- Practical recommendations

Use tables when useful.
""",

    "📄 Document AI": """
You are a document analysis assistant.

Analyze:
- PDFs
- Receipts
- Invoices
- Financial documents

Provide clear summaries and extract
important information.
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
    st.session_state.mode = "🤖 General AI"

if "document_text" not in st.session_state:
    st.session_state.document_text = ""

if "document_name" not in st.session_state:
    st.session_state.document_name = ""

if "expenses_df" not in st.session_state:
    st.session_state.expenses_df = pd.DataFrame()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("⚙️ AI Controls")

    if not st.session_state.api_key:

        entered_key = st.text_input(
            "Gemini API Key",
            type="password"
        )

        if entered_key:
            st.session_state.api_key = entered_key

    else:

        st.success("Gemini Connected")

    st.divider()

    st.subheader("🤖 AI Mode")

    st.session_state.mode = st.selectbox(
        "Select Mode",
        list(SYSTEM_PROMPTS.keys())
    )

    st.divider()

    model_name = st.text_input(
        "Gemini Model",
        value=DEFAULT_MODEL
    )

    st.divider()

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        st.session_state.messages = []
        st.rerun()

    st.caption(
        "⚡ Streaming enabled"
    )


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def build_conversation():

    conversation = []

    for message in st.session_state.messages:

        role = message["role"]

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
# GENERATE AI RESPONSE
# ============================================================

def generate_ai_response(user_message):

    client = get_gemini_client(
        st.session_state.api_key
    )

    conversation = build_conversation()

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
# PROCESS CHAT MESSAGE
# ============================================================

def process_message(user_message):

    if not user_message:
        return

    if not st.session_state.api_key:

        st.error(
            "Please enter your Gemini API Key."
        )

        return

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_message
        }
    )

    with st.chat_message("user"):

        st.markdown(user_message)

    with st.chat_message("assistant"):

        placeholder = st.empty()

        full_response = ""

        try:

            stream = generate_ai_response(
                user_message
            )

            for chunk in stream:

                if chunk.text:

                    full_response += chunk.text

                    placeholder.markdown(
                        full_response + "▌"
                    )

            placeholder.markdown(
                full_response
            )

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
# MAIN TITLE
# ============================================================

st.title("🤖 Personal AI Super App")

st.caption(
    "AI Chat • Finance • Expenses • Maintenance • Documents • Voice"
)


# ============================================================
# TABS
# ============================================================

tabs = st.tabs([

    "💬 AI Chat",

    "💰 Finance",

    "📊 Expenses",

    "🔧 Maintenance",

    "📄 Document AI",

    "🎙️ Voice AI",

    "🧠 AI Memory"

])


# ============================================================
# TAB 1 - AI CHAT
# ============================================================

with tabs[0]:

    st.header("💬 AI Chat")

    for message in st.session_state.messages:

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )

    prompt = st.chat_input(
        "Ask your AI assistant..."
    )

    if prompt:

        process_message(prompt)


# ============================================================
# TAB 2 - FINANCE
# ============================================================

with tabs[1]:

    st.header("💰 Personal Finance")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("📋 Budget Planner")

        monthly_income = st.number_input(
            "Monthly Income",
            min_value=0.0,
            value=3000.0
        )

        monthly_expenses = st.number_input(
            "Monthly Expenses",
            min_value=0.0,
            value=1500.0
        )

        savings = (
            monthly_income -
            monthly_expenses
        )

        st.metric(
            "Potential Monthly Savings",
            f"${savings:,.2f}"
        )

        if monthly_income > 0:

            savings_rate = (
                savings /
                monthly_income
            ) * 100

            st.metric(
                "Savings Rate",
                f"{savings_rate:.1f}%"
            )

    with col2:

        st.subheader("💡 Savings Calculator")

        monthly_saving = st.number_input(
            "Monthly Saving Amount",
            min_value=0.0,
            value=200.0
        )

        months = st.number_input(
            "Number of Months",
            min_value=1,
            value=12
        )

        total_savings = (
            monthly_saving *
            months
        )

        st.metric(
            "Total Savings",
            f"${total_savings:,.2f}"
        )


# ============================================================
# TAB 3 - EXPENSE ANALYZER
# ============================================================

with tabs[2]:

    st.header("📊 Expense Analyzer")

    st.subheader("➕ Add Expense")

    with st.form("expense_form"):

        expense_date = st.date_input(
            "Date",
            value=date.today()
        )

        description = st.text_input(
            "Description"
        )

        amount = st.number_input(
            "Amount",
            min_value=0.0
        )

        submitted = st.form_submit_button(
            "Add Expense"
        )

        if submitted and description:

            category = categorize_expense(
                description
            )

            connection = get_database()

            cursor = connection.cursor()

            cursor.execute(
                """
                INSERT INTO expenses
                (
                    expense_date,
                    category,
                    description,
                    amount
                )
                VALUES (?, ?, ?, ?)
                """,
                (
                    str(expense_date),
                    category,
                    description,
                    amount
                )
            )

            connection.commit()

            connection.close()

            st.success(
                f"Expense added to {category}"
            )

    st.divider()

    st.subheader("📁 Upload CSV or Excel")

    uploaded_file = st.file_uploader(

        "Upload expense data",

        type=[
            "csv",
            "xlsx",
            "xls"
        ]
    )

    if uploaded_file:

        try:

            if uploaded_file.name.endswith(
                ".csv"
            ):

                df = pd.read_csv(
                    uploaded_file
                )

            else:

                df = pd.read_excel(
                    uploaded_file
                )

            st.write(
                "Uploaded Data"
            )

            st.dataframe(
                df,
                use_container_width=True
            )

            if "description" in df.columns:

                df["category"] = df[
                    "description"
                ].apply(
                    categorize_expense
                )

            st.session_state.expenses_df = df

        except Exception as error:

            st.error(
                f"Upload error: {error}"
            )

    st.divider()

    st.subheader("📈 Expense History")

    connection = get_database()

    expenses = pd.read_sql_query(
        "SELECT * FROM expenses",
        connection
    )

    connection.close()

    if not expenses.empty:

        st.dataframe(
            expenses,
            use_container_width=True
        )

        total_expenses = expenses[
            "amount"
        ].sum()

        st.metric(
            "Total Expenses",
            f"${total_expenses:,.2f}"
        )

        category_data = expenses.groupby(
            "category",
            as_index=False
        )["amount"].sum()

        chart = px.pie(
            category_data,
            values="amount",
            names="category",
            title="Expenses by Category"
        )

        st.plotly_chart(
            chart,
            use_container_width=True
        )

    else:

        st.info(
            "No expenses added yet."
        )


# ============================================================
# TAB 4 - MAINTENANCE
# ============================================================

with tabs[3]:

    st.header("🔧 Maintenance Manager")

    st.subheader(
        "➕ Add Maintenance Task"
    )

    with st.form("maintenance_form"):

        item = st.text_input(
            "Item",
            placeholder="Example: Toyota Car"
        )

        maintenance_category = st.selectbox(
            "Category",
            [
                "🚗 Car",
                "🏠 Home",
                "🔧 Other"
            ]
        )

        maintenance_description = st.text_area(
            "Maintenance Task"
        )

        next_date = st.date_input(
            "Next Maintenance Date"
        )

        submitted = st.form_submit_button(
            "Add Maintenance"
        )

        if submitted and item:

            connection = get_database()

            cursor = connection.cursor()

            cursor.execute(
                """
                INSERT INTO maintenance
                (
                    item,
                    category,
                    description,
                    next_date,
                    status
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    item,
                    maintenance_category,
                    maintenance_description,
                    str(next_date),
                    "Pending"
                )
            )

            connection.commit()

            connection.close()

            st.success(
                "Maintenance task added."
            )

    st.divider()

    connection = get_database()

    maintenance_df = pd.read_sql_query(
        "SELECT * FROM maintenance",
        connection
    )

    connection.close()

    if not maintenance_df.empty:

        st.dataframe(
            maintenance_df,
            use_container_width=True
        )

        today = pd.Timestamp.today()

        maintenance_df["next_date"] = pd.to_datetime(
            maintenance_df["next_date"]
        )

        overdue = maintenance_df[
            maintenance_df["next_date"] < today
        ]

        if not overdue.empty:

            st.warning(
                f"⚠️ {len(overdue)} maintenance task(s) overdue!"
            )

            st.dataframe(overdue)

    else:

        st.info(
            "No maintenance tasks yet."
        )


# ============================================================
# PDF TEXT EXTRACTION
# ============================================================

def extract_pdf_text(pdf_file):

    reader = PdfReader(
        pdf_file
    )

    text = ""

    for page in reader.pages:

        extracted = page.extract_text()

        if extracted:

            text += extracted

    return text


# ============================================================
# TAB 5 - DOCUMENT AI
# ============================================================

with tabs[4]:

    st.header("📄 Document AI")

    uploaded_pdf = st.file_uploader(

        "Upload PDF, Receipt or Invoice",

        type=["pdf"]
    )

    if uploaded_pdf:

        try:

            document_text = extract_pdf_text(
                uploaded_pdf
            )

            st.session_state.document_text = (
                document_text
            )

            st.session_state.document_name = (
                uploaded_pdf.name
            )

            st.success(
                f"Loaded: {uploaded_pdf.name}"
            )

            with st.expander(
                "Preview Extracted Text"
            ):

                st.text(
                    document_text[:5000]
                )

        except Exception as error:

            st.error(
                f"PDF Error: {error}"
            )

    st.divider()

    document_question = st.text_input(
        "Ask a question about the uploaded document"
    )

    if st.button(
        "🔍 Analyze Document"
    ):

        if not st.session_state.document_text:

            st.warning(
                "Please upload a PDF first."
            )

        elif not document_question:

            st.warning(
                "Please enter a question."
            )

        else:

            try:

                client = get_gemini_client(
                    st.session_state.api_key
                )

                # Limit context for faster response
                document_context = (
                    st.session_state.document_text[:20000]
                )

                prompt = f"""
Document:

{document_context}

Question:

{document_question}

Analyze the document and answer clearly.
"""

                response = client.models.generate_content(

                    model=model_name,

                    contents=prompt,

                    config=types.GenerateContentConfig(

                        system_instruction=
                        SYSTEM_PROMPTS[
                            "📄 Document AI"
                        ]
                    )
                )

                st.subheader(
                    "AI Analysis"
                )

                st.write(
                    response.text
                )

            except Exception as error:

                st.error(
                    f"Document AI Error: {error}"
                )


# ============================================================
# VOICE TRANSCRIPTION
# ============================================================

def transcribe_audio(
    audio_bytes,
    mime_type="audio/wav"
):

    client = get_gemini_client(
        st.session_state.api_key
    )

    response = client.models.generate_content(

        model=model_name,

        contents=[
            types.Content(
                role="user",
                parts=[
                    types.Part.from_bytes(
                        data=audio_bytes,
                        mime_type=mime_type
                    ),
                    types.Part(
                        text="""
Transcribe this audio accurately.

Return only the spoken text.
"""
                    )
                ]
            )
        ]
    )

    return response.text


# ============================================================
# TAB 6 - VOICE AI
# ============================================================

with tabs[5]:

    st.header("🎙️ Voice AI")

    st.subheader(
        "🎤 Speech to Text"
    )

    audio = mic_recorder(

        start_prompt="🎙️ Start Recording",

        stop_prompt="⏹️ Stop Recording",

        just_once=True,

        key="voice_recorder"
    )

    if audio:

        audio_bytes = audio["bytes"]

        st.audio(
            audio_bytes
        )

        if st.button(
            "📝 Transcribe Voice"
        ):

            try:

                with st.spinner(
                    "Transcribing..."
                ):

                    transcription = (
                        transcribe_audio(
                            audio_bytes
                        )
                    )

                st.success(
                    "Transcription complete!"
                )

                st.text_area(
                    "Transcription",
                    transcription,
                    height=150
                )

                if st.button(
                    "💬 Send to AI Chat"
                ):

                    process_message(
                        transcription
                    )

            except Exception as error:

                st.error(
                    f"Voice Error: {error}"
                )

    st.divider()

    st.subheader(
        "🔊 Text to Speech"
    )

    tts_text = st.text_area(
        "Enter text to speak"
    )

    if st.button(
        "🔊 Generate Voice"
    ):

        if tts_text:

            try:

                tts = gTTS(
                    text=tts_text,
                    lang="en"
                )

                audio_buffer = io.BytesIO()

                tts.write_to_fp(
                    audio_buffer
                )

                audio_buffer.seek(0)

                st.audio(
                    audio_buffer,
                    format="audio/mp3"
                )

            except Exception as error:

                st.error(
                    f"TTS Error: {error}"
                )


# ============================================================
# AI MEMORY
# ============================================================

with tabs[6]:

    st.header("🧠 AI Memory")

    st.caption(
        "Store useful information for your personal AI."
    )

    memory_type = st.selectbox(
        "Memory Type",
        [
            "User Preference",
            "Budget History",
            "Maintenance History",
            "Important Note"
        ]
    )

    memory_content = st.text_area(
        "What should the AI remember?"
    )

    if st.button(
        "💾 Save Memory"
    ):

        if memory_content:

            connection = get_database()

            cursor = connection.cursor()

            cursor.execute(
                """
                INSERT INTO memories
                (
                    memory_type,
                    content,
                    created_at
                )
                VALUES (?, ?, ?)
                """,
                (
                    memory_type,
                    memory_content,
                    datetime.now().isoformat()
                )
            )

            connection.commit()

            connection.close()

            st.success(
                "Memory saved!"
            )

    st.divider()

    st.subheader(
        "Stored Memories"
    )

    connection = get_database()

    memories = pd.read_sql_query(
        "SELECT * FROM memories ORDER BY id DESC",
        connection
    )

    connection.close()

    if not memories.empty:

        st.dataframe(
            memories,
            use_container_width=True
        )

    else:

        st.info(
            "No memories saved yet."
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🤖 Personal AI Super App • "
    "Streaming • Memory • Finance • Maintenance • Documents • Voice"
)


if prompt:

    process_message(prompt)
