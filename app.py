import os
import streamlit as st
from google import genai

# Streamlit Page Setup
st.set_page_config(page_title="Finance & Maintenance AI", page_icon="⚙️", layout="centered")
st.title("⚙️ Personal Finance & Maintenance AI")

# Retrieve API Key securely
api_key = os.environ.get("GEMINI_API_KEY", "")

with st.sidebar:
    st.header("⚙️ App Controls")
    if not api_key:
        api_key = st.text_input("Gemini API Key:", type="password")
    
    st.divider()
    # Topic quick selector
    mode = st.radio(
        "Focus Area:",
        ["💰 Financial Budgeting", "🔧 Home & Car Maintenance", "📊 Expense Analyzer", "🤖 General AI Assistant"]
    )
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Define System Persona based on selected mode
SYSTEM_PROMPTS = {
    "💰 Financial Budgeting": "You are a professional financial advisor. Help users track expenses, create budgets, save money, and make smart investment or savings decisions.",
    "🔧 Home & Car Maintenance": "You are a master handyman and auto mechanic. Help users troubleshoot maintenance issues, plan preventive maintenance schedules, and calculate repair costs.",
    "📊 Expense Analyzer": "You are a financial analyst. Break down costs, analyze receipts or text expenses provided by the user, and give money-saving recommendations.",
    "🤖 General AI Assistant": "You are a friendly personal assistant specializing in managing daily tasks, expenses, and asset maintenance."
}

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render previous chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User prompt handling
if prompt := st.chat_input("Ask about financial advice, budgets, or maintenance..."):
    if not api_key:
        st.error("Please add your Gemini API Key in the sidebar.")
        st.stop()

    # Append user prompt
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Stream response from Gemini
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            client = genai.Client(api_key=api_key)
            
            # Request response with custom persona
            response = client.interactions.create(
                model="gemini-3.6-flash",
                system_instruction=SYSTEM_PROMPTS[mode],
                input=prompt,
                stream=True
            )
            
            for chunk in response:
                if hasattr(chunk, 'delta') and hasattr(chunk.delta, 'text') and chunk.delta.text:
                    full_response += chunk.delta.text
                    message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            st.error(f"Error generating response: {e}")
