import streamlit as st
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

st.title("🤖 AI Chatbot")

# Sidebar controls
with st.sidebar:
    st.header("Settings")
    
    
   
    model = st.selectbox("Choose Model", [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-120b"
])
    
    system_prompt = st.text_area(
        "System Prompt",
        value="You are a helpful, friendly AI assistant.",
        height=100
    )
    
    if st.button("🗑️ Clear Chat"):
        st.session_state.history = []
        st.rerun()

# Initialize history
if "history" not in st.session_state:
    st.session_state.history = []

# Display chat history
for message in st.session_state.history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User input
user_input = st.chat_input("Ask me anything...")

if user_input:
    st.session_state.history.append({"role": "user", "content": user_input})
    
    with st.chat_message("user"):
        st.write(user_input)

    with st.spinner("Thinking..."):
        # Keep last 20 messages to avoid token overflow
        trimmed_history = st.session_state.history[-20:]
        
        messages = [{"role": "system", "content": system_prompt}] + trimmed_history
        
        response = client.chat.completions.create(
            model=model,
            messages=messages
        )
        ai_answer = response.choices[0].message.content

    st.session_state.history.append({"role": "assistant", "content": ai_answer})
    
    with st.chat_message("assistant"):
        st.write(ai_answer)