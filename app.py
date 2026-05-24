import streamlit as st
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

st.title("AI Chatbot")

if "history" not in st.session_state:
    st.session_state.history = []

for message in st.session_state.history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

user_input = st.chat_input("Ask me anything...")

if user_input:
    st.session_state.history.append({"role": "user", "content": user_input})
    
    with st.chat_message("user"):
        st.write(user_input)

    with st.spinner("Thinking..."):
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=st.session_state.history
        )
        ai_answer = response.choices[0].message.content

    st.session_state.history.append({"role": "assistant", "content": ai_answer})
    
    with st.chat_message("assistant"):
        st.write(ai_answer)