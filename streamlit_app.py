# streamlit_app.py
import streamlit as st
import requests
import json
import os
from dotenv import load_dotenv

# --- Load API URL from environment variables ---
load_dotenv()
RENDER_API_URL = os.getenv("RENDER_API_URL")

if not RENDER_API_URL:
    st.error("RENDER_API_URL environment variable is not set. Please add it to your .env file.")
    st.stop()

st.title("Mental Health Virtual Assistant")

# --- Initialize chat history in a session state ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Display chat messages from history on app rerun ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Main chat input loop ---
if prompt := st.chat_input("How are you feeling today?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        response = requests.post(
            f"{RENDER_API_URL}/ask",
            headers={"Content-Type": "application/json"},
            data=json.dumps({"text": prompt})
        )
        response_json = response.json()
        agent_response = response_json.get("response", "Sorry, I am having trouble connecting to the service.")
    except Exception as e:
        agent_response = f"Sorry, there was an error connecting to the service: {e}"
        
    with st.chat_message("assistant"):
        st.markdown(agent_response)
        st.session_state.messages.append({"role": "assistant", "content": agent_response})