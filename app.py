import streamlit as st
import json
import os
from google import genai
from google.genai import types

# --- 1. CONFIG ---
PROJECT_ID = st.secrets["PROJECT_ID"]
LOCATION = "global" 
MODEL_ID = "gemini-3.1-pro-preview"

# --- 2. AUTH ---
if "gcp_service_account" in st.secrets:
    with open("gcp_key.json", "w") as f:
        json.dump(dict(st.secrets["gcp_service_account"]), f)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "gcp_key.json"

client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)

# --- 3. UI ---
st.set_page_config(page_title="Legal Strategist", layout="wide")
st.title("⚖️ Principal Legal Advisor")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- 4. ENGINE ---
if prompt := st.chat_input("Enter asset details..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # We use st.empty() to stream the "Synthesis" status without blocking the UI
        status_placeholder = st.empty()
        status_placeholder.info("⏳ Initializing Deep Think Synthesis...")

        try:
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=prompt,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(include_thoughts=True),
                    temperature=0.0
                )
            )

            # Clear the "Waiting" status
            status_placeholder.empty()

            # --- PARSING FOR READABILITY ---
            for candidate in response.candidates:
                for part in candidate.content.parts:
                    if part.thought:
                        # Move Reasoning to the MAIN window, but style it as a technical block
                        with st.expander("🔍 STRATEGIC REASONING (INTERNAL LOGIC)", expanded=True):
                            st.info(part.text)
                    
                    if part.text:
                        # Clean Markdown Formatting for the Final Answer
                        st.subheader("Final Legal Strategy")
                        st.markdown("---")
                        st.markdown(part.text)
                        st.session_state.messages.append({"role": "assistant", "content": part.text})

        except Exception as e:
            status_placeholder.error(f"Logic Engine Error: {e}")
