import streamlit as st
import json
import os
import re # Added for text cleaning
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

# --- 3. TEXT CLEANING UTILITY ---
def clean_legal_text(text):
    """Fixes common 2026 SDK formatting glitches like missing spaces and LaTeX artifacts."""
    if not text:
        return ""
    # Fix missing spaces after common words/punctuation
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text) 
    # Fix specific dollar sign and math symbol artifacts
    text = text.replace("add−back", "add-back").replace("S$", "S$ ")
    # Ensure paragraphs have proper spacing
    text = text.replace("\n", "\n\n")
    return text

# --- 4. UI ---
st.set_page_config(page_title="Legal Advisor", layout="wide")
st.title("⚖️ Principal Legal Advisor")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display History with cleaned formatting
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(clean_legal_text(msg["content"]))

# --- 5. ENGINE ---
if prompt := st.chat_input("Enter details regarding the S$160k or mortgage..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        status_placeholder = st.empty()
        status_placeholder.info("⏳ Synthesizing Legal Logic...")

        try:
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=prompt,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(include_thoughts=True),
                    temperature=0.0
                )
            )

            status_placeholder.empty()

            for candidate in response.candidates:
                for part in candidate.content.parts:
                    if part.thought:
                        with st.expander("🔍 INTERNAL STRATEGIC REASONING", expanded=False):
                            # Cleaning the internal thoughts too
                            st.info(clean_legal_text(part.text))
                    
                    if part.text:
                        # Clean the final output before displaying
                        clean_output = clean_legal_text(part.text)
                        st.subheader("Final Legal Strategy")
                        st.markdown("---")
                        st.markdown(clean_output)
                        st.session_state.messages.append({"role": "assistant", "content": clean_output})

        except Exception as e:
            status_placeholder.error(f"Logic Engine Error: {e}")
