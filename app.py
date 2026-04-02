import streamlit as st
import json
import os
import uuid
from google import genai
from google.genai import types
from pymilvus import connections, Collection, utility

# --- 1. CONFIGURATION ---
# Hardcoding 'global' and 'us-central1' is the safest path for the $300 credit
PROJECT_ID = st.secrets["PROJECT_ID"]
LOCATION = "global" 
MODEL_ID = "gemini-3.1-pro-preview"

# --- 2. AUTHENTICATION & CLIENT ---
if "gcp_service_account" in st.secrets:
    with open("gcp_key.json", "w") as f:
        json.dump(dict(st.secrets["gcp_service_account"]), f)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "gcp_key.json"

# Initialize the Vertex AI Client
client = genai.Client(
    vertexai=True, 
    project=PROJECT_ID, 
    location=LOCATION
)

# --- 3. UI SETUP ---
st.set_page_config(page_title="SG Legal Strategist", page_icon="⚖️")
st.title("⚖️ Principal Legal Advisor")
st.caption(f"Connected to {MODEL_ID} via Vertex AI (Global Endpoint)")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- 4. LEGAL LOGIC ENGINE ---
LEGAL_PROMPT = """
You are a Senior Legal Advisor specialized in Singapore Family Law.
Precedent: TQU v TQT [2020] SGCA 8.
Strategy: Achieve 75:25 asset division through Adverse Inference logic for non-disclosure.
Tone: Grounded, neutral, and professional. 
Rule: Do not provide general advice; analyze the specific user claim for logic gaps.
"""

if prompt := st.chat_input("State the asset details or non-disclosure claim..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # The 'Synthesis' stage: Show progress during the wait
        with st.status("Initializing Legal Synthesis...", expanded=True) as status:
            try:
                st.write("🔍 Retrieving precedents from Zilliz Cloud...")
                # (Optional: Add your Zilliz search call here)
                
                st.write("🧠 Engaging Gemini 3.1 Pro 'Deep Think' reasoning...")
                st.write("⚖️ Applying TQU v TQT adverse inference framework...")

                # Call the model
                response = client.models.generate_content(
                    model=MODEL_ID,
                    contents=f"{LEGAL_PROMPT}\n\nUSER CLAIM: {prompt}",
                    config=types.GenerateContentConfig(
                        thinking_config=types.ThinkingConfig(include_thoughts=True),
                        temperature=0.0
                    )
                )

                # Separate Reasoning (Thoughts) from the Result
                final_answer = ""
                for part in response.candidates[0].content.parts:
                    if part.thought:
                        # This displays the 'Synthesis' thought process in real-time
                        st.write(f"**Strategic Reasoning:** {part.text}")
                    else:
                        final_answer += part.text

                status.update(label="Synthesis Complete", state="complete", expanded=False)

                if final_answer:
                    st.markdown(final_answer)
                    st.session_state.messages.append({"role": "assistant", "content": final_answer})
                else:
                    st.warning("The engine processed the request but returned no text. Check reasoning logs.")

            except Exception as e:
                status.update(label="Synthesis Error", state="error")
                st.error(f"Logic Engine Error: {e}")
