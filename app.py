import streamlit as st
import json
import os
import re
from google import genai
from google.genai import types

# --- 1. CONFIGURATION ---
PROJECT_ID = st.secrets["PROJECT_ID"]
LOCATION = "global" 
MODEL_ID = "gemini-3.1-pro-preview"

# --- 2. ENHANCED LEGAL STRATEGY PROMPT ---
LEGAL_PROMPT = """
You are a Senior Legal Advisor specialized in Singapore Family Law. 
GOAL: Help the user achieve a 75:25 asset division ratio for Auxiliary Matters (AM).

PRECEDENTS:
- TQU v TQT [2020] SGCA 8: Use adverse inference logic for non-disclosure to secure an 'uplift' (75:25 target).
- ANJ v ANK: 3-step structured approach (Direct vs. Indirect contributions).

REVISION PROTOCOL:
- CRITICAL: Analyze the user's input for any "lapses" (e.g., missing bank statements, vague "family expense" claims, or untraced fund transfers like the S$160k).
- MANDATE: After your analysis, provide a "REVISED RESPONSE" for the user to submit to court. This revision must use precise legal language and close all evidential gaps to prevent adverse inference.
- PROOF: If proof is missing, explicitly tell the user WHAT document is needed to satisfy the court's tracing requirement.
"""

# --- 3. AUTHENTICATION & CLIENT ---
if "gcp_service_account" in st.secrets:
    with open("gcp_key.json", "w") as f:
        json.dump(dict(st.secrets["gcp_service_account"]), f)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "gcp_key.json"

client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)

# --- 4. TEXT CLEANING UTILITY ---
def clean_legal_text(text):
    if not text: return ""
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text) 
    text = text.replace("add−back", "add-back").replace("S$", "S$ ")
    text = text.replace("\n", "\n\n")
    return text

# --- 5. UI SETUP ---
st.set_page_config(page_title="Legal Advisor", layout="wide")
st.title("⚖️ Principal Legal Advisor")
st.caption(f"Engine: {MODEL_ID} | Revision Logic Active ($300 Credit)")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- 6. CHAT ENGINE ---
if prompt := st.chat_input("Enter your draft response for the S$160k claim..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        status_placeholder = st.empty()
        status_placeholder.info("⏳ Synthesizing Gap-Free Legal Strategy...")

        try:
            full_input = f"{LEGAL_PROMPT}\n\nUSER DRAFT FOR REVISION: {prompt}"
            
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=full_input,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(include_thoughts=True),
                    temperature=0.0
                )
            )

            status_placeholder.empty()

            for candidate in response.candidates:
                for part in candidate.content.parts:
                    if part.thought:
                        with st.expander("🔍 GAP ANALYSIS (Internal Reasoning)", expanded=True):
                            st.info(clean_legal_text(part.text))
                    
                    if part.text:
                        clean_output = clean_legal_text(part.text)
                        st.subheader("Revised Legal Submission")
                        st.markdown("---")
                        st.markdown(clean_output)
                        st.session_state.messages.append({"role": "assistant", "content": clean_output})

        except Exception as e:
            status_placeholder.error(f"Logic Engine Error: {e}")
