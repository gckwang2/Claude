import streamlit as st
import json
import os
import uuid
# --- NEW SDK IMPORTS ---
from google import genai
from google.genai import types
from pymilvus import connections, Collection, utility, FieldSchema, CollectionSchema, DataType

# --- 1. AUTH & CLIENT SETUP ---
# (Keeping your Service Account logic for Vertex AI)
if "gcp_service_account" in st.secrets:
    gcp_info = dict(st.secrets["gcp_service_account"])
    with open("gcp_key.json", "w") as f:
        json.dump(gcp_info, f)
    # The new SDK looks for this environment variable
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "gcp_key.json"

PROJECT_ID = st.secrets["PROJECT_ID"]
REGION = st.secrets.get("REGION", "us-east5")

# Initialize the NEW Gen AI Client for Vertex
client = genai.Client(
    vertexai=True, 
    project=PROJECT_ID, 
    location=REGION
)
# --- GLOBAL LEGAL STRATEGY PROMPT ---
LEGAL_PROMPT = """
You are a Senior Legal Advisor specialized in Singapore Family Law. 
GOAL: Help the user achieve a 75:25 asset division ratio for Auxiliary Matters (AM).

PRECEDENTS:
- TQU v TQT [2020] SGCA 8: Use adverse inference logic for non-disclosure to secure an 'uplift'.
- ANJ v ANK: 3-step structured approach (Direct vs. Indirect contributions).

EVIDENCE PROTOCOL:
- If a user makes a claim, ask for PROOF (bank statements, CPF, receipts).
- If proof is missing, explain how to INFER the asset existence via TQU v TQT.
- Warning: State clearly that claims without proof or logical inference will fail in court.
"""
# --- 2. THE CHAT LOOP WITH DEEP THINK ---
if prompt := st.chat_input("Detail your asset claim..."):
    # ... (Your Zilliz retrieval logic stays the same) ...

    with st.chat_message("assistant"):
        try:
            # NEW SDK CONFIGURATION
            # Setting thinking_level to 'high' for 75:25 strategy logic
            config = types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(include_thoughts=True),
                # Note: In the new SDK, 'thinking_level' is often handled 
                # dynamically by the model or explicitly in metadata
                temperature=0.0 
            )
            
            full_input = f"{LEGAL_PROMPT}\n\nUSER NEW CLAIM: {prompt}"
            
            # New call syntax
            response = client.models.generate_content(
                model="gemini-3.1-pro-preview", 
                contents=full_input,
                config=config
            )
            
            # Accessing parts (Thoughts + Text)
            answer = ""
            for part in response.candidates[0].content.parts:
                if part.thought:
                    # Optional: Show the reasoning in an expander
                    with st.expander("Advisor's Internal Reasoning"):
                        st.write(part.text)
                else:
                    answer += part.text
            
            st.markdown(answer)
            st.session_state.history[current_id].append({"role": "assistant", "content": answer})

        except Exception as e:
            st.error(f"Logic Engine Error: {e}")
