import streamlit as st
import json
import os
from anthropic import AnthropicVertex

# --- 1. SETTINGS & AUTH ---
st.set_page_config(page_title="Claude 4.6 Architect", layout="centered")

# Initialize Service Account from Secrets
if "gcp_service_account" in st.secrets:
    try:
        service_account_info = dict(st.secrets["gcp_service_account"])
        with open("gcp_key.json", "w") as f:
            json.dump(service_account_info, f)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "gcp_key.json"
    except Exception as e:
        st.error(f"Auth Error: {e}")

PROJECT_ID = st.secrets.get("PROJECT_ID")
REGION = st.secrets.get("REGION", "global") # Default to global if not set

# Initialize the Vertex Client
# Using the 2026 'global' location logic
client = AnthropicVertex(project_id=PROJECT_ID, region=REGION)

# --- 2. CHAT UI ---
st.title("🤖 Claude 4.6 Opus")
st.caption(f"Connected to Vertex AI | Region: {REGION}")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 3. CHAT LOGIC ---
if prompt := st.chat_input("How can I help with your AI architecture today?"):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Response
    with st.chat_message("assistant"):
        try:
            # Model ID for the 2026 Opus release
            # Note: 4.6 Opus supports a 1M context window
            response = client.messages.create(
                model="claude-opus-4-6", 
                max_tokens=4096,
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ]
            )
            
            full_response = response.content[0].text
            st.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            st.error(f"Vertex AI Error: {e}")
            if "403" in str(e):
                st.info("Check if 'Vertex AI User' and 'Service Usage Consumer' roles are assigned to your service account.")
