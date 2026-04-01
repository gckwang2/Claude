import streamlit as st
import json
import os
from anthropic import AnthropicVertex

# Ensure no spaces exist before the imports above
# --- Setup Credentials ---
if "gcp_service_account" in st.secrets:
    service_account_info = dict(st.secrets["gcp_service_account"])
    with open("gcp_key.json", "w") as f:
        json.dump(service_account_info, f)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "gcp_key.json"

PROJECT_ID = st.secrets.get("PROJECT_ID")
REGION = st.secrets.get("REGION", "us-central1")

# Initialize Claude 4.6
client = AnthropicVertex(project_id=PROJECT_ID, region=REGION)

st.title("💬 Claude 4.6 Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask Claude 4.6..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            # Using the 4.6 model ID
            message = client.messages.create(
                model="claude-sonnet-4-6", 
                max_tokens=2048,
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ]
            )
            response_text = message.content[0].text
            st.markdown(response_text)
            st.session_state.messages.append({"role": "assistant", "content": response_text})
        except Exception as e:
            st.error(f"Error: {e}")
