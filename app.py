import streamlit as st
import json
import os
from anthropic import AnthropicVertex

# --- 1. Authentication Setup ---
# This looks for the service account JSON stored in Streamlit Secrets
if "gcp_service_account" in st.secrets:
    try:
        service_account_info = dict(st.secrets["gcp_service_account"])
        # Vertex AI SDK requires a physical file path for the credentials
        with open("gcp_key.json", "w") as f:
            json.dump(service_account_info, f)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "gcp_key.json"
    except Exception as e:
        st.error(f"Authentication Setup Error: {e}")

# --- 2. Configuration ---
PROJECT_ID = st.secrets.get("PROJECT_ID", "gen-lang-client-0697333178")
REGION = st.secrets.get("REGION", "us-east5") # Ensure this matches where you enabled Opus

st.set_page_config(page_title="Claude 4.6 Opus Chat", layout="centered")
st.title("🤖 Claude 4.6 Opus")
st.caption("Powered by Google Vertex AI & Anthropic")

# --- 3. Initialize Client ---
try:
    client = AnthropicVertex(project_id=PROJECT_ID, region=REGION)
except Exception as e:
    st.error(f"Failed to initialize Vertex AI client: {e}")

# --- 4. Chat History Logic ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 5. Chat Input & Response ---
if prompt := st.chat_input("Message Claude 4.6 Opus..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant response
    with st.chat_message("assistant"):
        try:
            # Note: Using the specific Opus 4.6 ID
            response = client.messages.create(
                model="claude-4-6-opus@20260217", 
                max_tokens=4096,
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ]
            )
            
            full_response = response.content[0].text
            st.markdown(full_response)
            
            # Save to history
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            st.error(f"Vertex AI Error: {e}")
            if "404" in str(e):
                st.info("Tip: Double-check that your REGION matches where Opus 4.6 is enabled.")
            elif "403" in str(e):
                st.info("Tip: Check your IAM permissions or Quotas in Google Cloud Console.")
