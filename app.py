import streamlit as st
from anthropic import AnthropicVertex

# --- App Configuration ---
st.set_page_config(page_title="Claude Chatbot (Vertex AI)", layout="centered")
st.title("💬 Claude on Vertex AI")

# --- Setup Credentials ---
# In Streamlit Cloud, go to Settings -> Secrets and add:
# PROJECT_ID = "your-google-cloud-project-id"
# REGION = "us-central1" (or your preferred region)
PROJECT_ID = st.secrets.get("PROJECT_ID", "your-project-id")
REGION = st.secrets.get("REGION", "us-central1")

# Initialize the Claude Client for Vertex AI
client = AnthropicVertex(project_id=PROJECT_ID, region=REGION)

# --- Session State for Chat History ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Chat Input ---
if prompt := st.chat_input("Ask Claude anything..."):
    # 1. Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Generate Claude response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        try:
            # We use the Messages API which is compatible with Claude 3.5/3.0 models
            # Common models: "claude-3-5-sonnet-v2@20241022" or "claude-3-opus@20240229"
            message = client.messages.create(
                model="claude-3-5-sonnet@20240620", 
                max_tokens=1024,
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ]
            )
            
            full_response = message.content[0].text
            response_placeholder.markdown(full_response)
            
        except Exception as e:
            st.error(f"Error calling Vertex AI: {str(e)}")

    # 3. Save assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
