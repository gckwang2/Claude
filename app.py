import streamlit as st
import json
import os
import uuid
from google import genai
from google.genai import types
from pymilvus import connections, Collection, utility, FieldSchema, CollectionSchema, DataType

# --- 1. CONFIGURATION (HARDCODED REGION) ---
PROJECT_ID = st.secrets["PROJECT_ID"]
LOCATION = "global"  # Hardcoded to bypass secrets
MODEL_ID = "gemini-3.1-pro-preview"

# --- 2. AUTHENTICATION GATE ---
def check_password():
    if "password_correct" not in st.session_state:
        st.title("⚖️ Legal Advisor Login")
        user = st.text_input("Username")
        pw = st.text_input("Password", type="password")
        if st.button("Login"):
            if user == st.secrets["credentials"]["admin_user"] and pw == st.secrets["credentials"]["admin_password"]:
                st.session_state["password_correct"] = True
                st.rerun()
            else:
                st.error("Invalid credentials")
        return False
    return True

if not check_password():
    st.stop()

# --- 3. INITIALIZE VERTEX AI CLIENT ---
if "gcp_service_account" in st.secrets:
    with open("gcp_key.json", "w") as f:
        json.dump(dict(st.secrets["gcp_service_account"]), f)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "gcp_key.json"

client = genai.Client(
    vertexai=True, 
    project=PROJECT_ID, 
    location=LOCATION
)

# --- 4. ZILLIZ INITIALIZATION ---
def init_zilliz():
    connections.connect(uri=st.secrets["ZILLIZ_URI"], token=st.secrets["ZILLIZ_TOKEN"])
    col_name = "legal_memory"
    if not utility.has_collection(col_name):
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=768),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=5000),
            FieldSchema(name="session_id", dtype=DataType.VARCHAR, max_length=100)
        ]
        col = Collection(name=col_name, schema=CollectionSchema(fields))
        col.create_index("vector", {"metric_type": "L2", "index_type": "IVF_FLAT", "params": {"nlist": 128}})
    else:
        col = Collection(col_name)
    col.load()
    return col

collection = init_zilliz()

# --- 5. LEGAL STRATEGY PROMPT ---
LEGAL_PROMPT = """
You are a Senior Legal Advisor (Singapore Family Law).
Objective: Achieve 75:25 asset division via TQU v TQT [2020] SGCA 8.
Logic: Use Adverse Inference for non-disclosure. 
Strict Rule: Demand proof for every claim; if missing, instruct on how to infer gaps.
"""

# --- 6. CHAT UI ---
if "chat_id" not in st.session_state:
    st.session_state.chat_id = str(uuid.uuid4())[:8]
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("SG Legal Strategist")
st.caption(f"Engine: {MODEL_ID} | Region: {LOCATION}")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Describe the asset or claim..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

   with st.chat_message("assistant"):
        # The 'with' statement starts here
        with st.status("Initializing Legal Synthesis...", expanded=True) as status:
            try:
                # Everything inside this 'try' must be indented 4 spaces from 'try'
                st.write("🔍 Accessing Zilliz Cloud for case precedents...")
                # (Insert your vector search logic here)
                
                st.write("🧠 Engaging Gemini 3.1 Pro Reasoning Engine...")
                st.write("⚖️ Applying TQU v TQT [2020] SGCA 8 framework...")

                response = client.models.generate_content(
                    model="gemini-3.1-pro-preview",
                    contents=f"{LEGAL_PROMPT}\n\nUSER: {prompt}",
                    config=types.GenerateContentConfig(
                        thinking_config=types.ThinkingConfig(include_thoughts=True),
                        temperature=0.0
                    )
                )

                # Process the response parts
                final_answer = ""
                for part in response.candidates[0].content.parts:
                    if part.thought:
                        # This fulfills the 'synthesis' requirement during the wait
                        st.write(f"**Strategic Reasoning:** {part.text}")
                    else:
                        final_answer += part.text

                status.update(label="Synthesis Complete!", state="complete", expanded=False)
                
                # The final answer is displayed outside the status box
                st.markdown(final_answer)
                st.session_state.messages.append({"role": "assistant", "content": final_answer})

            except Exception as e:
                status.update(label="Synthesis Failed", state="error")
                st.error(f"Logic Engine Error: {e}")
