import streamlit as st
import json
import os
import re
import uuid
from google import genai
from google.genai import types
from pymilvus import connections, Collection, utility, FieldSchema, CollectionSchema, DataType

# --- 1. GLOBAL PROMPT & CONFIG ---
LEGAL_PROMPT = """
You are a Senior Legal Advisor specialized in Singapore Family Law. 
GOAL: Help the user achieve a 75:25 asset division ratio for Auxiliary Matters (AM).

PRECEDENTS:
- TQU v TQT [2020] SGCA 8: Use adverse inference logic for non-disclosure to secure an 'uplift' (75:25 target).
- ANJ v ANK: 3-step structured approach (Direct vs. Indirect contributions).

REVISION PROTOCOL:
- Analyze user input for "lapses" (e.g., missing bank statements, vague "family expense" claims, or untraced funds).
- Provide a "REVISED RESPONSE" for court submission using precise legal language to close all evidential gaps.
- Do not quote any case reference in the response, just infer to the court knowledge.
- Use a non aggressive tone, but firm and difficult to refute in revising the user reponse.
"""

PROJECT_ID = st.secrets["PROJECT_ID"]
LOCATION = "global" 
MODEL_ID = "gemini-3.1-pro-preview"
EMBED_MODEL = "text-embedding-004"

# --- 2. THE LOGIN GATE (Re-inserted) ---
def check_password():
    """Returns True if the user had the correct password."""
    def password_entered():
        if (
            st.session_state["username"] in st.secrets["passwords"]
            and st.session_state["password"]
            == st.secrets["passwords"][st.session_state["username"]]
        ):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store password
            del st.session_state["username"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("Username", on_change=None, key="username")
        st.text_input("Password", type="password", on_change=None, key="password")
        st.button("Log In", on_click=password_entered)
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("Username", on_change=None, key="username")
        st.text_input("Password", type="password", on_change=None, key="password")
        st.button("Log In", on_click=password_entered)
        st.error("😕 User not known or password incorrect")
        return False
    else:
        return True

if not check_password():
    st.stop()  # Do not run the rest of the app if not logged in

# --- 3. PERSISTENT IDENTITY ---
# We use the username or a fixed project ID to load the "Next Day" history
USER_IDENTITY = "Freddy_Legal_Project_2026"

# --- 4. AUTH & CLIENT ---
if "gcp_service_account" in st.secrets:
    with open("gcp_key.json", "w") as f:
        json.dump(dict(st.secrets["gcp_service_account"]), f)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "gcp_key.json"
client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)

# --- 5. ZILLIZ & UTILS ---
@st.cache_resource
def init_zilliz():
    connections.connect(uri=st.secrets["ZILLIZ_URI"], token=st.secrets["ZILLIZ_TOKEN"])
    col_name = "legal_memory_v2"
    if not utility.has_collection(col_name):
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=768),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=60000), 
            FieldSchema(name="session_id", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="role", dtype=DataType.VARCHAR, max_length=20)
        ]
        col = Collection(col_name, CollectionSchema(fields))
        col.create_index("vector", {"metric_type": "L2", "index_type": "IVF_FLAT", "params": {"nlist": 128}})
    else:
        col = Collection(col_name)
    col.load()
    return col

collection = init_zilliz()

def clean_legal_text(text):
    if not text: return ""
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = text.replace("add−back", "add-back").replace("S$", "S$ ")
    return text.replace("\n", "\n\n")

def load_history(session_id):
    try:
        results = collection.query(expr=f'session_id == "{session_id}"', output_fields=["text", "role"])
        return [{"role": r['role'], "content": r['text']} for r in sorted(results, key=lambda x: x['id'])]
    except: return []

# --- 6. UI & SESSION HYDRATION ---
st.set_page_config(page_title="Legal Advisor", layout="wide")
st.title("⚖️ Principal Legal Advisor")

if "messages" not in st.session_state:
    with st.spinner("🔄 Restoring previous day's consultations..."):
        st.session_state.messages = load_history(USER_IDENTITY)

if st.session_state.messages:
    with st.expander("📚 View Permanent Legal History", expanded=False):
        for msg in st.session_state.messages:
            st.markdown(f"**{msg['role'].upper()}:**\n{clean_legal_text(msg['content'])}")
            st.markdown("---")

# --- 7. CHAT ENGINE ---
if prompt := st.chat_input("Submit your draft for revision..."):
    user_emb = client.models.embed_content(model=EMBED_MODEL, contents=prompt)
    collection.insert([[user_emb.embeddings[0].values], [prompt], [USER_IDENTITY], ["user"]])
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.status("Analyzing Legal Logic...", expanded=True) as status:
            try:
                full_input = f"{LEGAL_PROMPT}\n\nUSER DRAFT: {prompt}"
                response = client.models.generate_content(
                    model=MODEL_ID,
                    contents=full_input,
                    config=types.GenerateContentConfig(thinking_config=types.ThinkingConfig(include_thoughts=True), temperature=0.0)
                )
                
                final_answer = ""
                for part in response.candidates[0].content.parts:
                    if part.thought:
                        with st.expander("🔍 GAP ANALYSIS", expanded=True):
                            st.info(clean_legal_text(part.text))
                    else:
                        final_answer += part.text

                if final_answer:
                    st.write("💾 Archiving to Zilliz Memory...")
                    ai_emb = client.models.embed_content(model=EMBED_MODEL, contents=final_answer[:59000])
                    collection.insert([[ai_emb.embeddings[0].values], [final_answer[:59000]], [USER_IDENTITY], ["assistant"]])
                    collection.flush()

                status.update(label="Complete", state="complete", expanded=False)
                st.subheader("Revised Legal Submission")
                st.markdown(clean_legal_text(final_answer))
                st.session_state.messages.append({"role": "assistant", "content": final_answer})
            except Exception as e:
                st.error(f"Error: {e}")
