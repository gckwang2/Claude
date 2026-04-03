import streamlit as st
import json
import os
import re
from anthropic import AnthropicVertex # REQUIRED: pip install anthropic[vertex]
from google import genai
from google.genai import types
from pymilvus import connections, Collection, utility, FieldSchema, CollectionSchema, DataType

# --- 1. GLOBAL PROMPT ---
LEGAL_PROMPT = """
ROLE: Elite Singapore Family Law Strategist.
GOAL: Construct a Reply Affidavit narrative for 75:25 asset division.
- ANJ v ANK: Direct contributions.
- TQU v TQT: Adverse inference for non-disclosure.
- TNL v TNK: Defend against dissipation.
NO LEGAL CITATIONS. Firm, objective, forensic tone.
"""

# --- 2. CONFIG & IDENTITY ---
PROJECT_ID = st.secrets["PROJECT_ID"]
LOCATION = "us-east5" 
MODEL_ID = "claude-sonnet-4@20250514" 
EMBED_MODEL = "text-embedding-004"
USER_IDENTITY = "Freddy_Legal_Project_2026"

# --- 3. LOGIN GATE ---
def check_password():
    if "passwords" not in st.secrets:
        st.error("🚨 Configuration Error: '[passwords]' missing.")
        return False
    def password_entered():
        if (st.session_state["username"] in st.secrets["passwords"] and 
            st.session_state["password"] == st.secrets["passwords"][st.session_state["username"]]):
            st.session_state["password_correct"] = True
            del st.session_state["password"]
            del st.session_state["username"]
        else:
            st.session_state["password_correct"] = False
    if "password_correct" not in st.session_state:
        st.text_input("Username", key="username")
        st.text_input("Password", type="password", key="password")
        st.button("Log In", on_click=password_entered)
        return False
    return st.session_state["password_correct"]

if not check_password():
    st.stop()

# --- 4. GCP AUTH & CLIENTS ---
if "gcp_service_account" in st.secrets:
    with open("gcp_key.json", "w") as f:
        json.dump(dict(st.secrets["gcp_service_account"]), f)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "gcp_key.json"

# Anthropic Client for the Brain
anthropic_client = AnthropicVertex(region=LOCATION, project_id=PROJECT_ID)
# Gemini Client for the Embeddings (Zilliz search)
gemini_client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)

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
    return text.replace("add−back", "add-back").replace("S$", "S$ ").replace("\n", "\n\n")

def load_history(session_id):
    try:
        res = collection.query(expr=f'session_id == "{session_id}"', output_fields=["id", "text", "role"])
        return sorted(res, key=lambda x: x['id'])
    except: return []

def delete_interaction(ids, idx):
    collection.delete(f"id in {ids}")
    collection.flush()
    st.session_state.messages.pop(idx)
    st.rerun()

# --- 6. RAG RETRIEVAL ENGINE ---
def retrieve_relevant_context(query_text):
    try:
        # Embed with Gemini
        search_emb = gemini_client.models.embed_content(
            model=EMBED_MODEL, contents=query_text
        ).embeddings[0].values
        
        # Search Zilliz
        res = collection.search(
            data=[search_emb], anns_field="vector", 
            param={"metric_type": "L2", "params": {"nprobe": 10}}, 
            limit=3, output_fields=["text"],
            expr=f'session_id == "{USER_IDENTITY}"'
        )
        return "\n\n---\n\n".join([hit.entity.get("text") for hit in res[0]])
    except: return "No relevant past context found."

# --- 7. UI SETUP ---
st.set_page_config(page_title="Legal Strategist", layout="wide")
st.title("⚖️ Principal Legal Advisor (Claude 4 Sonnet)")

if "messages" not in st.session_state:
    raw = load_history(USER_IDENTITY)
    st.session_state.messages = []
    temp = {}
    for item in raw:
        if item['role'] == 'user': temp = {"user": item['text'], "u_id": item['id']}
        elif item['role'] == 'assistant' and "user" in temp:
            st.session_state.messages.append({**temp, "assistant": item['text'], "a_id": item['id']})

# --- 8. DISPLAY HISTORY ---
for i, entry in enumerate(st.session_state.messages):
    with st.expander(f"📂 Interaction {i+1}: {entry['user'][:50]}..."):
        st.write(entry['user'])
        st.markdown("---")
        st.markdown(clean_legal_text(entry['assistant']))
        if st.button(f"🗑️ Delete {i+1}", key=f"del_{i}"):
            delete_interaction([entry["u_id"], entry["a_id"]], i)

# --- 9. CHAT ENGINE (Claude 4 Version) ---
if prompt := st.chat_input("Enter reply affidavit draft..."):
    with st.chat_message("assistant"):
        with st.status(f"Strategic Analysis via {MODEL_ID}...", expanded=True) as status:
            try:
                past_context = retrieve_relevant_context(prompt)
                
                # Claude 4 specific Message API call
                response = anthropic_client.messages.create(
                    model=MODEL_ID,
                    max_tokens=4096,
                    temperature=0.0,
                    system=LEGAL_PROMPT,
                    messages=[
                        {"role": "user", "content": f"PAST CONTEXT:\n{past_context}\n\nUSER DRAFT:\n{prompt}"}
                    ]
                )
                
                final_answer = response.content[0].text
                st.markdown(clean_legal_text(final_answer))

                # ARCHIVE
                u_emb = gemini_client.models.embed_content(model=EMBED_MODEL, contents=prompt[:59000]).embeddings[0].values
                a_emb = gemini_client.models.embed_content(model=EMBED_MODEL, contents=final_answer[:59000]).embeddings[0].values
                
                res = collection.insert([
                    [u_emb, a_emb], [prompt[:59000], final_answer[:59000]], 
                    [USER_IDENTITY, USER_IDENTITY], ["user", "assistant"]
                ])
                collection.flush()
                
                # Update UI state immediately
                p_keys = res.primary_keys
                st.session_state.messages.append({
                    "user": prompt, "assistant": final_answer,
                    "u_id": p_keys[0], "a_id": p_keys[1]
                })
                
                status.update(label="Analysis Complete", state="complete")
                st.rerun() 
                
            except Exception as e:
                st.error(f"Logic Engine Error: {e}")
