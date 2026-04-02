import streamlit as st
import json
import os
import uuid
import vertexai
from anthropic import AnthropicVertex
from pymilvus import connections, Collection, utility, FieldSchema, CollectionSchema, DataType
from vertexai.language_models import TextEmbeddingModel

# --- 1. LOGIN GATE ---
def check_password():
    def password_entered():
        if (st.session_state["username"] == st.secrets["credentials"]["admin_user"] and 
            st.session_state["password"] == st.secrets["credentials"]["admin_password"]):
            st.session_state["password_correct"] = True
            del st.session_state["password"]
            del st.session_state["username"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.title("⚖️ SG Legal Advisor Login")
        st.text_input("Username", key="username")
        st.text_input("Password", type="password", key="password")
        st.button("Login", on_click=password_entered)
        return False
    return st.session_state["password_correct"]

if not check_password():
    st.stop()

# --- 2. INITIALIZE GCP & VERTEX ---
if "gcp_service_account" in st.secrets:
    # Construct the JSON structure from secrets for Vertex AI SDK
    gcp_info = dict(st.secrets["gcp_service_account"])
    with open("gcp_key.json", "w") as f:
        json.dump(gcp_info, f)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "gcp_key.json"

PROJECT_ID = st.secrets["PROJECT_ID"]
REGION = st.secrets.get("REGION", "global")

# Initialize Vertex AI for Embeddings
vertexai.init(project=PROJECT_ID, location="us-central1") # Embeddings usually need a specific region
embed_model = TextEmbeddingModel.from_pretrained("text-embedding-004")
client = AnthropicVertex(project_id=PROJECT_ID, region=REGION)

def get_embedding(text):
    embeddings = embed_model.get_embeddings([text])
    return embeddings[0].values

# --- 3. ZILLIZ INITIALIZATION (1-YEAR TTL) ---
def init_zilliz():
    connections.connect(alias="default", uri=st.secrets["ZILLIZ_URI"], token=st.secrets["ZILLIZ_TOKEN"])
    col_name = "legal_chat_memory"
    if not utility.has_collection(col_name):
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=768), # dim 768 for text-embedding-004
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=5000),
            FieldSchema(name="session_id", dtype=DataType.VARCHAR, max_length=100)
        ]
        schema = CollectionSchema(fields, "Legal Strategy Context")
        # 31,536,000 seconds = 1 YEAR
        collection = Collection(name=col_name, schema=schema, properties={"collection.ttl.seconds": 31536000})
        collection.create_index(field_name="vector", index_params={"metric_type": "L2", "index_type": "IVF_FLAT", "params": {"nlist": 128}})
    else:
        collection = Collection(col_name)
        collection.set_properties(properties={"collection.ttl.seconds": 31536000})
        collection.load()
    return collection

collection = init_zilliz()

# --- 4. LEGAL STRATEGY PROMPT ---
LEGAL_PROMPT = """
You are a Senior Legal Advisor specialized in Singapore Family Law. 
GOAL: Help the user achieve a 75:25 asset division ratio for Auxiliary Matters (AM).

PRECEDENTS:
- TQU v TQT [2020] SGCA 8: Use adverse inference logic for non-disclosure to secure an 'uplift' (75:25 target).
- ANJ v ANK: 3-step structured approach (Direct vs. Indirect contributions).

EVIDENCE PROTOCOL:
- If a user makes a claim, ask for PROOF (bank statements, CPF, receipts).
- If proof is missing, explain how to INFER the asset existence via TQU v TQT.
- Warning: State clearly that claims without proof or logical inference will fail in court.
"""

# --- 5. UI & CHAT LOOP (RAG) ---
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = str(uuid.uuid4())[:8]
if "history" not in st.session_state:
    st.session_state.history = {}

with st.sidebar:
    st.title("⚖️ Case Management")
    if st.button("➕ New Strategy Session"):
        st.session_state.current_chat_id = str(uuid.uuid4())[:8]
        st.rerun()
    st.divider()
    st.info(f"User: {st.secrets['credentials']['admin_user']}\nSession: {st.session_state.current_chat_id}")

st.title("SG Legal Advisor: AM Strategy")

current_id = st.session_state.current_chat_id
if current_id not in st.session_state.history:
    st.session_state.history[current_id] = []

for msg in st.session_state.history[current_id]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("State a contribution or claim..."):
    st.session_state.history[current_id].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            # A. SEARCH PAST CONTEXT
            query_vec = get_embedding(prompt)
            search_results = collection.search(
                data=[query_vec], anns_field="vector", param={"metric_type": "L2"}, 
                limit=2, output_fields=["text"], expr=f"session_id == '{current_id}'"
            )
            
            context_str = ""
            for hit in search_results[0]:
                context_str += f"\n[Past context: {hit.entity.get('text')}]"

            # B. SYNTHESIZE WITH CLAUDE
            enriched_input = f"{context_str}\n\nUSER CLAIM: {prompt}"
            response = client.messages.create(
                model="claude-4-1-sonnet@20251022",
                max_tokens=4096,
                system=LEGAL_PROMPT,
                messages=[{"role": "user", "content": enriched_input}]
            )
            
            answer = response.content[0].text
            st.markdown(answer)
            st.session_state.history[current_id].append({"role": "assistant", "content": answer})

            # C. STORE FOR FUTURE
            full_record = f"Q: {prompt} | A: {answer}"
            record_vec = get_embedding(full_record)
            collection.insert([[record_vec], [full_record[:5000]], [current_id]])
            
        except Exception as e:
            st.error(f"Error: {e}")
