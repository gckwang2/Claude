import streamlit as st
import json
import os
import uuid
import vertexai
from anthropic import AnthropicVertex
from pymilvus import connections, Collection, utility, FieldSchema, CollectionSchema, DataType
from vertexai.language_models import TextEmbeddingModel

# --- 1. AUTHENTICATION GATE ---
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
        st.title("🔒 SG Legal Strategist Login")
        st.text_input("Username", key="username")
        st.text_input("Password", type="password", key="password")
        st.button("Login", on_click=password_entered)
        return False
    return st.session_state["password_correct"]

if not check_password():
    st.stop()

# --- 2. INITIALIZE GCP & VERTEX AI ---
if "gcp_service_account" in st.secrets:
    gcp_info = dict(st.secrets["gcp_service_account"])
    with open("gcp_key.json", "w") as f:
        json.dump(gcp_info, f)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "gcp_key.json"

PROJECT_ID = st.secrets["PROJECT_ID"]
REGION = st.secrets.get("REGION", "us-east5")

# Initialize Vertex for Embeddings and Claude
vertexai.init(project=PROJECT_ID, location=REGION)
embed_model = TextEmbeddingModel.from_pretrained("text-embedding-004")
client = AnthropicVertex(project_id=PROJECT_ID, region=REGION)

def get_embedding(text):
    embeddings = embed_model.get_embeddings([text])
    return embeddings[0].values

# --- 3. ZILLIZ INITIALIZATION (1-YEAR TTL) ---
def init_zilliz():
    connections.connect(alias="default", uri=st.secrets["ZILLIZ_URI"], token=st.secrets["ZILLIZ_TOKEN"])
    col_name = "legal_strategy_memory"
    
    if not utility.has_collection(col_name):
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=768), 
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=5000),
            FieldSchema(name="session_id", dtype=DataType.VARCHAR, max_length=100)
        ]
        schema = CollectionSchema(fields, "Legal Strategy Context")
        # 31,536,000 seconds = 1 YEAR TTL
        collection = Collection(name=col_name, schema=schema, properties={"collection.ttl.seconds": 31536000})
        collection.create_index(field_name="vector", index_params={"metric_type": "L2", "index_type": "IVF_FLAT", "params": {"nlist": 128}})
    else:
        collection = Collection(col_name)
        collection.set_properties(properties={"collection.ttl.seconds": 31536000})
        collection.load()
    return collection

collection = init_zilliz()

# --- 4. THE LEGAL STRATEGY PROMPT ---
LEGAL_PROMPT = """
You are a Senior Legal Advisor specialized in Singapore Family Law and AM hearings.
Your objective is to help the user achieve a 75:25 asset division ratio.

PRECEDENTS & STRATEGY:
- TQU v TQT [2020] SGCA 8: Use the 'Adverse Inference' rule if there is non-disclosure of assets. 
- ANJ v ANK: Use the 3-step structured approach (Direct Financial vs Indirect Contributions).

STRICT LITIGATION RULES:
1. WARNING ON PROOF: Every time the user makes a claim, you MUST warn them that they need documentary PROOF (bank statements, CPF, receipts).
2. PROOF VS. INFERENCE: If the user cannot prove a claim, instruct them on how to INFER it by identifying 'gaps' in the other party's disclosure to invite an adverse inference under TQU v TQT.
"""

# --- 5. UI & RAG LOOP ---
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
    st.info(f"User: {st.secrets['credentials']['admin_user']}\nSession ID: {st.session_state.current_chat_id}")

st.title("SG Legal Strategist: AM Division")
st.markdown("### Strategy: 75:25 Ratio (TQU v TQT)")

current_id = st.session_state.current_chat_id
if current_id not in st.session_state.history:
    st.session_state.history[current_id] = []

# Display Chat History
for msg in st.session_state.history[current_id]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Main Interaction
if prompt := st.chat_input("Describe a claim or missing asset..."):
    st.session_state.history[current_id].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            # A. RETRIEVAL (Search Zilliz for this session only)
            query_vec = get_embedding(prompt)
            search_results = collection.search(
                data=[query_vec], anns_field="vector", param={"metric_type": "L2"}, 
                limit=2, output_fields=["text"], expr=f"session_id == '{current_id}'"
            )
            
            context_str = ""
            for hit in search_results[0]:
                context_str += f"\n[Previous Record: {hit.entity.get('text')}]"

            # B. SYNTHESIS (Claude Opus 4.1 Flagship)
            enriched_input = f"{context_str}\n\nUSER NEW CLAIM: {prompt}"
            response = client.messages.create(
                model="anthropic-claude-opus-4-1", # Updated April 2026 Stable ID
                max_tokens=4096,
                system=LEGAL_PROMPT,
                messages=[{"role": "user", "content": enriched_input}]
            )
            
            answer = response.content[0].text
            st.markdown(answer)
            st.session_state.history[current_id].append({"role": "assistant", "content": answer})

            # C. STORAGE
            combined_record = f"Q: {prompt} | A: {answer}"
            record_vec = get_embedding(combined_record)
            collection.insert([[record_vec], [combined_record[:5000]], [current_id]])
            
        except Exception as e:
            st.error(f"Engine Error: {e}")
