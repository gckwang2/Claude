import streamlit as st
import json
import os
import re
import uuid
from google import genai
from google.genai import types
from pymilvus import connections, Collection, utility, FieldSchema, CollectionSchema, DataType

# --- 1. CONFIGURATION ---
PROJECT_ID = st.secrets["PROJECT_ID"]
LOCATION = "global" 
MODEL_ID = "gemini-3.1-pro-preview"
EMBED_MODEL = "text-embedding-004" 

# --- 2. LEGAL STRATEGY PROMPT ---
LEGAL_PROMPT = """
You are a Senior Legal Advisor specialized in Singapore Family Law. 
GOAL: Help the user achieve a 75:25 asset division ratio for Auxiliary Matters (AM).

PRECEDENTS:
- TQU v TQT [2020] SGCA 8: Use adverse inference logic for non-disclosure to secure an 'uplift' (75:25 target).
- ANJ v ANK: 3-step structured approach (Direct vs. Indirect contributions).

REVISION PROTOCOL:
- Analyze user input for "lapses" (e.g., missing bank statements, vague "family expense" claims, or untraced funds like S$160k).
- Provide a "REVISED RESPONSE" for court submission using precise legal language to close all evidential gaps.
"""

# --- 3. AUTH & CLIENT ---
if "gcp_service_account" in st.secrets:
    with open("gcp_key.json", "w") as f:
        json.dump(dict(st.secrets["gcp_service_account"]), f)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "gcp_key.json"

client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)

# --- 4. ZILLIZ SETUP ---
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
        col = Collection(col_name, CollectionSchema(fields))
        col.create_index("vector", {"metric_type": "L2", "index_type": "IVF_FLAT", "params": {"nlist": 128}})
    else:
        col = Collection(col_name)
    col.load()
    return col

collection = init_zilliz()

# --- 5. UTILITIES ---
def clean_legal_text(text):
    if not text: return ""
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = text.replace("add−back", "add-back").replace("S$", "S$ ")
    return text.replace("\n", "\n\n")

def retrieve_context(query, collection, top_k=2):
    try:
        emb = client.models.embed_content(model=EMBED_MODEL, contents=query)
        query_vector = emb.embeddings[0].values
        results = collection.search(data=[query_vector], anns_field="vector", param={"metric_type": "L2"}, limit=top_k, output_fields=["text"])
        return "\n---\n".join([hit.entity.get('text') for hit in results[0]])
    except: return ""

# --- 6. UI SETUP ---
st.set_page_config(page_title="Legal Advisor", layout="wide")
st.title("⚖️ Principal Legal Advisor")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_id" not in st.session_state:
    st.session_state.chat_id = str(uuid.uuid4())[:8]

# --- COLLAPSIBLE HISTORY LOGIC ---
if st.session_state.messages:
    with st.expander("📚 View Previous Legal Consultations", expanded=False):
        for i, msg in enumerate(st.session_state.messages):
            role_label = "👤 User Query" if msg["role"] == "user" else "⚖️ Advisor Response"
            st.markdown(f"**{role_label}:**")
            st.markdown(clean_legal_text(msg["content"]))
            st.markdown("---")

# --- 7. CHAT & RAG ENGINE ---
if prompt := st.chat_input("Submit your claim or draft for revision..."):
    # Display the current query immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.status("Initializing Legal Synthesis...", expanded=True) as status:
            try:
                st.write("🔍 Searching Zilliz memory...")
                context = retrieve_context(prompt, collection)
                
                st.write("🧠 Engaging Gemini 3.1 Pro 'Deep Think'...")
                full_input = f"{LEGAL_PROMPT}\n\nPREVIOUS CONTEXT:\n{context}\n\nUSER DRAFT: {prompt}"
                
                response = client.models.generate_content(
                    model=MODEL_ID,
                    contents=full_input,
                    config=types.GenerateContentConfig(thinking_config=types.ThinkingConfig(include_thoughts=True), temperature=0.0)
                )

                final_answer = ""
                for part in response.candidates[0].content.parts:
                    if part.thought:
                        with st.expander("🔍 INTERNAL STRATEGIC REASONING", expanded=True):
                            st.info(clean_legal_text(part.text))
                    else:
                        final_answer += part.text

                if final_answer:
                    st.write("💾 Archiving to Zilliz...")
                    emb = client.models.embed_content(model=EMBED_MODEL, contents=final_answer)
                    collection.insert([[emb.embeddings[0].values], [final_answer], [st.session_state.chat_id]])
                    collection.flush()

                status.update(label="Synthesis Complete", state="complete", expanded=False)
                
                # Render the final output clearly
                st.subheader("Revised Legal Submission")
                st.markdown("---")
                st.markdown(clean_legal_text(final_answer))
                st.session_state.messages.append({"role": "assistant", "content": final_answer})

            except Exception as e:
                status.update(label="Process Failed", state="error")
                st.error(f"Logic Engine Error: {e}")
