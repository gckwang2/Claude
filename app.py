import streamlit as st
import json
import os
import re
import uuid
from google import genai
from google.genai import types
from pymilvus import connections, Collection, utility, FieldSchema, CollectionSchema, DataType

# --- 1. GLOBAL PROMPT (Aggressive Reply Affidavit Protocol) ---
LEGAL_PROMPT = """
You are a Senior Legal Strategist specialized in Singapore Family Law. 
GOAL: Help the user achieve a 75:25 asset division ratio for Auxiliary Matters (AM).

PRECEDENTS:
- TQU v TQT [2020] SGCA 8: Use adverse inference logic for non-disclosure to secure an 'uplift' (75:25 target).
- ANJ v ANK: 3-step structured approach (Direct vs. Indirect contributions).

REVISION PROTOCOL:
- Analyze user input for "lapses" (e.g., missing bank statements, vague "family expense" claims, or contradiction or untraced funds).
- IMPORTANT: This is a reply affidavit. DO NOT quote any case names or references (like TQU v TQT) in the revised text. Simply infer the legal principles as being within the court's knowledge.
- TONE: Use a respectful, firm, and infer tone that is intellectually difficult to refute. Focus on the Respondent's failure to discharge their evidential burden.
"""

# --- 2. CONFIG & IDENTITY ---
PROJECT_ID = st.secrets["PROJECT_ID"]
LOCATION = "global" 
MODEL_ID = "gemini-3.1-pro-preview"
EMBED_MODEL = "text-embedding-004"
USER_IDENTITY = "Freddy_Legal_Project_2026"

# --- 3. LOGIN GATE ---
def check_password():
    if "passwords" not in st.secrets:
        st.error("🚨 Configuration Error: '[passwords]' section missing in Secrets.")
        return False
    def password_entered():
        if st.session_state["username"] in st.secrets["passwords"] and st.session_state["password"] == st.secrets["passwords"][st.session_state["username"]]:
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

# --- 4. ZILLIZ & UTILS ---
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
        schema = CollectionSchema(fields)
        col = Collection(col_name, schema)
        col.create_index("vector", {"metric_type": "L2", "index_type": "IVF_FLAT", "params": {"nlist": 128}})
    else:
        col = Collection(col_name)
    col.load()
    return col

collection = init_zilliz()

def clean_legal_text(text):
    if not text: return ""
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    return text.replace("add−back", "add-back").replace("S$", "S$ ").replace("\n", "\n\n")

def load_history(session_id):
    try:
        # We fetch the 'id' field now so we can delete specific records later
        results = collection.query(expr=f'session_id == "{session_id}"', output_fields=["id", "text", "role"])
        return sorted(results, key=lambda x: x['id'])
    except:
        return []

def delete_interaction(ids_to_delete, index_in_state):
    """Deletes records from Zilliz and removes from the local session state."""
    try:
        # Construct deletion expression based on primary keys
        delete_expr = f"id in {ids_to_delete}"
        collection.delete(delete_expr)
        collection.flush()
        # Remove from local UI list
        st.session_state.messages.pop(index_in_state)
        st.success("Interaction deleted successfully.")
        st.rerun()
    except Exception as e:
        st.error(f"Deletion failed: {e}")

# --- 5. UI SETUP ---
st.set_page_config(page_title="Legal Strategist", layout="wide")
st.title("⚖️ Principal Legal Advisor")

if "messages" not in st.session_state:
    raw_history = load_history(USER_IDENTITY)
    st.session_state.messages = []
    temp_pair = {}
    for item in raw_history:
        if item['role'] == 'user':
            temp_pair = {"user": item['text'], "u_id": item['id']}
        elif item['role'] == 'assistant' and "user" in temp_pair:
            temp_pair["assistant"] = item['text']
            temp_pair["a_id"] = item['id']
            st.session_state.messages.append(temp_pair)
            temp_pair = {}

# --- 6. DISPLAY HISTORY ---
st.subheader("Consultation History")
for i, entry in enumerate(st.session_state.messages):
    with st.expander(f"📂 Interaction {i+1}: {entry['user'][:50]}...", expanded=False):
        st.markdown("**👤 Your Query:**")
        st.write(entry['user'])
        st.markdown("---")
        st.markdown("**⚖️ Advisor Strategy:**")
        st.markdown(clean_legal_text(entry['assistant']))
        
        # Delete Button
        if st.button(f"🗑️ Delete Interaction {i+1}", key=f"del_{i}"):
            # We pass the list of IDs (User record ID and Assistant record ID)
            ids = [entry["u_id"], entry["a_id"]]
            delete_interaction(ids, i)

# --- 7. CHAT ENGINE ---
client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)

if prompt := st.chat_input("Enter your reply affidavit draft..."):
    with st.chat_message("assistant"):
        with st.status("Synthesizing Adversarial Logic...", expanded=True) as status:
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
                        with st.expander("🔍 INTERNAL GAP ANALYSIS", expanded=True):
                            st.info(clean_legal_text(part.text))
                    else:
                        final_answer += part.text

                # --- 8. SMART ARCHIVING ---
                if final_answer:
                    st.write("💾 Archiving to Zilliz...")
                    safe_final = final_answer[:59000]
                    safe_prompt = prompt[:59000]
                    
                    u_emb = client.models.embed_content(model=EMBED_MODEL, contents=safe_prompt).embeddings[0].values
                    a_emb = client.models.embed_content(model=EMBED_MODEL, contents=safe_final).embeddings[0].values
                    
                    # Inserting and getting the new IDs
                    res = collection.insert([
                        [u_emb, a_emb], 
                        [safe_prompt, safe_final], 
                        [USER_IDENTITY, USER_IDENTITY], 
                        ["user", "assistant"]
                    ])
                    collection.flush()
                    
                    # Add new interaction to session state so it can be deleted immediately if needed
                    new_ids = res.primary_keys
                    st.session_state.messages.append({
                        "user": prompt, 
                        "assistant": final_answer,
                        "u_id": new_ids[0],
                        "a_id": new_ids[1]
                    })
                    
                status.update(label="Analysis Complete", state="complete", expanded=False)
                st.rerun() # Refresh to show the new interaction in the history list
                
            except Exception as e:
                st.error(f"Logic Engine Error: {e}")
