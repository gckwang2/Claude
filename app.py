import streamlit as st
import json
import os
import re
from google import genai
from google.cloud import aiplatform  # Added for Dedicated Endpoint access
from pymilvus import connections, Collection, utility, FieldSchema, CollectionSchema, DataType

# --- 1. GLOBAL PROMPT (Elite Singapore Family Law Strategist) ---
LEGAL_PROMPT = """
ROLE:
You are an Elite Singapore Family Law Strategist specializing in High-Conflict Auxiliary Matters (AM). 

GOAL: 
Construct a Reply Affidavit narrative that systematically secures a 75:25 asset division by:
1. Proving 100% Direct Financial Contribution for key assets (e.g., 18 Simon Road).
2. Triggering a Robust Adverse Inference due to the Respondent’s documented non-disclosure (UK Accounts/Income).
3. Offsetting Matrimonial Liabilities (Unsecured Loans/Brother’s Loan) against the Gross Pool.

STRATEGIC FRAMEWORK (INTERNAL LOGIC):
- ANJ v ANK: Maximize Step 1 (Direct) by linking mortgages/downpayments to the User’s sole earnings.
- TQU v TQT: Use "Lack of Candor" as a mechanical multiplier for the final ratio.
- TNL v TNK: Defend against "Dissipation" by proving the "Money Trail" to regulated brokers (IBKR) and household maintenance (Status Quo).

OPERATIONAL PROTOCOLS:
- THE "CREDIBILITY TRAP": Whenever the Respondent makes a "Bare Allegation" (e.g., Cambodia property), demand the "Substratum of Evidence." Highlight that a "Factual Impossibility" (e.g., URA tracking overseas land) is evidence of perjury.
- THE "NET POOL" DEFENSE: Explicitly link specific debts to specific assets (e.g., Renovation Loan vs. Sutton Park Valuation) to "neutralize" low-value assets.
- NO LEGAL CITATIONS: Do not mention case names. Write with the "Voice of the Court"—firm, objective, and mathematically driven.
- BANNED PHRASES: Avoid "I feel" or "I think." Use "The objective evidence confirms..." or "The Respondent has failed to produce..."
"""

# --- 2. CONFIG & IDENTITY ---
PROJECT_ID = st.secrets["PROJECT_ID"]
LOCATION = "us-central1" 
# Your specific Endpoint ID from the console
ENDPOINT_ID = "b72257f8-3872-4ddd-8981-54688cf9c4a5" 
EMBED_MODEL = "text-embedding-004"
USER_IDENTITY = "Freddy_Legal_Project_2026"

# --- 3. LOGIN GATE ---
def check_password():
    if "passwords" not in st.secrets:
        st.error("🚨 Configuration Error: '[passwords]' section missing in Secrets.")
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

# --- 4. GCP AUTH FIX ---
if "gcp_service_account" in st.secrets:
    with open("gcp_key.json", "w") as f:
        json.dump(dict(st.secrets["gcp_service_account"]), f)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "gcp_key.json"
else:
    st.error("GCP Service Account credentials missing in secrets!")
    st.stop()

# Initialize Vertex AI for the Dedicated Endpoint
aiplatform.init(project=PROJECT_ID, location=LOCATION)
# Client for embeddings (still uses genai for text-embedding-004)
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
        results = collection.query(expr=f'session_id == "{session_id}"', output_fields=["id", "text", "role"])
        return sorted(results, key=lambda x: x['id'])
    except:
        return []

def delete_interaction(ids_to_delete, index_in_state):
    try:
        delete_expr = f"id in {ids_to_delete}"
        collection.delete(delete_expr)
        collection.flush()
        st.session_state.messages.pop(index_in_state)
        st.success("Interaction purged.")
        st.rerun()
    except Exception as e:
        st.error(f"Deletion failed: {e}")

# --- 6. RAG RETRIEVAL ENGINE ---
def retrieve_relevant_context(query_text, top_k=3):
    try:
        search_emb = gemini_client.models.embed_content(
            model=EMBED_MODEL, 
            contents=query_text
        ).embeddings[0].values
        
        results = collection.search(
            data=[search_emb], 
            anns_field="vector", 
            param={"metric_type": "L2", "params": {"nprobe": 10}}, 
            limit=top_k, 
            output_fields=["text"],
            expr=f'session_id == "{USER_IDENTITY}"'
        )
        
        context_snippets = [hit.entity.get("text") for hit in results[0]]
        return "\n\n---\n\n".join(context_snippets) if context_snippets else "No relevant past context found."
    except Exception as e:
        st.warning(f"Memory Retrieval failed: {e}")
        return ""

# --- 7. UI SETUP ---
st.set_page_config(page_title="Legal Strategist", layout="wide")
st.title("⚖️ Principal Legal Advisor (gpt-oss-120b)")

if "messages" not in st.session_state:
    raw_history = load_history(USER_IDENTITY)
    st.session_state.messages = []
    temp_pair = {}
    for item in raw_history:
        if item['role'] == 'user':
            temp_pair = {"user": item['text'], "u_id": item['id']}
        elif item['role'] == 'assistant' and "user" in temp_pair:
            st.session_state.messages.append({
                "user": temp_pair["user"], 
                "assistant": item['text'],
                "u_id": temp_pair["u_id"],
                "a_id": item['id']
            })
            temp_pair = {}

# --- 8. DISPLAY HISTORY ---
st.subheader("Consultation History")
for i, entry in enumerate(st.session_state.messages):
    with st.expander(f"📂 Interaction {i+1}: {entry['user'][:50]}...", expanded=False):
        st.markdown("**👤 Your Query:**")
        st.write(entry['user'])
        st.markdown("---")
        st.markdown("**⚖️ Advisor Strategy:**")
        st.markdown(clean_legal_text(entry['assistant']))
        if st.button(f"🗑️ Delete Interaction {i+1}", key=f"del_{i}"):
            delete_interaction([entry["u_id"], entry["a_id"]], i)

# --- 9. CHAT ENGINE (DEDICATED ENDPOINT) ---
if prompt := st.chat_input("Enter your reply affidavit draft..."):
    with st.chat_message("assistant"):
        with st.status("Accessing Legal Memory & Generating Strategy...", expanded=True) as status:
            try:
                # STEP 1: RETRIEVE
                past_context = retrieve_relevant_context(prompt)
                
                # STEP 2: AUGMENT
                full_prompt = f"{LEGAL_PROMPT}\n\n### CONTEXT:\n{past_context}\n\n### DRAFT:\n{prompt}\n\n### REVISION:"

                # STEP 3: GENERATE via Dedicated Endpoint
                endpoint = aiplatform.Endpoint(
                    endpoint_name=f"projects/{PROJECT_ID}/locations/{LOCATION}/endpoints/{ENDPOINT_ID}"
                )
                
                # Note: vLLM deployments usually expect this instances format
                instances = [{
                    "prompt": full_prompt,
                    "max_tokens": 2048,
                    "temperature": 0.0,
                    "top_p": 1.0,
                }]
                
                response = endpoint.predict(instances=instances)
                
                # vLLM responses are typically in response.predictions[0]
                if response.predictions:
                    final_answer = response.predictions[0]
                    # If vLLM returns the full prompt + completion, strip the prompt
                    if "### REVISION:" in final_answer:
                        final_answer = final_answer.split("### REVISION:")[-1].strip()
                else:
                    final_answer = "Error: No output from model."

                st.markdown(clean_legal_text(final_answer))

                # STEP 4: ARCHIVE
                st.write("💾 Archiving to Zilliz...")
                u_emb = gemini_client.models.embed_content(model=EMBED_MODEL, contents=prompt[:59000]).embeddings[0].values
                a_emb = gemini_client.models.embed_content(model=EMBED_MODEL, contents=final_answer[:59000]).embeddings[0].values
                
                res = collection.insert([
                    [u_emb, a_emb], 
                    [prompt[:59000], final_answer[:59000]], 
                    [USER_IDENTITY, USER_IDENTITY], 
                    ["user", "assistant"]
                ])
                collection.flush()
                
                status.update(label="Strategic Revision Complete", state="complete", expanded=False)
                st.rerun() 
                
            except Exception as e:
                st.error(f"Logic Engine Error: {e}")
