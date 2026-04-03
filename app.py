import streamlit as st
import json
import os
import re
from google import genai
from google.genai import types
from pymilvus import connections, Collection, utility, FieldSchema, CollectionSchema, DataType

# --- 1. FULL LEGAL PROMPT (Logic Flow Intact) ---
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
- THE "CREDIBILITY TRAP": Whenever the Respondent makes a "Bare Allegation" (e.g., Cambodia property), demand the "Substratum of Evidence." 
- NO LEGAL CITATIONS: Do not mention case names. Write with the "Voice of the Court"—firm, objective, and mathematically driven.
- BANNED PHRASES: Avoid "I feel" or "I think." Use "The objective evidence confirms..."
- USE respectful tone
"""

# --- 2. CONFIG & IDENTITY ---
PROJECT_ID = st.secrets["PROJECT_ID"]
LOCATION = "us-east5" 
# Updated to the current 3.1 Preview path
MODEL_ID = "gemini-3.1-flash-preview" 
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

client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)

# --- 5. ZILLIZ & UTILS (RAG PERSISTENCE) ---
@st.cache_resource
def init_zilliz():
    connections.connect(uri=st.secrets["ZILLIZ_URI"], token=st.secrets["ZILLIZ_TOKEN"])
    col = Collection("legal_memory_v2")
    col.load()
    return col

collection = init_zilliz()

def clean_legal_text(text):
    if not text: return ""
    return text.replace("add−back", "add-back").replace("S$", "S$ ").replace("\n", "\n\n")

# --- 6. RAG RETRIEVAL ENGINE ---
def retrieve_relevant_context(query_text):
    try:
        # Embedding the query for semantic search
        search_emb = client.models.embed_content(
            model=EMBED_MODEL, 
            contents=query_text
        ).embeddings[0].values
        
        # Searching the user-specific memory pool
        res = collection.search(
            data=[search_emb], 
            anns_field="vector", 
            param={"metric_type": "L2", "params": {"nprobe": 10}}, 
            limit=3, 
            output_fields=["text"],
            expr=f'session_id == "{USER_IDENTITY}"'
        )
        
        context_snippets = [hit.entity.get("text") for hit in res[0]]
        return "\n\n---\n\n".join(context_snippets) if context_snippets else "No past context found."
    except Exception as e:
        return ""

# --- 7. UI SETUP ---
st.set_page_config(page_title="Legal Strategist", layout="wide")
st.title("⚖️ Principal Legal Advisor (Gemini 3.1 Flash)")

# --- 8. CHAT ENGINE (Gemini 3.1 Flash RAG) ---
if prompt := st.chat_input("Enter your reply affidavit draft..."):
    with st.chat_message("assistant"):
        with st.status(f"Analyzing with {MODEL_ID}...", expanded=True) as status:
            try:
                # RAG STEP 1: RETRIEVE
                past_context = retrieve_relevant_context(prompt)
                
                # RAG STEP 2: AUGMENT
                full_input = f"{LEGAL_PROMPT}\n\n### CASE CONTEXT:\n{past_context}\n\n### CURRENT DRAFT:\n{prompt}"

                # RAG STEP 3: GENERATE (Temperature=0 for Freddy's precise logic)
                response = client.models.generate_content(
                    model=MODEL_ID,
                    contents=full_input,
                    config=types.GenerateContentConfig(temperature=0.0)
                )
                
                final_answer = response.text
                st.markdown(clean_legal_text(final_answer))

                # RAG STEP 4: ARCHIVE NEW INTERACTION
                if final_answer:
                    u_emb = client.models.embed_content(model=EMBED_MODEL, contents=prompt[:59000]).embeddings[0].values
                    a_emb = client.models.embed_content(model=EMBED_MODEL, contents=final_answer[:59000]).embeddings[0].values
                    
                    collection.insert([
                        [u_emb, a_emb], 
                        [prompt[:59000], final_answer[:59000]], 
                        [USER_IDENTITY, USER_IDENTITY], 
                        ["user", "assistant"]
                    ])
                    collection.flush()
                
                status.update(label="Strategic Revision Complete", state="complete")
                
            except Exception as e:
                st.error(f"Logic Engine Error: {e}")
