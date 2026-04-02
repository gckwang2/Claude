import streamlit as st
import json
import os
import uuid
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig, ThinkingConfig
from pymilvus import connections, Collection, utility, FieldSchema, CollectionSchema, DataType
from vertexai.language_models import TextEmbeddingModel

# --- 1. AUTH & GCP SETUP ---
# ... (Keep your existing login check logic here) ...

if "gcp_service_account" in st.secrets:
    gcp_info = dict(st.secrets["gcp_service_account"])
    with open("gcp_key.json", "w") as f:
        json.dump(gcp_info, f)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "gcp_key.json"

PROJECT_ID = st.secrets["PROJECT_ID"]
REGION = st.secrets.get("REGION", "us-east5")

vertexai.init(project=PROJECT_ID, location=REGION)

# Initialize Models
# Note: Use 'gemini-3.1-pro-preview' for the latest 2026 reasoning features
model = GenerativeModel("gemini-3.1-pro-preview")
embed_model = TextEmbeddingModel.from_pretrained("text-embedding-004")

# --- 2. ZILLIZ SETUP (1-YEAR TTL) ---
# ... (Keep your existing init_zilliz function here) ...
collection = init_zilliz()

# --- 3. LEGAL STRATEGY PROMPT ---
LEGAL_PROMPT = """
You are a Senior Legal Advisor (Singapore Family Law).
Goal: 75:25 asset division via TQU v TQT [2020] SGCA 8.
Strategy: Identify gaps in disclosure to trigger Adverse Inference.
Rule: If a claim isn't proven, you must show how to infer it.
"""

# --- 4. CHAT & REASONING LOGIC ---
if prompt := st.chat_input("Detail your asset claim..."):
    st.session_state.history[current_id].append({"role": "user", "content": prompt})
    
    with st.chat_message("assistant"):
        try:
            # A. RETRIEVAL
            query_vec = embed_model.get_embeddings([prompt])[0].values
            results = collection.search(
                data=[query_vec], anns_field="vector", param={"metric_type": "L2"}, 
                limit=3, output_fields=["text"], expr=f"session_id == '{current_id}'"
            )
            context = "\n".join([hit.entity.get('text') for hit in results[0]])

            # B. DEEP THINKING GENERATION
            # thinking_level="high" activates the 77% logic score mode
            config = GenerationConfig(
                thinking_config=ThinkingConfig(thinking_level="high"),
                temperature=0.0 # Grounded, neutral tone as per your instructions
            )
            
            full_input = f"{LEGAL_PROMPT}\n\nPAST CONTEXT:\n{context}\n\nNEW CLAIM: {prompt}"
            
            response = model.generate_content(full_input, generation_config=config)
            answer = response.text
            
            st.markdown(answer)
            st.session_state.history[current_id].append({"role": "assistant", "content": answer})

            # C. STORAGE
            # ... (Keep your existing collection.insert logic here) ...

        except Exception as e:
            st.error(f"Logic Engine Error: {e}")
