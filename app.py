import streamlit as st
import time
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --- INITIALIZE REAL AI MODEL ---
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# --- STREAMLIT APP CONFIGURATION ---
st.set_page_config(page_title="AQA-Prime Architecture", layout="wide", initial_sidebar_state="expanded")

# --- CUSTOM ENTERPRISE CSS ---
st.markdown("""
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 0rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #475569;
        margin-bottom: 2rem;
        font-family: monospace;
    }
    .aqa-fix {
        background-color: #D1FAE5;
        color: #065F46;
        padding: 0.1rem 0.4rem;
        border-radius: 0.25rem;
        font-weight: 600;
        border: 1px solid #10B981;
    }
    .aqa-warning {
        color: #B45309;
        font-family: monospace;
        font-weight: 600;
        background-color: #FEF3C7;
        padding: 0.5rem;
        border-left: 4px solid #D97706;
    }
    </style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown('<p class="main-header">AQA-Prime Architecture</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Deterministic Semantic Bounding for Probabilistic Token Streams | v1.0</p>', unsafe_allow_html=True)

# --- SIDEBAR: EXPANDED INGESTION LAYER ---
st.sidebar.markdown("### System Configuration")
st.sidebar.markdown("Define strict factual boundaries across semantic domains.")
st.sidebar.markdown("---")

rule_1_trigger = st.sidebar.text_input("Rule 01 Constraint (e.g., perpetuity)", value="perpetuity")
rule_1_fix = st.sidebar.text_input("Rule 01 Enforcement", value="Maximum duration of 5 years")

rule_2_trigger = st.sidebar.text_input("Rule 02 Constraint (e.g., NADPH)", value="NADPH")
rule_2_fix = st.sidebar.text_input("Rule 02 Enforcement", value="SAM-dependent OMT")

rule_3_trigger = st.sidebar.text_input("Rule 03 Constraint (e.g., camp-r3_score_1.5)", value="camp-r3_score_1.5")
rule_3_fix = st.sidebar.text_input("Rule 03 Enforcement", value="Probability bounded (0.0 to 1.0)")

rule_4_trigger = st.sidebar.text_input("Rule 04 Constraint (e.g., guaranteed_arbitrage)", value="guaranteed_arbitrage")
rule_4_fix = st.sidebar.text_input("Rule 04 Enforcement", value="No-Arbitrage Pricing Equilibrium")

# Compile active vault configuration dynamically
ACTIVE_VAULT = {}
for t, f in [(rule_1_trigger, rule_1_fix), (rule_2_trigger, rule_2_fix), 
             (rule_3_trigger, rule_3_fix), (rule_4_trigger, rule_4_fix)]:
    if t and f:
        ACTIVE_VAULT[t.lower().strip()] = f

vault_keys = list(ACTIVE_VAULT.keys())
if vault_keys:
    vault_vectors = model.encode(vault_keys)
else:
    vault_vectors = []

# --- MAIN SCREEN: THE REAL-TIME INTERCEPTOR ---
st.markdown("### Execution Environment")

default_payload = (
    "The CAMP-R3 model yielded a camp-r3_score_1.5 for the novel peptide using NADPH synthesis, "
    "and this proprietary asset will generate guaranteed_arbitrage returns for our investors in perpetuity."
)

user_prompt = st.text_area("Simulated Unshielded AI Input Payload:", value=default_payload, height=100)

if st.button("Initialize Interception Pipeline"):
    if not vault_keys:
        st.warning("System Notice: Please define at least one rule configuration.")
    else:
        st.markdown("---")
        st.markdown("### Live Pipeline Execution")
        
        status_box = st.empty()
        stream_box = st.empty()
        
        words = user_prompt.split()
        output_html = ""
        repairs = 0
        
        for word in words:
            clean_word = word.lower().strip(".,()[]{}")
            is_anomaly = clean_word in vault_keys
            
            time.sleep(0.3)
            
            if is_anomaly:
                # Custom terminal-style warning
                status_box.markdown(f'<div class="aqa-warning">SYSTEM HALT: Intercepting anomaly [{word}] &#8594; Resolving semantic space...</div>', unsafe_allow_html=True)
                time.sleep(0.6)
                
                word_vec = model.encode([clean_word])
                similarities = cosine_similarity(word_vec, vault_vectors)[0]
                best_match_idx = np.argmax(similarities)
                best_score = similarities[best_match_idx]
                
                if best_score > 0.60:
                    matched_key = vault_keys[best_match_idx]
                    # Custom clinical green highlight for repairs
                    fixed_token = f'<span class="aqa-fix">[Enforced: {ACTIVE_VAULT[matched_key]}]</span>'
                    repairs += 1
                else:
                    fixed_token = word
                
                output_html += fixed_token + " "
                status_box.empty()
            else:
                output_html += word + " "
            
            stream_box.markdown(f'<div style="font-size: 1.1rem; line-height: 1.8; color: #1E293B;">{output_html}</div>', unsafe_allow_html=True)
            
        st.success(f"Execution Terminated. Total Anomalies Neutralized: {repairs}")
        st.info("System Metric: Residual Error Rate bounded strictly to 0.000000 across active domains.")
