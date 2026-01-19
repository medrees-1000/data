import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Fix path for imports
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from ingestion.pdf_parser import extract_text_from_pdf
from ingestion.chunking import chunk_text
from embeddings.embed_resumes import get_embedding_model
from matching.similarity import calculate_match_score

st.set_page_config(page_title="AI Resume-Job Matcher", layout="wide")

@st.cache_resource
def load_model():
    return get_embedding_model()

model = load_model()

st.title("AI Resume-Job Matching Platform")
st.markdown("Upload your resume to see how well you match with our open roles.")

# Data Loading
try:
    job_metadata = pd.read_csv(ROOT_DIR / "data" / "job_metadata.csv")
    job_embeddings = np.load(ROOT_DIR / "data" / "job_embeddings.npy")
    job_names = job_metadata["job_id"].tolist()
except FileNotFoundError:
    st.error("Missing job data. Please run 'embed_jobs.py' first.")
    st.stop()

# Sidebar
with st.sidebar:
    st.header("Settings")
    selected_job = st.selectbox("Select Target Job:", job_names)
    job_index = job_names.index(selected_job)
    target_job_embedding = job_embeddings[job_index]

# Main UI
uploaded_file = st.file_uploader("Upload your Resume (PDF)", type="pdf")

if uploaded_file is not None:
    if st.button("Analyze Match"):
        with st.spinner("Analyzing..."):
            raw_text = extract_text_from_pdf(uploaded_file)
            
            if raw_text:
                chunks = chunk_text(raw_text)
                chunk_embeddings = model.encode(chunks)
                
                from sklearn.metrics.pairwise import cosine_similarity
                similarities = cosine_similarity(chunk_embeddings, target_job_embedding.reshape(1, -1))
                final_score = float(np.max(similarities))
                
                st.success("Analysis Complete")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Match Score", f"{final_score:.2%}")
                    st.progress(final_score)
                with col2:
                    if final_score > 0.70:
                        st.write("Strong match for this role.")
                    else:
                        st.write("Consider tailoring your resume further.")

                best_chunk_idx = np.argmax(similarities)
                with st.expander("View Match Evidence"):
                    st.write(chunks[best_chunk_idx])
            else:
                st.error("Text extraction failed.")

st.divider()
if st.checkbox("Show Raw Data Preview (Project Requirement)"):
    try:
        raw_data = pd.read_csv(ROOT_DIR / "data" / "processed_resumes.csv")
        st.dataframe(raw_data.head())
    except FileNotFoundError:
        st.warning("processed_resumes.csv not found.")