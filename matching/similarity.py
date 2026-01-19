import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

# Paths
RESUME_VEC = Path("data/resume_embeddings.npy")
JOB_VEC = Path("data/job_embeddings.npy")
RESUME_CSV = Path("data/resume_chunks_with_ids.csv")
JOB_CSV = Path("data/job_metadata.csv")

def rank_resumes(job_index=0, top_k=5):
    """
    Ranks resume chunks against a specific job description.
    """
    # 1. Load data
    resume_embeddings = np.load(RESUME_VEC)
    job_embeddings = np.load(JOB_VEC)
    resume_chunks = pd.read_csv(RESUME_CSV)
    job_metadata = pd.read_csv(JOB_CSV)

    # 2. Prepare vectors
    job_vector = job_embeddings[job_index].reshape(1, -1)
    
    # 3. Calculate Cosine Similarity
    # result is a list of scores for every resume chunk
    similarities = cosine_similarity(job_vector, resume_embeddings)[0]

    # 4. Attach scores and sort
    resume_chunks["score"] = similarities
    top_chunks = resume_chunks.sort_values("score", ascending=False).head(top_k)

    # Print the Job being matched
    job_name = job_metadata.iloc[job_index]["job_id"]
    print(f"--- Top {top_k} Matches for Job: {job_name} ---")
    
    return top_chunks

if __name__ == "__main__":
    # Ensure the matching folder exists
    Path("matching").mkdir(exist_ok=True)
    
    try:
        results = rank_resumes(job_index=0, top_k=5)
        print(results[["resume_id", "score"]])
    except FileNotFoundError:
        print("Error: Embedding files not found. Run embed_jobs.py first.")