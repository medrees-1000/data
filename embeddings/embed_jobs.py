import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Setup paths
JOB_DIR = Path("data/jobs")
VEC_OUTPUT = Path("data/job_embeddings.npy")
META_OUTPUT = Path("data/job_metadata.csv")

def main():
    if not JOB_DIR.exists() or not list(JOB_DIR.glob("*.txt")):
        print("Error: data/jobs/ folder is empty or does not exist.")
        return

    # 1. Load Model
    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    records = []
    
    # 2. Process each job file
    print("Embedding job descriptions...")
    for job_file in JOB_DIR.glob("*.txt"):
        text = job_file.read_text(encoding="utf-8")
        embedding = model.encode(text)

        records.append({
            "job_id": job_file.stem,
            "job_text": text,
            "embedding": embedding
        })

    # 3. Save results
    df = pd.DataFrame(records)
    # Save the raw vectors
    np.save(VEC_OUTPUT, df["embedding"].tolist())
    # Save the text metadata separately
    df.drop(columns=["embedding"]).to_csv(META_OUTPUT, index=False)

    print("Success: Job descriptions embedded.")
    print(f"Saved to {VEC_OUTPUT} and {META_OUTPUT}")

if __name__ == "__main__":
    main()