import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Setup paths
CHUNK_FILE = Path("data/resume_chunks.csv")
VEC_OUTPUT = Path("data/resume_embeddings.npy")
MAPPING_OUTPUT = Path("data/resume_chunks_with_ids.csv")
Path("embeddings").mkdir(exist_ok=True)

def main():
    if not CHUNK_FILE.exists():
        print(f"Error: {CHUNK_FILE} not found. Run 'chunk_resumes.py' first.")
        return

    # 1. Load the Chunks
    df = pd.read_csv(CHUNK_FILE)
    
    # 2. Load the Transformer Model
    print("Loading 'all-MiniLM-L6-v2' (this may take a moment on first run)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # 3. Encode the text into vectors
    print(f"Generating embeddings for {len(df)} chunks...")
    embeddings = model.encode(
        df["chunk_text"].tolist(),
        show_progress_bar=True
    )

    # 4. Save results
    np.save(VEC_OUTPUT, embeddings)
    df.to_csv(MAPPING_OUTPUT, index=False)

    print("\n" + "="*40)
    print(f"Embeddings Shape: {embeddings.shape}")
    print(f"Saved Vectors: {VEC_OUTPUT}")
    print(f"Saved Mapping: {MAPPING_OUTPUT}")
    print("="*40)

if __name__ == "__main__":
    main()