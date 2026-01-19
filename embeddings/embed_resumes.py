import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

def get_embedding_model():
    """Loads and returns the transformer model."""
    return SentenceTransformer("all-MiniLM-L6-v2")

def embed_text(model, text):
    """Generates an embedding vector for the given text (string or list)."""
    return model.encode(text)

def main():
    CHUNK_FILE = Path("data/resume_chunks_with_ids.csv")
    VEC_OUTPUT = Path("data/resume_embeddings.npy")
    
    if not CHUNK_FILE.exists():
        print(f"Error: {CHUNK_FILE} not found.")
        return

    df = pd.read_csv(CHUNK_FILE)
    model = get_embedding_model()
    print("Embedding resume chunks...")
    embeddings = model.encode(df["chunk_text"].tolist(), show_progress_bar=True)
    np.save(VEC_OUTPUT, embeddings)
    print(f"Saved to {VEC_OUTPUT}")

if __name__ == "__main__":
    main()