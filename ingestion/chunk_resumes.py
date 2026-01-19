import pandas as pd
from pathlib import Path
from chunking import chunk_text

# Paths
INPUT_FILE = Path("data/processed_resumes.csv")
OUTPUT_FILE = Path("data/resume_chunks.csv")

def main():
    if not INPUT_FILE.exists():
        print(f"Error: {INPUT_FILE} not found. Did you run Day 1 correctly?")
        return

    df = pd.read_csv(INPUT_FILE)
    chunk_records = []

    print(f"Chunking {len(df)} resumes...")

    for _, row in df.iterrows():
        chunks = chunk_text(str(row["raw_text"]))
        for i, chunk in enumerate(chunks):
            chunk_records.append({
                "resume_id": row["resume_id"],
                "chunk_id": f"{row['resume_id']}_{i}",
                "chunk_text": chunk
            })

    chunk_df = pd.DataFrame(chunk_records)
    chunk_df.to_csv(OUTPUT_FILE, index=False)

    print(f"Created {len(chunk_df)} resume chunks.")
    print(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()