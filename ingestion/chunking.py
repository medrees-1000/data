from typing import List

def chunk_text(
    text: str,
    chunk_size: int = 400,
    overlap: int = 50
) -> List[str]:
    """
    Splits text into chunks based on word count.
    """
    words = text.split()
    chunks = []

    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = words[start:end]
        chunks.append(" ".join(chunk))
        start += chunk_size - overlap
        
        if chunk_size <= overlap:
            break

    return chunks