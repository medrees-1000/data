from typing import List

def chunk_text(
    text: str,
    chunk_size: int = 400,
    overlap: int = 50
) -> List[str]:
    """
    Splits text into chunks based on word count to maintain semantic meaning.
    Args:
        text: Raw resume text
        chunk_size: Number of words per chunk
        overlap: Number of words to overlap between chunks
    """
    words = text.split()
    chunks = []

    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = words[start:end]
        chunks.append(" ".join(chunk))
        # Move the window forward, but stay back by 'overlap' words
        start += chunk_size - overlap
        
        # Prevent infinite loop if overlap >= chunk_size
        if chunk_size <= overlap:
            break

    return chunks