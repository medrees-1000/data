import pandas as pd

def aggregate_resume_scores(top_chunks_df, top_n=5):
    """
    Aggregates chunk-level scores into resume-level rankings.
    Uses the MAX score per resume (identifying the best matching experience).
    """
    # Group by resume_id and find the highest chunk score for each person
    agg = (
        top_chunks_df
        .groupby("resume_id")["score"]
        .max()
        .reset_index()
        .sort_values("score", ascending=False)
        .head(top_n)
        .reset_index(drop=True)  # Added this line to fix the Rank numbering
    )
    return agg

def get_match_reason(resume_chunks_df, resume_id, top_k=1):
    """
    Retrieves the specific text snippet that caused the high score.
    """
    reason = (
        resume_chunks_df[resume_chunks_df["resume_id"] == resume_id]
        .sort_values("score", ascending=False)
        .head(top_k)["chunk_text"]
        .values[0]
    )
    # Return the first 200 characters as a summary reason
    return reason[:200] + "..."