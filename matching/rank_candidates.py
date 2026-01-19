import pandas as pd
from similarity import rank_resumes
from scoring_utils import aggregate_resume_scores, get_match_reason  

def main():
    # 1. Get chunk-level matches (increase top_k to 20 to ensure we find all chunks)
    print("Running similarity search...")
    # job_index=0 is cybersecurity based on your Day 3 run
    top_chunks = rank_resumes(job_index=0, top_k=20) 

    # 2. Aggregate to Candidate level
    print("Aggregating scores per candidate...")
    top_candidates = aggregate_resume_scores(top_chunks, top_n=5)

    # 3. Generate Report
    print("\n" + "="*60)
    print("RECRUITER REPORT: TOP CANDIDATES")
    print("="*60)

    for i, row in top_candidates.iterrows():
        res_id = row['resume_id']
        score = row['score']
        
        # Get the 'Why' for this match
        reason = get_match_reason(top_chunks, res_id)
        
        print(f"RANK #{i+1}")
        print(f"CANDIDATE ID: {res_id}")
        print(f"MATCH SCORE : {score:.4f}")
        print(f"MATCH REASON: {reason}")
        print("-" * 30)

    print("="*60)

if __name__ == "__main__":
    main()