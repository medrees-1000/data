import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def calculate_match_score(user_embedding, job_embedding):
    """
    Calculates the cosine similarity between a user resume and a job.
    """
    # Ensure vectors are 2D for sklearn
    user_vec = user_embedding.reshape(1, -1)
    job_vec = job_embedding.reshape(1, -1)
    
    score = cosine_similarity(user_vec, job_vec)[0][0]
    return float(score)