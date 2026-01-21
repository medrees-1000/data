"""
Keyword extraction and matching for resume-job comparison.
Extracts technical skills, tools, and requirements.
"""

import re
from typing import Dict, List, Set

# Common technical skills database
TECH_SKILLS = {
    # Programming Languages
    "python", "java", "javascript", "c++", "c#", "r", "sql", "scala", "go", "rust",
    "typescript", "php", "swift", "kotlin", "ruby",
    
    # Data Science & ML
    "machine learning", "deep learning", "nlp", "natural language processing",
    "computer vision", "data science", "data analysis", "statistics", "pandas",
    "numpy", "scikit-learn", "tensorflow", "pytorch", "keras", "lightgbm", "xgboost",
    
    # Big Data & Cloud
    "aws", "azure", "gcp", "google cloud", "spark", "hadoop", "kafka", "airflow",
    "databricks", "snowflake", "bigquery", "redshift",
    
    # Databases
    "mysql", "postgresql", "mongodb", "redis", "cassandra", "dynamodb", "oracle",
    "sql server",
    
    # Tools & Frameworks
    "git", "github", "docker", "kubernetes", "jenkins", "terraform", "ansible",
    "tableau", "power bi", "excel", "jupyter", "flask", "django", "fastapi",
    "react", "node.js", "spring boot",
    
    # Data Engineering
    "etl", "data pipeline", "data warehouse", "api", "rest api", "microservices",
    
    # AI & GenAI
    "generative ai", "genai", "llm", "large language model", "langchain",
    "hugging face", "openai", "vertex ai", "prompt engineering",
    
    # Methodologies
    "agile", "scrum", "ci/cd", "devops", "mlops"
}

# Education keywords
EDUCATION_LEVELS = {
    "phd", "doctorate", "master", "masters", "bachelor", "bachelors", "bs", "ba", "ms", "mba"
}

# Experience levels
EXPERIENCE_KEYWORDS = {
    "intern", "internship", "entry level", "junior", "mid level", "senior", "lead", "principal"
}


def extract_keywords(text: str) -> Dict[str, Set[str]]:
    """
    Extract structured keywords from text.
    
    Returns:
        dict: {
            "technical_skills": set of tech skills found,
            "tools": set of tools/frameworks,
            "education": set of education levels,
            "experience_level": set of experience indicators
        }
    """
    text_lower = text.lower()
    
    # Find technical skills
    found_skills = set()
    for skill in TECH_SKILLS:
        # Use word boundaries to avoid partial matches
        if re.search(r'\b' + re.escape(skill) + r'\b', text_lower):
            found_skills.add(skill)
    
    # Find education levels
    found_education = set()
    for edu in EDUCATION_LEVELS:
        if re.search(r'\b' + re.escape(edu) + r'\b', text_lower):
            found_education.add(edu)
    
    # Find experience level
    found_experience = set()
    for exp in EXPERIENCE_KEYWORDS:
        if re.search(r'\b' + re.escape(exp) + r'\b', text_lower):
            found_experience.add(exp)
    
    return {
        "technical_skills": found_skills,
        "education": found_education,
        "experience_level": found_experience
    }


def calculate_keyword_match(resume_keywords: Dict, job_keywords: Dict) -> Dict[str, float]:
    """
    Calculate keyword overlap between resume and job.
    
    Returns:
        dict: {
            "technical_score": 0-1,
            "education_score": 0-1,
            "experience_score": 0-1,
            "overall_keyword_score": 0-1,
            "matched_skills": list,
            "missing_skills": list
        }
    """
    resume_skills = resume_keywords.get("technical_skills", set())
    job_skills = job_keywords.get("technical_skills", set())
    
    # Technical skills match
    if job_skills:
        matched_skills = resume_skills.intersection(job_skills)
        missing_skills = job_skills - resume_skills
        technical_score = len(matched_skills) / len(job_skills)
    else:
        matched_skills = set()
        missing_skills = set()
        technical_score = 0.5  # Neutral if no specific skills listed
    
    # Education match
    resume_edu = resume_keywords.get("education", set())
    job_edu = job_keywords.get("education", set())
    
    education_score = 1.0 if (resume_edu and job_edu and resume_edu.intersection(job_edu)) else 0.5
    
    # Experience level match
    resume_exp = resume_keywords.get("experience_level", set())
    job_exp = job_keywords.get("experience_level", set())
    
    experience_score = 1.0 if (resume_exp and job_exp and resume_exp.intersection(job_exp)) else 0.7
    
    # Overall keyword score (weighted average)
    overall = (
        0.70 * technical_score +
        0.20 * education_score +
        0.10 * experience_score
    )
    
    return {
        "technical_score": technical_score,
        "education_score": education_score,
        "experience_score": experience_score,
        "overall_keyword_score": overall,
        "matched_skills": sorted(list(matched_skills)),
        "missing_skills": sorted(list(missing_skills))
    }


def get_improvement_suggestions(missing_skills: List[str], matched_skills: List[str]) -> List[str]:
    """
    Generate actionable improvement suggestions.
    """
    suggestions = []
    
    if len(missing_skills) > 0:
        top_missing = missing_skills[:5]
        suggestions.append(f"Add these key skills to your resume: {', '.join(top_missing)}")
    
    if len(matched_skills) < 5:
        suggestions.append("Expand your technical skills section with more specific tools and frameworks")
    
    if "python" in missing_skills:
        suggestions.append("Python is required - add Python projects to your experience section")
    
    if any(skill in missing_skills for skill in ["aws", "azure", "gcp"]):
        suggestions.append("Consider getting cloud platform experience (AWS/Azure/GCP)")
    
    if not suggestions:
        suggestions.append("Strong skill match! Consider highlighting achievements and impact in your experience")
    
    return suggestions