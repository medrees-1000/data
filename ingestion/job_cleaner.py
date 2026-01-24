"""
Automatically extract relevant sections from job descriptions.
Removes company info, benefits, salary, etc.
"""

import re

# Keywords that indicate relevant sections
RELEVANT_SECTION_KEYWORDS = [
    "responsibilities", "requirements", "qualifications", "required",
    "preferred", "skills", "experience", "education", "what you'll do",
    "what you will do", "what you need", "you will", "must have",
    "should have", "nice to have", "technical skills", "key responsibilities"
]

# Keywords that indicate irrelevant sections  
IRRELEVANT_SECTION_KEYWORDS = [
    "about us", "company overview", "who we are", "our mission", "our values",
    "benefits", "compensation", "salary", "perks", "what we offer",
    "equal opportunity", "eeo", "diversity", "application process",
    "how to apply", "contact", "location details"
]


def extract_requirements_section(job_text: str) -> dict:
    """
    Extract only the requirements/qualifications from job description.
    Separates required vs preferred skills.
    
    Returns:
        dict: {
            "cleaned_text": str (requirements only),
            "required_skills": str,
            "preferred_skills": str,
            "full_text": str (original)
        }
    """
    
    lines = job_text.split('\n')
    
    relevant_lines = []
    required_section = []
    preferred_section = []
    
    current_section = None
    skip_section = False
    
    for line in lines:
        line_lower = line.lower().strip()
        
        # Check if we should skip this section
        if any(keyword in line_lower for keyword in IRRELEVANT_SECTION_KEYWORDS):
            skip_section = True
            continue
        
        # Check if this is a relevant section header
        if any(keyword in line_lower for keyword in RELEVANT_SECTION_KEYWORDS):
            skip_section = False
            
            # Determine if it's required or preferred
            if any(word in line_lower for word in ["required", "must have", "qualifications"]):
                current_section = "required"
            elif any(word in line_lower for word in ["preferred", "nice to have", "bonus"]):
                current_section = "preferred"
            else:
                current_section = "general"
        
        # Add line if not skipping
        if not skip_section and line.strip():
            relevant_lines.append(line)
            
            if current_section == "required":
                required_section.append(line)
            elif current_section == "preferred":
                preferred_section.append(line)
    
    # Combine sections
    cleaned_text = '\n'.join(relevant_lines)
    required_text = '\n'.join(required_section)
    preferred_text = '\n'.join(preferred_section)
    
    # If extraction failed, use heuristic
    if len(cleaned_text) < 100:
        cleaned_text = extract_middle_section(job_text)
    
    return {
        "cleaned_text": cleaned_text,
        "required_skills": required_text,
        "preferred_skills": preferred_text,
        "full_text": job_text
    }


def extract_middle_section(text: str) -> str:
    """
    Fallback: Extract middle 60% of text (usually where requirements are).
    """
    lines = [l for l in text.split('\n') if l.strip()]
    
    if len(lines) < 10:
        return text
    
    # Skip first 20% and last 20%
    start_idx = len(lines) // 5
    end_idx = len(lines) - (len(lines) // 5)
    
    return '\n'.join(lines[start_idx:end_idx])


def identify_required_vs_preferred(job_text: str) -> dict:
    """
    Use regex to separate required vs preferred skills.
    """
    
    # Look for "Required:" section
    required_match = re.search(
        r'(?:required|must have|qualifications?)[\s\S]*?(?=preferred|nice to have|$)',
        job_text,
        re.IGNORECASE
    )
    
    # Look for "Preferred:" section  
    preferred_match = re.search(
        r'(?:preferred|nice to have|bonus)[\s\S]*?(?=\n\n|$)',
        job_text,
        re.IGNORECASE
    )
    
    required_text = required_match.group(0) if required_match else ""
    preferred_text = preferred_match.group(0) if preferred_match else ""
    
    return {
        "required": required_text,
        "preferred": preferred_text
    }