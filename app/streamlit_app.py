"""
AI Resume-Job Matching Platform
Hybrid scoring with semantic embeddings + keyword matching
"""

import streamlit as st
import sys
from pathlib import Path
import os

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Import core functions
from ingestion.process_resume import process_uploaded_resume, process_job_description
from matching.similarity import calculate_match_score, get_top_matching_chunks
from matching.keyword_matcher import extract_keywords, calculate_keyword_match, get_improvement_suggestions
from matching.hybrid_scorer import calculate_hybrid_score, generate_score_explanation
from rag.groq_explainer import generate_match_explanation_groq, generate_simple_explanation_fallback

# Page config
st.set_page_config(
    page_title="AI Resume Matcher",
    page_icon="🎯",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .match-score-box {
        font-size: 4rem;
        font-weight: bold;
        text-align: center;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
    }
    .excellent { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; }
    .good { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; }
    .moderate { background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); color: white; }
    .low { background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); color: white; }
    
    .info-box {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .warning-box {
        background: #fff3cd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        padding: 0.75rem;
        font-size: 1.1rem;
        border-radius: 10px;
        border: none;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">🎯 AI Resume-Job Matching Platform</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Hybrid scoring: Semantic embeddings + Keyword matching + AI explanations</p>', unsafe_allow_html=True)

# Sidebar - Info & Tips
with st.sidebar:
    st.header("📋 How It Works")
    
    st.markdown("""
    **1. Upload Resume**
    - PDF format only
    - Text-based (not scanned)
    
    **2. Paste Job Description**
    - Include requirements only
    - Skip company info
    
    **3. Get Analysis**
    - Match score breakdown
    - Skill gap analysis
    - AI-powered insights
    """)
    
    st.divider()
    
    st.header("🎯 Scoring Method")
    st.markdown("""
    **Hybrid Score Breakdown:**
    - 35% Keyword matching
    - 35% Semantic similarity
    - 20% Experience match
    - 10% Education match
    """)
    
    st.divider()
    
    # API Status
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        st.success("✅ AI Explanations: Active")
    else:
        st.warning("⚠️ AI Explanations: Disabled\n\nAdd GROQ_API_KEY to .env")
        st.caption("[Get free key](https://console.groq.com)")

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📄 Upload Resume")
    
    st.markdown("""
    <div class="info-box">
    <b>✅ Requirements:</b><br>
    • PDF format only<br>
    • Text-based (not scanned image)<br>
    • Standard resume format<br>
    • Under 5MB
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="warning-box">
    <b>⚠️ Common Issues:</b><br>
    • Scanned PDFs won't work<br>
    • Highly graphic resumes may fail<br>
    • Use text-selectable PDFs
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose PDF file",
        type=["pdf"],
        help="Upload candidate resume for analysis"
    )
    
    if uploaded_file:
        st.success(f"✅ Uploaded: {uploaded_file.name}")

with col2:
    st.subheader("💼 Job Description")
    
    st.markdown("""
    <div class="info-box">
    <b>💡 Best Results:</b><br>
    • Paste ONLY job requirements<br>
    • Include: responsibilities, skills, qualifications<br>
    • Skip: company info, benefits, salary
    </div>
    """, unsafe_allow_html=True)
    
    # Sample job selector
    sample_jobs = {
        "Select a sample...": "",
        "Data Scientist": Path("data/jobs/data_scientist.txt"),
        "Data Analyst": Path("data/jobs/data_analyst.txt"),
        "Software Engineer": Path("data/jobs/software_engineer.txt"),
        "DevOps Engineer": Path("data/jobs/devops_engineer.txt"),
        "Cybersecurity Specialist": Path("data/jobs/cybersecurity.txt")
    }
    
    selected_sample = st.selectbox("📂 Load sample job (optional):", list(sample_jobs.keys()))
    
    # Load sample text
    default_text = ""
    if selected_sample != "Select a sample...":
        job_file = sample_jobs[selected_sample]
        if job_file.exists():
            default_text = job_file.read_text()
    
    job_description = st.text_area(
        "Paste job description:",
        value=default_text,
        height=250,
        placeholder="Paste the job requirements here..."
    )

# Analysis button
st.divider()

if st.button("🚀 Analyze Match", type="primary"):
    
    # Validation
    if not uploaded_file:
        st.error("❌ Please upload a resume first!")
        st.stop()
    
    if not job_description or len(job_description.strip()) < 50:
        st.error("❌ Please provide a job description (at least 50 characters)!")
        st.stop()
    
    # Processing with progress
    with st.spinner("🔄 Processing resume and analyzing match..."):
        
        # Step 1: Process resume
        progress_bar = st.progress(0)
        st.caption("Step 1/5: Extracting text from PDF...")
        
        resume_result = process_uploaded_resume(uploaded_file)
        
        if not resume_result["success"]:
            st.error(f"❌ Resume processing failed: {resume_result['error']}")
            st.stop()
        
        progress_bar.progress(20)
        st.caption("Step 2/5: Embedding resume chunks...")
        
        # Step 2: Process job description
        job_result = process_job_description(job_description)
        
        if not job_result["success"]:
            st.error(f"❌ Job processing failed: {job_result['error']}")
            st.stop()
        
        progress_bar.progress(40)
        st.caption("Step 3/5: Calculating semantic similarity...")
        
        # Step 3: Calculate semantic similarity
        top_chunks = get_top_matching_chunks(
            resume_result["chunks"],
            resume_result["embeddings"],
            job_result["embedding"],
            top_k=5
        )
        
        # Average of top 3 chunks for semantic score
        semantic_score = sum([c["score"] for c in top_chunks[:3]]) / 3
        
        progress_bar.progress(60)
        st.caption("Step 4/5: Extracting keywords and matching...")
        
        # Step 4: Keyword matching
        resume_keywords = extract_keywords(resume_result["text"])
        job_keywords = extract_keywords(job_description)
        keyword_results = calculate_keyword_match(resume_keywords, job_keywords)
        
        progress_bar.progress(80)
        st.caption("Step 5/5: Generating hybrid score...")
        
        # Step 5: Hybrid scoring
        score_breakdown = calculate_hybrid_score(
            semantic_score,
            keyword_results,
            top_chunks
        )
        
        progress_bar.progress(100)
        st.caption("✅ Analysis complete!")
        
    # Clear progress indicators
    progress_bar.empty()
    
    # Display Results
    st.success("✅ Analysis Complete!")
    
    # Overall Score Display
    st.markdown("---")
    st.subheader("📊 Overall Match Score")
    
    hybrid_score = score_breakdown["hybrid_score"]
    match_icon = score_breakdown["match_icon"]
    match_category = score_breakdown["match_category"]
    
    # Determine color class
    if hybrid_score >= 0.75:
        score_class = "excellent"
    elif hybrid_score >= 0.60:
        score_class = "good"
    elif hybrid_score >= 0.45:
        score_class = "moderate"
    else:
        score_class = "low"
    
    st.markdown(
        f'<div class="match-score-box {score_class}">{match_icon} {hybrid_score:.1%}<br><small>{match_category}</small></div>',
        unsafe_allow_html=True
    )
    
    st.info(f"**Recommendation:** {score_breakdown['recommendation']}")
    
    # Score Breakdown
    st.markdown("---")
    st.subheader("📈 Score Breakdown")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "🔧 Technical Skills",
            f"{score_breakdown['technical_score']:.1%}",
            help="Keyword matching of technical skills"
        )
    
    with col2:
        st.metric(
            "🧠 Semantic Fit",
            f"{score_breakdown['semantic_score']:.1%}",
            help="Overall contextual match using embeddings"
        )
    
    with col3:
        st.metric(
            "💼 Experience",
            f"{score_breakdown['experience_score']:.1%}",
            help="Experience level alignment"
        )
    
    with col4:
        st.metric(
            "🎓 Education",
            f"{score_breakdown['education_score']:.1%}",
            help="Education requirements match"
        )
    
    # Skills Analysis
    st.markdown("---")
    st.subheader("🎯 Skills Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**✅ Matched Skills**")
        matched_skills = score_breakdown["matched_skills"]
        if matched_skills:
            for skill in matched_skills[:10]:
                st.markdown(f"- {skill}")
            if len(matched_skills) > 10:
                st.caption(f"...and {len(matched_skills) - 10} more")
        else:
            st.info("No specific technical skills detected in job description")
    
    with col2:
        st.markdown("**⚠️ Missing Skills**")
        missing_skills = score_breakdown["missing_skills"]
        if missing_skills:
            for skill in missing_skills[:10]:
                st.markdown(f"- {skill}")
            if len(missing_skills) > 10:
                st.caption(f"...and {len(missing_skills) - 10} more")
        else:
            st.success("All required skills found!")
    
    # Tabs for detailed analysis
    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["🤖 AI Explanation", "🔍 Matching Sections", "📋 Full Resume"])
    
    with tab1:
        st.subheader("AI-Powered Analysis")
        
        if os.getenv("GROQ_API_KEY"):
            with st.spinner("Generating AI explanation..."):
                explanation = generate_match_explanation_groq(
                    [c["chunk"] for c in top_chunks],
                    job_description,
                    score_breakdown
                )
        else:
            explanation = generate_simple_explanation_fallback(score_breakdown)
        
        st.markdown("**Why This Candidate Matches:**")
        st.write(explanation["explanation"])
        
        if explanation["strengths"]:
            st.markdown("**✨ Key Strengths:**")
            for strength in explanation["strengths"]:
                st.markdown(f"- {strength}")
        
        if explanation["gaps"]:
            st.markdown("**⚠️ Areas for Improvement:**")
            for gap in explanation["gaps"]:
                st.markdown(f"- {gap}")
        
        if explanation.get("suggestions"):
            st.markdown("**💡 Suggestions:**")
            for suggestion in explanation["suggestions"]:
                st.markdown(f"- {suggestion}")
    
    with tab2:
        st.subheader("Top Matching Resume Sections")
        
        for i, chunk in enumerate(top_chunks[:5], 1):
            with st.expander(f"Match #{i} - Relevance: {chunk['score']:.1%}", expanded=(i==1)):
                st.markdown(f"**Similarity Score:** {chunk['score']:.1%}")
                st.markdown("**Content:**")
                st.text_area(
                    f"Section {i}",
                    chunk["chunk"],
                    height=150,
                    key=f"chunk_{i}",
                    label_visibility="collapsed"
                )
    
    with tab3:
        st.subheader("Complete Resume Content")
        st.text_area(
            "Full resume text:",
            resume_result["text"],
            height=400,
            label_visibility="collapsed"
        )
        
        # Resume stats
        st.caption(f"📊 Resume Statistics:")
        st.caption(f"• Total length: {len(resume_result['text'])} characters")
        st.caption(f"• Number of chunks: {len(resume_result['chunks'])}")
        st.caption(f"• Chunks analyzed: {len(top_chunks)}")

# Footer
st.markdown("---")
st.caption("🧠 Powered by all-mpnet-base-v2 embeddings + Groq Llama 3.3 | Hybrid scoring system")
st.caption("💡 This tool uses semantic AI to understand context, not just keyword matching")