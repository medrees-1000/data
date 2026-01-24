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
from ingestion.job_cleaner import extract_requirements_section
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

# Custom CSS with FIXED colors
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
        background: #e3f2fd;
        padding: 1.2rem;
        border-radius: 10px;
        border-left: 5px solid #2196f3;
        margin: 1rem 0;
        color: #1a1a1a;
    }
    .info-box b {
        color: #1565c0;
        font-size: 1.05rem;
    }
    .warning-box {
        background: #fff8e1;
        padding: 1.2rem;
        border-radius: 10px;
        border-left: 5px solid #ff9800;
        margin: 1rem 0;
        color: #1a1a1a;
    }
    .warning-box b {
        color: #e65100;
        font-size: 1.05rem;
    }
    .success-box {
        background: #e8f5e9;
        padding: 1.2rem;
        border-radius: 10px;
        border-left: 5px solid #4caf50;
        margin: 1rem 0;
        color: #1a1a1a;
    }
    .success-box b {
        color: #2e7d32;
        font-size: 1.05rem;
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
    
    /* Speedometer gauge */
    .gauge-container {
        text-align: center;
        margin: 2rem 0;
    }
    .gauge {
        width: 300px;
        height: 150px;
        margin: 0 auto;
        position: relative;
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
    - Paste everything - we clean it!
    
    **3. Get Analysis**
    - Match score breakdown
    - Skill gap analysis
    - AI-powered insights
    """)
    
    st.divider()
    
    st.header("🎯 Scoring Method")
    st.markdown("""
    **Hybrid Score Breakdown:**
    - 40% Keyword matching
    - 30% Semantic similarity
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
    <b>✅ What Works Best:</b><br><br>
    • <b>PDF format only</b> (no Word docs)<br>
    • <b>Text-based PDF</b> - you can select/copy text<br>
    • <b>Standard layouts</b> - traditional resume format<br>
    • <b>Under 5MB</b> file size
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="warning-box">
    <b>⚠️ Won't Work:</b><br><br>
    • <b>Scanned images</b> saved as PDF<br>
    • <b>Highly graphic/creative</b> resumes<br>
    • <b>Password-protected</b> files
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose your resume PDF",
        type=["pdf"],
        help="Upload a text-based PDF resume"
    )
    
    if uploaded_file:
        st.markdown(f"""
        <div class="success-box">
        <b>✅ File Uploaded!</b><br>
        {uploaded_file.name} ({uploaded_file.size / 1024:.1f} KB)
        </div>
        """, unsafe_allow_html=True)

with col2:
    st.subheader("💼 Job Description")
    
    st.markdown("""
    <div class="info-box">
    <b>💡 Pro Tips:</b><br><br>
    • <b>Paste ENTIRE job post</b> - we'll clean it automatically!<br>
    • <b>Include everything:</b> company info, requirements, all of it<br>
    • <b>Don't edit</b> - just copy & paste as-is<br>
    • <b>More detail = better results!</b>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="success-box">
    <b>✨ Auto-Cleaning:</b><br><br>
    Our AI automatically removes:<br>
    • Company fluff & benefits<br>
    • Separates Required vs Preferred skills<br>
    • Focuses on actual qualifications
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
        "Paste the complete job posting:",
        value=default_text,
        height=280,
        placeholder="Copy and paste the ENTIRE job description here - we'll handle the rest!",
        help="Don't worry about cleaning it - our system does that automatically"
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
        st.caption("Step 2/5: Cleaning job description...")
        
        # Step 2: Clean and process job description
        job_sections = extract_requirements_section(job_description)
        cleaned_job_text = job_sections["cleaned_text"]
        
        job_result = process_job_description(cleaned_job_text)
        
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
        st.caption("Step 4/5: Matching keywords (Required vs Preferred)...")
        
        # Step 4: Keyword matching with job sections
        resume_keywords = extract_keywords(resume_result["text"])
        job_keywords = extract_keywords(cleaned_job_text)
        keyword_results = calculate_keyword_match(resume_keywords, job_keywords, job_sections)
        
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
    
    # Overall Score Display with Gauge
    st.markdown("---")
    st.subheader("📊 Overall Match Score")
    
    hybrid_score = score_breakdown["hybrid_score"]
    match_icon = score_breakdown["match_icon"]
    match_category = score_breakdown["match_category"]
    
    # Determine color
    if hybrid_score >= 0.70:
        score_class = "excellent"
        color = "#667eea"
    elif hybrid_score >= 0.55:
        score_class = "good"
        color = "#f093fb"
    elif hybrid_score >= 0.40:
        score_class = "moderate"
        color = "#4facfe"
    else:
        score_class = "low"
        color = "#fa709a"
    
    # Create gauge visualization
    import plotly.graph_objects as go
    
    # Determine gauge color based on score
    if hybrid_score >= 0.70:
        gauge_color = "#4caf50"  # Green for excellent
    elif hybrid_score >= 0.55:
        gauge_color = "#2196f3"  # Blue for good
    elif hybrid_score >= 0.40:
        gauge_color = "#ff9800"  # Orange for moderate
    else:
        gauge_color = "#f44336"  # Red for low
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = hybrid_score * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {
            'text': f"<b>{match_icon} {match_category}</b>", 
            'font': {'size': 28, 'color': '#1a1a1a', 'family': 'Arial'}
        },
        number = {
            'suffix': "%", 
            'font': {'size': 80, 'color': '#1a1a1a', 'family': 'Arial Bold'}
        },
        gauge = {
            'axis': {
                'range': [None, 100], 
                'tickwidth': 2, 
                'tickcolor': "#666",
                'tickfont': {'size': 14, 'color': '#1a1a1a'}
            },
            'bar': {'color': gauge_color, 'thickness': 0.75},
            'bgcolor': "#f5f5f5",
            'borderwidth': 3,
            'bordercolor': "#e0e0e0",
            'steps': [
                {'range': [0, 40], 'color': '#ffebee'},
                {'range': [40, 55], 'color': '#fff9c4'},
                {'range': [55, 70], 'color': '#e1f5fe'},
                {'range': [70, 100], 'color': '#e8f5e9'}
            ],
            'threshold': {
                'line': {'color': gauge_color, 'width': 6},
                'thickness': 0.8,
                'value': hybrid_score * 100
            }
        }
    ))
    
    fig.update_layout(
        height=380,
        margin=dict(l=40, r=40, t=80, b=20),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={'color': "#1a1a1a", 'family': "Arial"}
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
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
    
    # Skills Analysis with Required vs Preferred
    st.markdown("---")
    st.subheader("🎯 Skills Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**✅ Matched Skills**")
        matched_skills = score_breakdown["matched_skills"]
        if matched_skills:
            for skill in matched_skills[:10]:
                st.markdown(f"✓ {skill}")
            if len(matched_skills) > 10:
                st.caption(f"...and {len(matched_skills) - 10} more")
        else:
            st.info("No specific technical skills detected")
    
    with col2:
        st.markdown("**⚠️ Missing Skills**")
        
        missing_required = score_breakdown.get("missing_required", [])
        missing_preferred = score_breakdown.get("missing_preferred", [])
        
        if missing_required:
            st.markdown("**🔴 Required (High Priority):**")
            for skill in missing_required[:5]:
                st.markdown(f"❌ {skill}")
        
        if missing_preferred:
            st.markdown("**🟡 Preferred (Nice to Have):**")
            for skill in missing_preferred[:3]:
                st.markdown(f"⚠️ {skill}")
        
        if not missing_required and not missing_preferred:
            st.success("All skills found!")
    
    # Tabs for detailed analysis
    st.markdown("---")
    tab1, tab2, tab3, tab4 = st.tabs(["🤖 AI Explanation", "🔍 Matching Sections", "📋 Full Resume", "🧹 Job Cleaning"])
    
    with tab1:
        st.subheader("AI-Powered Analysis")
        
        if os.getenv("GROQ_API_KEY"):
            with st.spinner("Generating AI explanation..."):
                explanation = generate_match_explanation_groq(
                    [c["chunk"] for c in top_chunks],
                    cleaned_job_text,
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
    
    with tab4:
        st.subheader("How We Cleaned the Job Description")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Original", f"{len(job_description)} chars")
            st.metric("Cleaned", f"{len(cleaned_job_text)} chars")
            reduction = ((len(job_description) - len(cleaned_job_text)) / len(job_description)) * 100
            st.metric("Removed", f"{reduction:.0f}%")
        
        with col2:
            st.markdown("**🎯 Kept:**")
            st.markdown("- Responsibilities")
            st.markdown("- Requirements")
            st.markdown("- Skills")
            
            st.markdown("**🗑️ Removed:**")
            st.markdown("- Company info")
            st.markdown("- Benefits")
            st.markdown("- Salary")
        
        with st.expander("View Cleaned Job Text"):
            st.text_area(
                "Cleaned version:",
                cleaned_job_text,
                height=300,
                label_visibility="collapsed"
            )

# Footer
st.markdown("---")
st.caption("🧠 Powered by all-mpnet-base-v2 embeddings + Groq Llama 3.3 | Weighted Required/Preferred scoring")
st.caption("💡 Auto-cleans job descriptions • Smaller chunks • Cross-domain compatible")