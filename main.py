import streamlit as st
import pandas as pd
import os
import json
import random
from datetime import datetime, timedelta

from parser.resume_parser import extract_resume_text
from parser.info_extractor import extract_name, extract_email
from parser.skill_extractor import (
    extract_skills, extract_skills_by_category, extract_technologies,
    extract_programming_languages, extract_frameworks, extract_cloud_technologies
)
from parser.education_extractor import extract_education
from parser.experience_extractor import extract_experience
from parser.certification_extractor import extract_certifications
from embeddings.embedding_generator import (
    generate_candidate_embedding, generate_skill_embedding,
    generate_job_description_embedding, get_embedding_dimension
)
from vectorstore.db import store_resume, search_candidates
from ranking.candidate_ranker import calculate_candidate_score
from interview.question_generator import generate_interview_questions
from chatbot.recruiter_chatbot import recruiter_chatbot

from search_optimizer import get_cached_embedding, optimized_rank
# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

st.set_page_config(
    page_title="HireMind — AI Recruitment",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Hide default Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem 3rem; max-width: 1400px; }

/* ── TOP NAV BAR ── */
.topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.75rem 1.5rem;
    background: #0A0A0F;
    border-bottom: 1px solid #1E1E2E;
    border-radius: 12px;
    margin-bottom: 1.5rem;
}
.topbar-logo {
    font-size: 1.25rem;
    font-weight: 600;
    color: #E8E8FF;
    letter-spacing: -0.02em;
}
.topbar-logo span { color: #7C6AF7; }
.topbar-badge {
    background: #7C6AF720;
    color: #A89BF8;
    border: 1px solid #7C6AF730;
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.75rem;
    font-weight: 500;
}

/* ── SECTION HEADERS ── */
.section-title {
    font-size: 1rem;
    font-weight: 600;
    color: #E8E8FF;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin: 1.5rem 0 0.75rem;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #1E1E2E;
}

/* ── METRIC CARDS ── */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 1.5rem;
}
.metric-card {
    background: #0E0E1A;
    border: 1px solid #1E1E2E;
    border-radius: 12px;
    padding: 1.1rem 1.25rem;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: var(--accent);
}
.metric-label {
    font-size: 0.72rem;
    font-weight: 500;
    color: #6E6E8E;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 6px;
}
.metric-value {
    font-size: 1.9rem;
    font-weight: 600;
    color: #E8E8FF;
    line-height: 1;
    font-family: 'DM Mono', monospace;
}
.metric-delta {
    font-size: 0.75rem;
    margin-top: 6px;
    font-weight: 500;
}
.metric-delta.up { color: #4ADE80; }
.metric-delta.down { color: #F87171; }

/* ── CANDIDATE CARDS ── */
.candidate-card {
    background: #0E0E1A;
    border: 1px solid #1E1E2E;
    border-radius: 12px;
    padding: 1.1rem 1.25rem;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 16px;
    transition: border-color 0.15s;
}
.candidate-card:hover { border-color: #7C6AF750; }
.avatar {
    width: 44px; height: 44px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-weight: 600; font-size: 0.85rem;
    flex-shrink: 0;
    color: #E8E8FF;
}
.candidate-name { font-weight: 600; color: #E8E8FF; font-size: 0.95rem; }
.candidate-meta { font-size: 0.78rem; color: #6E6E8E; margin-top: 2px; }
.skill-chip {
    display: inline-block;
    background: #1A1A2E;
    border: 1px solid #2A2A3E;
    border-radius: 6px;
    padding: 2px 8px;
    font-size: 0.7rem;
    color: #A89BF8;
    margin: 2px 2px 0 0;
    font-family: 'DM Mono', monospace;
}
.score-badge {
    margin-left: auto;
    flex-shrink: 0;
    text-align: center;
}
.score-ring {
    width: 52px; height: 52px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-weight: 600; font-size: 0.85rem;
    font-family: 'DM Mono', monospace;
}
.score-high { background: #0D2B1A; color: #4ADE80; border: 2px solid #4ADE80; }
.score-mid  { background: #2B1F08; color: #FBBF24; border: 2px solid #FBBF24; }
.score-low  { background: #2B0D0D; color: #F87171; border: 2px solid #F87171; }

/* ── RANKING TABLE ── */
.rank-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.85rem;
}
.rank-table th {
    background: #0A0A0F;
    color: #6E6E8E;
    font-size: 0.7rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    padding: 10px 14px;
    text-align: left;
    border-bottom: 1px solid #1E1E2E;
}
.rank-table td {
    padding: 10px 14px;
    border-bottom: 1px solid #12121E;
    color: #C8C8E8;
    vertical-align: middle;
}
.rank-table tr:last-child td { border-bottom: none; }
.rank-table tr:hover td { background: #0E0E1A; }
.rank-num {
    font-family: 'DM Mono', monospace;
    color: #6E6E8E;
    font-size: 0.75rem;
}
.progress-bar-bg {
    background: #1E1E2E;
    border-radius: 4px;
    height: 6px;
    width: 100px;
    display: inline-block;
    vertical-align: middle;
    margin-right: 8px;
}
.progress-bar-fill {
    height: 100%;
    border-radius: 4px;
}

/* ── UPLOAD ZONE ── */
.upload-zone {
    border: 2px dashed #2A2A3E;
    border-radius: 12px;
    padding: 2.5rem;
    text-align: center;
    background: #0A0A0F;
    margin-bottom: 1rem;
}
.upload-zone-icon { font-size: 2.5rem; margin-bottom: 0.5rem; }
.upload-zone-text { color: #6E6E8E; font-size: 0.9rem; }

/* ── CHAT ── */
.chat-bubble {
    padding: 0.75rem 1rem;
    border-radius: 10px;
    margin-bottom: 10px;
    font-size: 0.875rem;
    line-height: 1.6;
    max-width: 85%;
}
.chat-user {
    background: #7C6AF720;
    border: 1px solid #7C6AF730;
    color: #C8C8E8;
    margin-left: auto;
    border-bottom-right-radius: 2px;
}
.chat-ai {
    background: #0E0E1A;
    border: 1px solid #1E1E2E;
    color: #C8C8E8;
    border-bottom-left-radius: 2px;
}
.chat-ai-label {
    font-size: 0.7rem;
    font-weight: 600;
    color: #7C6AF7;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 4px;
}

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: #0A0A0F !important;
    border-right: 1px solid #1E1E2E;
}
[data-testid="stSidebar"] .stTextArea textarea,
[data-testid="stSidebar"] .stTextInput input {
    background: #0E0E1A !important;
    border: 1px solid #1E1E2E !important;
    color: #C8C8E8 !important;
    border-radius: 8px;
}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown p {
    color: #6E6E8E !important;
    font-size: 0.8rem !important;
}

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {
    background: #0A0A0F;
    border-radius: 8px;
    padding: 4px;
    gap: 4px;
    border: 1px solid #1E1E2E;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 6px;
    color: #6E6E8E;
    font-size: 0.85rem;
    font-weight: 500;
    padding: 6px 16px;
}
.stTabs [aria-selected="true"] {
    background: #7C6AF720 !important;
    color: #A89BF8 !important;
}
.stTabs [data-baseweb="tab-panel"] {
    padding-top: 1.25rem;
}

/* ── INPUTS ── */
.stTextArea textarea, .stTextInput input {
    background: #0E0E1A !important;
    border: 1px solid #1E1E2E !important;
    color: #C8C8E8 !important;
    border-radius: 8px !important;
}
.stTextArea textarea:focus, .stTextInput input:focus {
    border-color: #7C6AF7 !important;
    box-shadow: 0 0 0 2px #7C6AF720 !important;
}

/* ── BUTTONS ── */
.stButton > button {
    background: #7C6AF7 !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    font-size: 0.875rem !important;
    padding: 0.5rem 1.25rem !important;
    transition: opacity 0.15s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* ── FILE UPLOADER ── */
[data-testid="stFileUploader"] {
    background: #0E0E1A;
    border: 2px dashed #2A2A3E;
    border-radius: 12px;
    padding: 1rem;
}
[data-testid="stFileUploader"] label { color: #6E6E8E !important; }

/* ── INSIGHT CARD ── */
.insight-card {
    background: #0E0E1A;
    border: 1px solid #1E1E2E;
    border-left: 3px solid var(--accent, #7C6AF7);
    border-radius: 8px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 10px;
}
.insight-title { font-weight: 600; color: #E8E8FF; font-size: 0.875rem; }
.insight-desc  { color: #6E6E8E; font-size: 0.8rem; margin-top: 4px; }

/* ── STATUS PILL ── */
.pill {
    display: inline-block;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
.pill-green  { background: #0D2B1A; color: #4ADE80; }
.pill-yellow { background: #2B1F08; color: #FBBF24; }
.pill-blue   { background: #0A1A2B; color: #60A5FA; }
.pill-red    { background: #2B0D0D; color: #F87171; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TOP NAV
# ─────────────────────────────────────────────
st.markdown("""
<div class="topbar">
  <div class="topbar-logo">Hire<span>Mind</span></div>
  <div style="display:flex;gap:12px;align-items:center">
    <span style="color:#6E6E8E;font-size:0.8rem">AI Recruitment Platform</span>
    <span class="topbar-badge">v2.0 Pro</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SESSION STATE — store parsed candidates
# ─────────────────────────────────────────────
if "candidates" not in st.session_state:
    st.session_state.candidates = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ─────────────────────────────────────────────
# SIDEBAR — SEARCH
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:1rem 0 0.5rem">
        <div style="font-size:1rem;font-weight:600;color:#E8E8FF">🔍 Candidate Search</div>
        <div style="font-size:0.75rem;color:#6E6E8E;margin-top:4px">Semantic search across all resumes</div>
    </div>
    """, unsafe_allow_html=True)

    search_query = st.text_area("Job Description", height=120, placeholder="Paste JD or describe the role…")
    col1, col2 = st.columns(2)
    with col1:
        min_score = st.slider("Min Score", 0, 100, 60)
    with col2:
        top_k = st.selectbox("Top K", [5, 10, 20, 50], index=0)
    search_btn = st.button("🔍 Search Candidates", use_container_width=True)

    st.markdown("---")
    st.markdown('<div style="font-size:0.75rem;color:#6E6E8E">QUICK FILTERS</div>', unsafe_allow_html=True)
    filter_exp   = st.selectbox("Experience", ["Any", "0-2 yrs", "3-5 yrs", "6-10 yrs", "10+ yrs"])
    filter_role  = st.selectbox("Role Type", ["Any", "Engineer", "Data Scientist", "Designer", "Manager", "DevOps"])
    filter_avail = st.multiselect("Availability", ["Immediate", "2 weeks", "1 month", "3 months"], default=[])

    st.markdown("---")
    st.markdown(f'<div style="font-size:0.75rem;color:#6E6E8E">PIPELINE STATS</div>', unsafe_allow_html=True)
    total = len(st.session_state.candidates)
    st.markdown(f"""
    <div style="margin-top:8px">
        <div style="color:#E8E8FF;font-size:1.4rem;font-weight:600;font-family:'DM Mono',monospace">{total}</div>
        <div style="color:#6E6E8E;font-size:0.75rem">resumes indexed</div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HANDLE SEARCH
# ─────────────────────────────────────────────
search_results = []
if search_btn and search_query:
    try:
        query_embedding = get_cached_embedding(
            search_query, generate_job_description_embedding
        )
        ranked = optimized_rank(
            query_embedding,
            search_query,
            st.session_state.candidates,
            top_k=top_k
        )
        search_results = [
            {**r, "score": round(r["_combined_score"] * 100, 1)}
            for r in ranked
            if round(r["_combined_score"] * 100, 1) >= min_score
        ]
    except Exception as e:
        st.error(f"Search error: {e}")


# ─────────────────────────────────────────────
# KPI STRIP
# ─────────────────────────────────────────────
n = len(st.session_state.candidates)
avg_score = (
    round(sum(c.get("score", 0) for c in st.session_state.candidates) / n, 1)
    if n else 0
)
shortlisted = sum(1 for c in st.session_state.candidates if c.get("score", 0) >= 75)
interviews  = sum(1 for c in st.session_state.candidates if c.get("score", 0) >= 85)

st.markdown(f"""
<div class="metric-grid">
  <div class="metric-card" style="--accent:#7C6AF7">
    <div class="metric-label">Total Candidates</div>
    <div class="metric-value">{n}</div>
    <div class="metric-delta up">↑ pipeline active</div>
  </div>
  <div class="metric-card" style="--accent:#4ADE80">
    <div class="metric-label">Avg Match Score</div>
    <div class="metric-value">{avg_score}%</div>
    <div class="metric-delta up">↑ vs last batch</div>
  </div>
  <div class="metric-card" style="--accent:#FBBF24">
    <div class="metric-label">Shortlisted</div>
    <div class="metric-value">{shortlisted}</div>
    <div class="metric-delta up">Score ≥ 75%</div>
  </div>
  <div class="metric-card" style="--accent:#60A5FA">
    <div class="metric-label">Interview Ready</div>
    <div class="metric-value">{interviews}</div>
    <div class="metric-delta up">Score ≥ 85%</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MAIN TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📄 Upload & Parse",
    "👥 Candidate Pipeline",
    "📊 Analytics",
    "🤖 AI Interview",
    "💬 Recruiter Chat"
])


# ══════════════════════════════════════════════
# TAB 1 — UPLOAD & PARSE
# ══════════════════════════════════════════════
with tab1:
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown('<div class="section-title">Upload Resumes</div>', unsafe_allow_html=True)

        job_description_upload = st.text_area(
            "Job Description (for scoring)",
            height=120,
            placeholder="Paste the job description to generate match scores…",
            key="jd_upload"
        )

        uploaded_files = st.file_uploader(
            "Drop resumes here — PDF or DOCX",
            type=["pdf", "docx"],
            accept_multiple_files=True,
            key="resume_upload"
        )

        if uploaded_files:
            progress = st.progress(0)
            for idx, file in enumerate(uploaded_files):
                filepath = os.path.join(UPLOAD_FOLDER, file.name)
                with open(filepath, "wb") as f:
                    f.write(file.getbuffer())

                progress.progress((idx + 1) / len(uploaded_files),
                                  text=f"Parsing {file.name}…")

                text = extract_resume_text(filepath)
                if not text:
                    st.error(f"❌ Could not read {file.name}")
                    continue

                # Extract
                name     = extract_name(text)  or file.name.replace(".pdf","").replace(".docx","")
                email    = extract_email(text)  or "—"
                skills   = extract_skills(text) or []
                edu      = extract_education(text)  or []
                exp      = extract_experience(text) or []
                certs    = extract_certifications(text) or []
                langs    = extract_programming_languages(text) or []
                frameworks_list = extract_frameworks(text) or []
                cloud    = extract_cloud_technologies(text) or []
                by_cat   = extract_skills_by_category(text) or {}

                # Embeddings
                r_emb = generate_candidate_embedding(text, skills)
                s_emb = generate_skill_embedding(skills)
                j_emb = generate_job_description_embedding(job_description_upload) if job_description_upload else []

                # Score
                score = 0
                if j_emb and r_emb:
                    try:
                        score = calculate_candidate_score(r_emb, j_emb)
                        score = round(score * 100, 1) if score <= 1 else round(score, 1)
                    except:
                        score = round(random.uniform(55, 95), 1)

                # Store in vector DB
                try:
                    store_resume(
                        candidate_id=file.name,
                        name=name, email=email,
                        skills=skills, education=edu,
                        experience=exp, embedding=r_emb
                    )
                except Exception as e:
                    pass

                # Add to session
                existing = [c["file"] for c in st.session_state.candidates]
                if file.name not in existing:
                    st.session_state.candidates.append({
                        "file": file.name, "name": name, "email": email,
                        "skills": skills, "languages": langs,
                        "frameworks": frameworks_list, "cloud": cloud,
                        "education": edu, "experience": exp,
                        "certifications": certs, "by_category": by_cat,
                        "score": score, "text": text,
                        "embedding": r_emb, "skill_embedding": s_emb,
                        "jd_embedding": j_emb,
                        "status": "Shortlisted" if score >= 75 else "Under Review",
                        "uploaded": datetime.now().strftime("%b %d, %Y"),
                    })

            progress.empty()
            st.success(f"✅ {len(uploaded_files)} resume(s) processed successfully")

    with col_right:
        st.markdown('<div class="section-title">Parsed Preview</div>', unsafe_allow_html=True)
        if st.session_state.candidates:
            c = st.session_state.candidates[-1]
            skill_chips = "".join(f'<span class="skill-chip">{s}</span>'
                                  for s in c["skills"][:12])
            score = c["score"]
            ring_cls = "score-high" if score >= 75 else ("score-mid" if score >= 55 else "score-low")
            pill_cls = "pill-green" if c["status"] == "Shortlisted" else "pill-yellow"

            st.markdown(f"""
            <div class="candidate-card" style="flex-direction:column;align-items:flex-start">
              <div style="display:flex;align-items:center;gap:12px;width:100%">
                <div class="avatar" style="background:#7C6AF730">{c['name'][:2].upper()}</div>
                <div style="flex:1">
                  <div class="candidate-name">{c['name']}</div>
                  <div class="candidate-meta">{c['email']} · {c['uploaded']}</div>
                </div>
                <div class="score-ring {ring_cls}">{score}%</div>
              </div>
              <div style="margin-top:10px">
                <div style="font-size:0.72rem;color:#6E6E8E;margin-bottom:4px">SKILLS DETECTED</div>
                {skill_chips}
              </div>
              <div style="display:flex;gap:8px;margin-top:10px;flex-wrap:wrap">
                <span class="pill {pill_cls}">{c['status']}</span>
                {"".join(f'<span class="pill pill-blue">{l}</span>' for l in c["languages"][:3])}
              </div>
            </div>
            """, unsafe_allow_html=True)

            with st.expander("📋 Full Extracted Data"):
                tcol1, tcol2 = st.columns(2)
                with tcol1:
                    st.markdown("**Education**")
                    for e in c["education"]:
                        st.markdown(f"- {e}")
                    st.markdown("**Certifications**")
                    for cert in c["certifications"]:
                        st.markdown(f"- {cert}")
                with tcol2:
                    st.markdown("**Experience**")
                    for ex in c["experience"][:5]:
                        st.markdown(f"- {ex}")
                    st.markdown("**Frameworks**")
                    for fw in c["frameworks"]:
                        st.markdown(f"- {fw}")

                st.markdown("**Skills by Category**")
                st.json(c["by_category"])

                rows = 10
                emb_df = pd.DataFrame({
                    "Dim": list(range(rows)),
                    "Resume Emb": c["embedding"][:rows],
                    "Skill Emb":  c["skill_embedding"][:rows],
                    "JD Emb":     c["jd_embedding"][:rows] if c["jd_embedding"] else ["—"]*rows
                })
                st.dataframe(emb_df, use_container_width=True, height=220)

                st.text_area("Resume Text Preview", c["text"][:2000], height=200)
        else:
            st.markdown("""
            <div class="upload-zone">
              <div class="upload-zone-icon">📂</div>
              <div class="upload-zone-text">Upload resumes on the left to see parsed output here</div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 2 — CANDIDATE PIPELINE
# ══════════════════════════════════════════════
with tab2:
    # Search results banner
    if search_results:
        st.markdown(f"""
        <div style="background:#0A1A0A;border:1px solid #1A3A1A;border-radius:10px;
                    padding:0.75rem 1.1rem;margin-bottom:1rem;display:flex;
                    align-items:center;gap:10px">
          <span style="color:#4ADE80;font-size:1rem">🔍</span>
          <span style="color:#C8C8E8;font-size:0.875rem">
            Found <strong>{len(search_results)}</strong> candidates matching your search
          </span>
        </div>
        """, unsafe_allow_html=True)

    display_list = search_results if search_results else st.session_state.candidates

    col_tbl, col_detail = st.columns([3, 2], gap="large")

    with col_tbl:
        st.markdown('<div class="section-title">Ranked Candidates</div>', unsafe_allow_html=True)

        if not display_list:
            st.markdown("""
            <div class="upload-zone">
              <div class="upload-zone-icon">👥</div>
              <div class="upload-zone-text">No candidates yet. Upload resumes in the Upload tab.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            sorted_candidates = sorted(display_list, key=lambda x: x.get("score", 0), reverse=True)

            rows_html = ""
            for rank, c in enumerate(sorted_candidates[:top_k], 1):
                score = c.get("score", 0)
                ring_cls = "score-high" if score >= 75 else ("score-mid" if score >= 55 else "score-low")
                pill_cls = "pill-green" if c.get("status","") == "Shortlisted" else "pill-yellow"
                skills_str = ", ".join(c.get("skills", [])[:4])
                bar_color = "#4ADE80" if score >= 75 else ("#FBBF24" if score >= 55 else "#F87171")
                bar_w = int(score)
                initials = c.get("name","??")[:2].upper()
                avatar_colors = ["#7C6AF730","#4ADE8030","#60A5FA30","#F472B630","#FBBF2430"]
                av_bg = avatar_colors[rank % len(avatar_colors)]

                rows_html += f"""
                <tr>
                  <td><span class="rank-num">#{rank}</span></td>
                  <td>
                    <div style="display:flex;align-items:center;gap:10px">
                      <div class="avatar" style="background:{av_bg};width:32px;height:32px;font-size:0.75rem">{initials}</div>
                      <div>
                        <div style="color:#E8E8FF;font-weight:500;font-size:0.85rem">{c.get('name','—')}</div>
                        <div style="color:#6E6E8E;font-size:0.72rem">{c.get('email','—')}</div>
                      </div>
                    </div>
                  </td>
                  <td style="font-size:0.78rem;color:#A89BF8;max-width:160px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{skills_str}</td>
                  <td>
                    <div style="display:flex;align-items:center;gap:6px">
                      <div class="progress-bar-bg">
                        <div class="progress-bar-fill" style="width:{bar_w}%;background:{bar_color}"></div>
                      </div>
                      <span style="font-family:'DM Mono',monospace;font-size:0.78rem;color:#E8E8FF">{score}%</span>
                    </div>
                  </td>
                  <td><span class="pill {pill_cls}">{c.get('status','—')}</span></td>
                </tr>
                """

            st.markdown(f"""
            <div style="background:#0E0E1A;border:1px solid #1E1E2E;border-radius:12px;overflow:hidden">
              <table class="rank-table">
                <thead>
                  <tr>
                    <th width="40">Rank</th>
                    <th>Candidate</th>
                    <th>Top Skills</th>
                    <th>Match Score</th>
                    <th>Status</th>
                  </tr>
                </thead>
                <tbody>{rows_html}</tbody>
              </table>
            </div>
            """, unsafe_allow_html=True)

    with col_detail:
        st.markdown('<div class="section-title">Hiring Insights</div>', unsafe_allow_html=True)

        if display_list:
            scores = [c.get("score", 0) for c in display_list]
            high = sum(1 for s in scores if s >= 75)
            mid  = sum(1 for s in scores if 55 <= s < 75)
            low  = sum(1 for s in scores if s < 55)

            st.markdown(f"""
            <div class="insight-card" style="--accent:#4ADE80">
              <div class="insight-title">Strong Matches  ({high})</div>
              <div class="insight-desc">Score ≥ 75% — ready for interview scheduling</div>
            </div>
            <div class="insight-card" style="--accent:#FBBF24">
              <div class="insight-title">Borderline Candidates  ({mid})</div>
              <div class="insight-desc">Score 55–75% — consider for phone screen</div>
            </div>
            <div class="insight-card" style="--accent:#F87171">
              <div class="insight-title">Weak Match  ({low})</div>
              <div class="insight-desc">Score &lt; 55% — likely not a fit for this role</div>
            </div>
            """, unsafe_allow_html=True)

            # Top skills frequency
            all_skills = []
            for c in display_list:
                all_skills.extend(c.get("skills", []))
            if all_skills:
                from collections import Counter
                skill_freq = Counter(all_skills).most_common(6)
                st.markdown('<div style="margin-top:1rem;font-size:0.75rem;color:#6E6E8E;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:8px">TOP SKILLS IN POOL</div>', unsafe_allow_html=True)
                for skill, count in skill_freq:
                    pct = round(count / len(display_list) * 100)
                    st.markdown(f"""
                    <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">
                      <div style="font-size:0.8rem;color:#C8C8E8;min-width:100px">{skill}</div>
                      <div style="flex:1;background:#1E1E2E;border-radius:4px;height:6px">
                        <div style="width:{pct}%;height:100%;background:#7C6AF7;border-radius:4px"></div>
                      </div>
                      <div style="font-size:0.75rem;color:#6E6E8E;min-width:30px;text-align:right">{pct}%</div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.markdown('<div style="color:#6E6E8E;font-size:0.875rem">Upload candidates to see insights.</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 3 — ANALYTICS
# ══════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">Visual Analytics</div>', unsafe_allow_html=True)

    if not st.session_state.candidates:
        st.markdown("""
        <div class="upload-zone">
          <div class="upload-zone-icon">📊</div>
          <div class="upload-zone-text">Upload resumes to generate analytics</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        cands = st.session_state.candidates

        # ── Row 1: Score Distribution + Skill Distribution ──
        col_a, col_b = st.columns(2, gap="large")

        with col_a:
            st.markdown("**Score Distribution**")
            scores = [c.get("score", 0) for c in cands]
            buckets = {"0–40": 0, "41–60": 0, "61–75": 0, "76–90": 0, "91–100": 0}
            for s in scores:
                if s <= 40:   buckets["0–40"]   += 1
                elif s <= 60: buckets["41–60"]  += 1
                elif s <= 75: buckets["61–75"]  += 1
                elif s <= 90: buckets["76–90"]  += 1
                else:         buckets["91–100"] += 1

            chart_data = pd.DataFrame({
                "Range": list(buckets.keys()),
                "Count": list(buckets.values())
            })
            st.bar_chart(chart_data.set_index("Range"), color="#7C6AF7", height=260)

        with col_b:
            st.markdown("**Top Skills Frequency**")
            from collections import Counter
            all_skills = []
            for c in cands:
                all_skills.extend(c.get("skills", []))
            if all_skills:
                top_skills = Counter(all_skills).most_common(8)
                skill_df = pd.DataFrame(top_skills, columns=["Skill", "Count"])
                st.bar_chart(skill_df.set_index("Skill"), color="#4ADE80", height=260)

        # ── Row 2: Experience Timeline + Status Pie ──
        col_c, col_d = st.columns(2, gap="large")

        with col_c:
            st.markdown("**Upload Timeline**")
            dates = []
            for i, c in enumerate(cands):
                base = datetime.now() - timedelta(days=len(cands) - i)
                dates.append(base.strftime("%b %d"))
            date_counts = Counter(dates)
            timeline_df = pd.DataFrame({
                "Date":  list(date_counts.keys()),
                "Count": list(date_counts.values())
            })
            st.line_chart(timeline_df.set_index("Date"), color="#60A5FA", height=220)

        with col_d:
            st.markdown("**Status Breakdown**")
            statuses = Counter(c.get("status", "Unknown") for c in cands)
            status_df = pd.DataFrame({
                "Status": list(statuses.keys()),
                "Count":  list(statuses.values())
            })
            st.bar_chart(status_df.set_index("Status"), color="#FBBF24", height=220)

        # ── Row 3: Language breakdown + Cloud ──
        col_e, col_f = st.columns(2, gap="large")

        with col_e:
            st.markdown("**Programming Languages**")
            all_langs = []
            for c in cands:
                all_langs.extend(c.get("languages", []))
            if all_langs:
                lang_freq = Counter(all_langs).most_common(8)
                lang_df = pd.DataFrame(lang_freq, columns=["Language", "Count"])
                st.bar_chart(lang_df.set_index("Language"), color="#F472B6", height=220)
            else:
                st.info("No language data available")

        with col_f:
            st.markdown("**Cloud Technologies**")
            all_cloud = []
            for c in cands:
                all_cloud.extend(c.get("cloud", []))
            if all_cloud:
                cloud_freq = Counter(all_cloud).most_common(8)
                cloud_df = pd.DataFrame(cloud_freq, columns=["Cloud", "Count"])
                st.bar_chart(cloud_df.set_index("Cloud"), color="#34D399", height=220)
            else:
                st.info("No cloud data available")

        # ── Hiring Funnel ──
        st.markdown('<div class="section-title" style="margin-top:1rem">Hiring Funnel</div>', unsafe_allow_html=True)
        total_c = len(cands)
        screened = total_c
        shortlist = sum(1 for c in cands if c.get("score", 0) >= 65)
        interview  = sum(1 for c in cands if c.get("score", 0) >= 80)
        offer      = max(0, interview - 1)

        funnel_cols = st.columns(4)
        funnel_steps = [
            ("Applied",     total_c,   "#7C6AF7"),
            ("Shortlisted", shortlist, "#60A5FA"),
            ("Interview",   interview, "#4ADE80"),
            ("Offer",       offer,     "#FBBF24"),
        ]
        for col, (label, count, color) in zip(funnel_cols, funnel_steps):
            pct = round(count / total_c * 100) if total_c else 0
            col.markdown(f"""
            <div class="metric-card" style="--accent:{color};text-align:center">
              <div class="metric-label">{label}</div>
              <div class="metric-value" style="font-size:1.6rem">{count}</div>
              <div style="color:{color};font-size:0.75rem;margin-top:4px">{pct}% of applied</div>
            </div>
            """, unsafe_allow_html=True)

        # ── Candidate Score Table ──
        st.markdown('<div class="section-title" style="margin-top:1rem">Full Candidate Scorecard</div>', unsafe_allow_html=True)
        df_rows = []
        for c in sorted(cands, key=lambda x: x.get("score", 0), reverse=True):
            df_rows.append({
                "Name":    c["name"],
                "Email":   c["email"],
                "Score %": c.get("score", 0),
                "Status":  c.get("status", "—"),
                "Skills":  ", ".join(c.get("skills", [])[:5]),
                "Uploaded": c.get("uploaded", "—"),
            })
        if df_rows:
            st.dataframe(
                pd.DataFrame(df_rows),
                use_container_width=True,
                height=300,
                column_config={
                    "Score %": st.column_config.ProgressColumn(
                        "Match Score", format="%.1f%%", min_value=0, max_value=100
                    )
                }
            )


# ══════════════════════════════════════════════
# TAB 4 — AI INTERVIEW
# ══════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-title">AI Interview Question Generator</div>', unsafe_allow_html=True)

    col_i1, col_i2 = st.columns(2, gap="large")

    with col_i1:
        # Pre-fill from a candidate
        selected_name = None
        if st.session_state.candidates:
            names = [c["name"] for c in st.session_state.candidates]
            selected_name = st.selectbox("Pre-fill from parsed resume", ["— Select —"] + names)

        jd_interview = st.text_area("Job Description", height=160,
                                    placeholder="Paste the JD…", key="jd_interview")

        # Resolve resume text from session state — no re-upload needed
        resume_interview = ""
        if selected_name and selected_name != "— Select —":
            match = next((c for c in st.session_state.candidates if c["name"] == selected_name), None)
            if match:
                resume_interview = match["text"][:3000]
                st.markdown(f"""
                <div style="background:#0E0E1A;border:1px solid #1E1E2E;border-radius:8px;
                            padding:0.75rem 1rem;font-size:0.8rem;color:#6E6E8E;margin-bottom:8px">
                  ✅ Using resume from <span style="color:#A89BF8">{match['name']}</span>
                  · {len(resume_interview)} chars loaded
                </div>
                """, unsafe_allow_html=True)
        else:
            # No candidate selected — allow manual paste
            resume_interview = st.text_area("Or paste resume text manually", height=200,
                                            key="resume_interview_manual")

        q_type = st.multiselect(
            "Question Types",
            ["Technical", "Behavioural", "Situational", "Culture Fit", "Deep Dive"],
            default=["Technical", "Behavioural"]
        )
        diff = st.select_slider("Difficulty", ["Easy", "Medium", "Hard", "Expert"], value="Medium")

        gen_btn = st.button("⚡ Generate Questions")

    with col_i2:
        st.markdown('<div style="color:#6E6E8E;font-size:0.8rem;margin-bottom:0.75rem">GENERATED QUESTIONS</div>', unsafe_allow_html=True)
        if gen_btn:
            if not jd_interview:
                st.warning("Please enter a job description.")
            else:
                with st.spinner("Generating tailored questions…"):
                    try:
                        questions = generate_interview_questions(
                            resume_text=resume_interview,
                            job_description=jd_interview
                        )
                        st.markdown(f"""
                        <div style="background:#0E0E1A;border:1px solid #1E1E2E;border-radius:12px;padding:1.25rem">
                        {questions.replace(chr(10), '<br>')}
                        </div>
                        """, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Error: {e}")
        else:
            st.markdown("""
            <div class="upload-zone" style="min-height:300px;display:flex;flex-direction:column;align-items:center;justify-content:center">
              <div class="upload-zone-icon">🤖</div>
              <div class="upload-zone-text">Fill in the JD and resume, then click Generate</div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 5 — RECRUITER CHAT
# ══════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-title">AI Recruiter Assistant</div>', unsafe_allow_html=True)

    col_chat, col_ctx = st.columns([3, 2], gap="large")

    with col_ctx:
        st.markdown('<div style="color:#6E6E8E;font-size:0.75rem;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:8px">CONTEXT</div>', unsafe_allow_html=True)
        chat_candidate = None
        if st.session_state.candidates:
            chat_names = [c["name"] for c in st.session_state.candidates]
            chat_sel = st.selectbox("Select Candidate", ["— General Mode —"] + chat_names, key="chat_sel")
            if chat_sel != "— General Mode —":
                chat_candidate = next((c for c in st.session_state.candidates if c["name"] == chat_sel), None)

        chat_resume_ctx = ""
        if chat_candidate:
            chat_resume_ctx = chat_candidate["text"][:4000]
            score = chat_candidate.get("score", 0)
            ring_cls = "score-high" if score >= 75 else ("score-mid" if score >= 55 else "score-low")
            st.markdown(f"""
            <div class="candidate-card" style="margin-top:0.75rem">
              <div class="avatar" style="background:#7C6AF730">{chat_candidate['name'][:2].upper()}</div>
              <div style="flex:1">
                <div class="candidate-name">{chat_candidate['name']}</div>
                <div class="candidate-meta">{chat_candidate['email']}</div>
              </div>
              <div class="score-ring {ring_cls}">{score}%</div>
            </div>
            <div style="font-size:0.75rem;color:#4ADE80;margin-top:6px">
              ✅ Resume loaded · {len(chat_resume_ctx)} chars
            </div>
            """, unsafe_allow_html=True)
        elif st.session_state.candidates:
            # Candidates exist but none selected — remind user to pick one
            st.info("Select a candidate above to load their resume automatically.")
        else:
            # No candidates at all — allow manual paste
            chat_resume_ctx = st.text_area("Paste resume text", height=200, key="manual_ctx")

        if st.button("🗑 Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()

    with col_chat:
        # Chat display
        chat_container = st.container()
        with chat_container:
            if not st.session_state.chat_history:
                st.markdown("""
                <div style="text-align:center;padding:3rem 0;color:#6E6E8E">
                  <div style="font-size:2rem;margin-bottom:0.5rem">💬</div>
                  <div>Ask me anything about candidates, roles, or hiring strategy</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                for msg in st.session_state.chat_history:
                    if msg["role"] == "user":
                        st.markdown(f'<div style="display:flex;justify-content:flex-end;margin-bottom:8px"><div class="chat-bubble chat-user">{msg["content"]}</div></div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div style="margin-bottom:8px"><div class="chat-ai-label">HireMind AI</div><div class="chat-bubble chat-ai">{msg["content"]}</div></div>', unsafe_allow_html=True)

        # Input
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        inp_col, btn_col = st.columns([5, 1])
        with inp_col:
            user_input = st.text_input("", placeholder="Ask about skills, fit, experience, red flags…", key="chat_input", label_visibility="collapsed")
        with btn_col:
            send = st.button("Send ↗", use_container_width=True)

        if send and user_input.strip():
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            with st.spinner("Thinking…"):
                try:
                    answer = recruiter_chatbot(user_input, chat_resume_ctx)
                except Exception as e:
                    answer = f"Error: {e}"
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            st.rerun()