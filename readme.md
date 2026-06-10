# 🧠 HireMind AI — Recruitment Assistant Platform

An AI-powered recruitment assistant that helps recruiters upload resumes, extract skills, rank candidates, generate interview questions, and interact with an AI chatbot — all from a single Streamlit dashboard.

---

## ✨ Features

- **Multi-Resume Upload** — Upload PDF and DOCX resumes in bulk
- **Resume Parsing** — Automatically extracts name, email, education, experience, and certifications
- **Skill Extraction** — Detects programming languages, frameworks, tools, and cloud technologies using spaCy NLP
- **Semantic Embeddings** — Generates vector embeddings using `sentence-transformers/all-MiniLM-L6-v2`
- **Vector Search** — Stores and retrieves candidate profiles using ChromaDB
- **AI Candidate Ranking** — Scores candidates based on skill match, experience, education, and semantic similarity
- **Job Description Matching** — Matches resumes to a job description semantically
- **AI Interview Question Generator** — Generates tailored Technical, Behavioural, Situational, and Culture Fit questions via LLM
- **AI Recruiter Chatbot** — Chat with an AI assistant about candidates, skills, and hiring decisions
- **Visual Analytics Dashboard** — Hiring funnel, skill distribution, candidate scorecard, and experience analytics

---

## 🏗️ AI Architecture

```
Resumes (PDF/DOCX)
       │
       ▼
Resume Parser (PyPDF / pdfplumber / python-docx)
       │
       ├──► NLP Pipeline (spaCy) ──► Skill / Entity Extraction
       │
       ▼
Embedding Model (sentence-transformers/all-MiniLM-L6-v2)
       │
       ▼
Vector Store (ChromaDB) ──► Semantic Search
       │
       ▼
Candidate Ranker ──► Scoring & Ranking
       │
       ├──► Interview Question Generator (OpenRouter LLM)
       │
       └──► Recruiter Chatbot (OpenRouter LLM)
```

---

## 🗂️ Project Structure

```
hiremind-ai/
│
├── app/
├── chatbot/
│   └── recruiter_chatbot.py       # AI recruiter assistant
├── embeddings/
│   └── embedding_generator.py     # Sentence-transformer embeddings
├── interview/
│   └── question_generator.py      # LLM-powered interview questions
├── parser/
│   ├── resume_parser.py           # PDF/DOCX text extraction
│   ├── info_extractor.py          # Name & email extraction
│   ├── skill_extractor.py         # Skill / tech / language detection
│   ├── education_extractor.py     # Education parsing
│   ├── experience_extractor.py    # Experience parsing
│   └── certification_extractor.py # Certification detection
├── ranking/
│   └── candidate_ranker.py        # Scoring & ranking logic
├── vectorstore/
│   └── db.py                      # ChromaDB store & search
├── resumes/                       # Sample resumes
├── screenshots/                   # Project screenshots
├── uploads/                       # Uploaded resume files
├── main.py                        # Streamlit application entry point
├── search_optimizer.py            # Embedding cache & optimised ranking
├── requirements.txt
├── .env.example
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/hiremind-ai-recruitment-assistant.git
cd hiremind-ai-recruitment-assistant
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies

```bash
pip install langchain transformers sentence-transformers chromadb streamlit spacy \
            python-dotenv pypdf pdfplumber python-docx torch scikit-learn
python -m spacy download en_core_web_sm
```

Or use the requirements file:

```bash
pip install -r requirements.txt
```

---

## 🔑 API Configuration

1. Create a free account at [OpenRouter](https://openrouter.ai/) and generate an API key.
2. Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

3. Add your key to `.env`:

```
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

---

## 🚀 Running the App

```bash
streamlit run main.py
```

The app will open at `http://localhost:8501`.

---

## 📊 Dashboard Tabs

| Tab | Description |
|-----|-------------|
| **Upload** | Upload and parse single or multiple resumes |
| **Candidates** | View parsed candidate profiles with skill chips and scores |
| **Analytics** | Hiring funnel, skill distribution, experience insights, full scorecard |
| **AI Interview** | Generate interview questions tailored to a candidate and job description |
| **Recruiter Chat** | AI chatbot for querying candidate profiles and hiring advice |

---

## 🛠️ Tech Stack

| Layer | Tool |
|-------|------|
| Frontend | Streamlit |
| NLP | spaCy (`en_core_web_sm`) |
| Embeddings | Sentence Transformers (`all-MiniLM-L6-v2`) |
| Vector DB | ChromaDB |
| LLM API | OpenRouter (free tier) |
| PDF Parsing | PyPDF, pdfplumber |
| DOCX Parsing | python-docx |
| AI Orchestration | LangChain |
| ML Utilities | scikit-learn, PyTorch |

---

## ☁️ Deployment

### Option 1 — HuggingFace Spaces

1. Create a new Space at [huggingface.co/spaces](https://huggingface.co/spaces)
2. Set SDK to **Streamlit**
3. Push the repository and add `OPENROUTER_API_KEY` as a Space secret

### Option 2 — Render

1. Create a new **Web Service** at [render.com](https://render.com)
2. Connect your GitHub repository
3. Set the start command to `streamlit run main.py --server.port $PORT`
4. Add `OPENROUTER_API_KEY` as an environment variable

---

## 📸 Screenshots

Screenshots are stored in the `/screenshots` folder covering:

- Python & dependency installation
- API key configuration
- Resume parsing output
- Skill extraction results
- Embedding generation
- Semantic search results
- Candidate ranking table
- Interview question generator
- AI recruiter chatbot
- Live deployed application

---

## 🔮 Future Improvements

- OCR support for scanned/image-based PDFs
- Multi-language resume parsing
- AI bias detection in candidate scoring
- Voice interview evaluation
- Candidate comparison engine (side-by-side)
- AI-generated hiring reports (PDF export)
- Local LLM integration (Ollama / LM Studio)
- Emotion/sentiment analysis from interview responses

---

## 📄 License

This project was built as part of an AI engineering internship task. See `LICENSE` for details.

---

## 🙌 Acknowledgements

- [OpenRouter](https://openrouter.ai/) — Free LLM API access
- [HuggingFace](https://huggingface.co/) — Sentence Transformers & model hosting
- [ChromaDB](https://www.trychroma.com/) — Open-source vector database
- [Streamlit](https://streamlit.io/) — Rapid AI dashboard framework
- [spaCy](https://spacy.io/) — Industrial-strength NLP