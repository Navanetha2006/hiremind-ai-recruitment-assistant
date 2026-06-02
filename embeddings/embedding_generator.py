from sentence_transformers import SentenceTransformer
import numpy as np
from search_optimizer import get_cached_embedding

# Load model once when application starts
model = None

def get_model():
    global model
    if model is None:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return model


def generate_embedding(text):
    """
    Generate embedding for a single text
    """
    if text is None:
        return []

    text = str(text).strip()

    if text == "":
        return []

    embedding = get_model().encode(text, convert_to_numpy=True)

    return embedding.tolist()


def generate_resume_embedding(resume_text):
    """
    Generate embedding for resume text
    """
    return generate_embedding(resume_text)


def generate_skill_embedding(skills):
    """
    Generate embedding for skills
    """
    if isinstance(skills, list):
        skills = " ".join(map(str, skills))

    return generate_embedding(skills)


def _raw_generate_job_description_embedding(text):  # ← fixed: was using wrong variable name
    """
    Generate embedding for job description
    """
    return generate_embedding(text)  # ← fixed: was `job_description`, now `text`


def generate_job_description_embedding(text):
    """
    Cached version — same text is never re-embedded twice
    """
    return get_cached_embedding(text, _raw_generate_job_description_embedding)


def generate_candidate_embedding(resume_text, skills=None):
    """
    Create combined embedding using resume text + extracted skills
    """
    combined_text = str(resume_text)

    if skills:
        if isinstance(skills, list):
            skills = " ".join(map(str, skills))
        combined_text += " " + skills

    return generate_embedding(combined_text)


def generate_embeddings(texts):
    """
    Generate embeddings for multiple texts
    """
    if not texts:
        return []

    embeddings = model.encode(
        texts,
        convert_to_numpy=True
    )

    return embeddings.tolist()


def get_embedding_dimension():
    """
    Returns embedding dimension
    """
    return model.get_sentence_embedding_dimension()


def cosine_similarity(vec1, vec2):
    """
    Calculate cosine similarity
    """
    if not vec1 or not vec2:
        return 0

    vec1 = np.array(vec1)
    vec2 = np.array(vec2)

    numerator = np.dot(vec1, vec2)
    denominator = np.linalg.norm(vec1) * np.linalg.norm(vec2)

    if denominator == 0:
        return 0

    return float(numerator / denominator)


def calculate_match_score(resume_embedding, job_embedding):
    """
    Returns percentage match score
    """
    similarity = cosine_similarity(resume_embedding, job_embedding)
    return round(similarity * 100, 2)


def compare_candidates(candidate_embeddings, job_embedding):
    """
    Rank candidates against a job description
    """
    rankings = []

    for index, embedding in enumerate(candidate_embeddings):
        score = calculate_match_score(embedding, job_embedding)
        rankings.append({
            "candidate_index": index,
            "score": score
        })

    rankings = sorted(rankings, key=lambda x: x["score"], reverse=True)

    return rankings