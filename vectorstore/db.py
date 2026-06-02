import chromadb

# -----------------------------
# INIT CHROMA DB
# -----------------------------
chroma_client = chromadb.Client()  # in-memory, no disk write

collection = chroma_client.get_or_create_collection(
    name="resumes"
)

# -----------------------------
# STORE RESUME
# -----------------------------
def store_resume(candidate_id, name, email, skills, education, experience, embedding):
    collection.add(
        ids=[candidate_id],
        embeddings=[embedding],
        metadatas=[{
            "name": name,
            "email": email,
            "skills": skills,
            "education": education,
            "experience": experience
        }]
    )

# -----------------------------
# SEMANTIC SEARCH
# -----------------------------
def search_candidates(query_embedding, top_k=5):
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["metadatas", "distances"]
    )
    return results