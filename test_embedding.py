from embeddings.embedding_generator import *

resume = """
Python Developer with AWS,
Docker and Machine Learning.
"""

skills = [
    "Python",
    "AWS",
    "Machine Learning"
]

job_description = """
Looking for a Python Developer
with AWS experience.
"""

resume_embedding = generate_resume_embedding(
    resume
)

skill_embedding = generate_skill_embedding(
    skills
)

job_embedding = generate_job_description_embedding(
    job_description
)

print(
    "Embedding Dimension:",
    get_embedding_dimension()
)

print(
    "Resume Length:",
    len(resume_embedding)
)

print(
    "Skill Length:",
    len(skill_embedding)
)

print(
    "Job Description Length:",
    len(job_embedding)
)

print(
    "Match Score:",
    calculate_match_score(
        resume_embedding,
        job_embedding
    )
)