def calculate_candidate_score(
    skill_match,
    experience_score,
    education_score,
    similarity_score
):
    """
    Final AI Candidate Ranking Score

    Weights:
    Skill Match        -> 35%
    Experience         -> 25%
    Education          -> 15%
    Semantic Similarity -> 25%
    """

    final_score = (
        skill_match * 0.35 +
        experience_score * 0.25 +
        education_score * 0.15 +
        similarity_score * 0.25
    )

    return round(final_score, 2)