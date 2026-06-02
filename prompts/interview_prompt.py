def build_interview_prompt(name, skills, experience, education, job_description):

    return f"""
You are an expert AI technical interviewer.

Generate structured interview questions for the candidate.

CANDIDATE DETAILS:
Name: {name}
Skills: {skills}
Experience: {experience}
Education: {education}

JOB DESCRIPTION:
{job_description}

TASK:
Generate interview questions in the following format:

1. Technical Interview Questions (5 questions)
2. HR / Behavioral Questions (5 questions)
3. Practical / Coding Assessment Questions (5 questions)

Rules:
- Make questions specific to the candidate skills
- Make questions relevant to job role
- Mix easy, medium, hard difficulty
- Be professional and structured
"""