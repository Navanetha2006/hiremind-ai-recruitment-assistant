import os
from dotenv import load_dotenv
from openai import OpenAI
from token_manager import build_prompt

load_dotenv()

def generate_interview_questions(resume_text, job_description):

    client = OpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1"
    )

    system = "You are an expert technical interviewer."

    # Build token-safe prompt — trims resume + JD if too long
    system, context, _, stats = build_prompt(
        system_prompt=system,
        user_message="Generate interview questions.",
        resume_text=resume_text,
        job_description=job_description,
    )

    prompt = f"""{context}

Generate:

1. 10 Technical Interview Questions relevant to the job description.
2. 5 HR Interview Questions.
3. 5 Scenario-Based Questions.
4. 1 Technical Assessment Task.

Questions should evaluate whether the candidate matches the job requirements.

Format properly with headings.
"""

    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-3.3-70b-instruct",
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1500
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"Error: {str(e)}"