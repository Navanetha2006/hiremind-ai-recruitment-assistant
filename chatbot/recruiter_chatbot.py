import os
from dotenv import load_dotenv
from openai import OpenAI
from token_manager import build_prompt

load_dotenv()

def recruiter_chatbot(question, resume_text, chat_history=[]):

    # Check if resume exists
    if not resume_text or len(str(resume_text).strip()) == 0:
        return "No resume data found. Please upload a resume first."

    try:
        client = OpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1"
        )

        system = """You are an expert AI Recruiter Assistant.

Instructions:
- Use ONLY information from the resume.
- Summarize candidate experience and skills when asked.
- Recommend suitable job roles.
- Identify strengths and weaknesses.
- Generate interview questions if requested.
- Explain why the candidate is or isn't a good fit.
- Give concise recruiter-friendly answers."""

        # Build token-safe prompt — trims resume + history if too long
        system, context, messages, stats = build_prompt(
            system_prompt=system,
            user_message=question,
            resume_text=resume_text,
            chat_history=chat_history,
        )

        prompt = f"""{context}

Recruiter Question:
{question}

Answer:"""

        response = client.chat.completions.create(
            model="meta-llama/llama-3.3-70b-instruct",
            messages=[
                {"role": "system", "content": system},
                *messages,                               # compressed chat history
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=1000
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"Error: {str(e)}"