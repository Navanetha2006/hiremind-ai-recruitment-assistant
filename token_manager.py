"""
STEP 26 — TOKEN MANAGEMENT
Prevent oversized prompts · optimize AI context · limit unnecessary tokens
"""

import re
from typing import List, Dict, Tuple, Optional

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

MAX_PROMPT_TOKENS   = 8_192   # hard ceiling for claude-3 / GPT-4 calls
SYSTEM_BUDGET       = 512     # reserved for system prompt
RESPONSE_BUDGET     = 1_800   # reserved for model response
HISTORY_BUDGET      = 1_400   # sliding window for chat history
CONTEXT_BUDGET      = MAX_PROMPT_TOKENS - SYSTEM_BUDGET - RESPONSE_BUDGET - HISTORY_BUDGET
# → 4,480 tokens for retrieved resume/JD context


# ─────────────────────────────────────────────
# 26.1 TOKEN COUNTER (no tiktoken dependency)
# ─────────────────────────────────────────────

def count_tokens(text: str) -> int:
    """
    Fast approximation: ~4 chars per token (OpenAI/Anthropic average).
    Good enough for budget enforcement without an external library.
    """
    return max(1, len(text) // 4)


# ─────────────────────────────────────────────
# 26.2 CONTEXT OPTIMIZER
# ─────────────────────────────────────────────

def _deduplicate_chunks(chunks: List[str]) -> List[str]:
    """Remove near-duplicate sentences across chunks."""
    seen, unique = set(), []
    for chunk in chunks:
        key = re.sub(r"\s+", " ", chunk.strip().lower())[:120]
        if key not in seen:
            seen.add(key)
            unique.append(chunk)
    return unique

def _sentence_summarize(text: str, max_sentences: int = 6) -> str:
    """Keep only the first N sentences as a lightweight extractive summary."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return " ".join(sentences[:max_sentences])

def optimize_context(
    resume_text: str,
    job_description: str,
    extra_chunks: Optional[List[str]] = None,
) -> Tuple[str, Dict[str, int]]:
    """
    Fit resume + JD into CONTEXT_BUDGET tokens.

    Strategy:
    1. Deduplicate any extra chunks.
    2. Allocate 55% of budget to JD, 45% to resume.
    3. Truncate at sentence boundary if over budget.

    Returns (context_string, token_counts_dict).
    """
    jd_budget     = int(CONTEXT_BUDGET * 0.45)
    resume_budget = int(CONTEXT_BUDGET * 0.45)
    chunk_budget  = CONTEXT_BUDGET - jd_budget - resume_budget

    # Trim JD
    jd_tokens = count_tokens(job_description)
    if jd_tokens > jd_budget:
        chars = jd_budget * 4
        job_description = job_description[:chars].rsplit(".", 1)[0] + "."

    # Trim resume
    resume_tokens = count_tokens(resume_text)
    if resume_tokens > resume_budget:
        chars = resume_budget * 4
        resume_text = resume_text[:chars].rsplit(".", 1)[0] + "."
    elif resume_tokens > resume_budget * 0.8:
        resume_text = _sentence_summarize(resume_text, max_sentences=10)

    # Extra chunks (retrieved docs, prior context)
    chunk_block = ""
    if extra_chunks:
        deduped = _deduplicate_chunks(extra_chunks)
        used = 0
        kept = []
        for chunk in deduped:
            t = count_tokens(chunk)
            if used + t > chunk_budget:
                break
            kept.append(chunk)
            used += t
        chunk_block = "\n\n".join(kept)

    context = f"[JOB DESCRIPTION]\n{job_description}\n\n[CANDIDATE RESUME]\n{resume_text}"
    if chunk_block:
        context += f"\n\n[ADDITIONAL CONTEXT]\n{chunk_block}"

    token_counts = {
        "job_description": count_tokens(job_description),
        "resume":          count_tokens(resume_text),
        "extra_chunks":    count_tokens(chunk_block),
        "total_context":   count_tokens(context),
        "budget":          CONTEXT_BUDGET,
        "headroom":        CONTEXT_BUDGET - count_tokens(context),
    }
    return context, token_counts


# ─────────────────────────────────────────────
# 26.3 HISTORY COMPRESSOR (chat sliding window)
# ─────────────────────────────────────────────

def compress_history(
    history: List[Dict[str, str]],
    budget: int = HISTORY_BUDGET,
) -> List[Dict[str, str]]:
    """
    Keep the most recent messages that fit within `budget` tokens.
    Always keeps at least the last 2 turns (1 user + 1 assistant).
    """
    if not history:
        return []

    kept, total = [], 0
    for msg in reversed(history):
        t = count_tokens(msg.get("content", ""))
        if total + t > budget and len(kept) >= 2:
            break
        kept.append(msg)
        total += t

    return list(reversed(kept))


# ─────────────────────────────────────────────
# 26.4 PROMPT BUILDER — enforces hard ceiling
# ─────────────────────────────────────────────

def build_prompt(
    system_prompt: str,
    user_message: str,
    resume_text: str = "",
    job_description: str = "",
    chat_history: Optional[List[Dict[str, str]]] = None,
    extra_chunks: Optional[List[str]] = None,
) -> Tuple[str, str, List[Dict[str, str]], Dict[str, int]]:
    """
    Assemble a token-safe prompt tuple: (system, context, messages, stats).

    Usage in recruiter_chatbot.py / question_generator.py:
        system, context, messages, stats = build_prompt(...)
        # pass system + context + messages to your LLM call
    """
    # 1. Trim system prompt
    sys_tokens = count_tokens(system_prompt)
    if sys_tokens > SYSTEM_BUDGET:
        system_prompt = system_prompt[: SYSTEM_BUDGET * 4]

    # 2. Build optimized context block
    context, ctx_stats = optimize_context(resume_text, job_description, extra_chunks)

    # 3. Compress history
    history = compress_history(chat_history or [], budget=HISTORY_BUDGET)

    # 4. Check user message fits
    user_tokens = count_tokens(user_message)
    if user_tokens > 512:
        user_message = user_message[: 512 * 4].rsplit(".", 1)[0] + "."

    # 5. Final budget check
    total = (
        count_tokens(system_prompt)
        + ctx_stats["total_context"]
        + sum(count_tokens(m["content"]) for m in history)
        + count_tokens(user_message)
        + RESPONSE_BUDGET
    )

    stats = {
        **ctx_stats,
        "system":       count_tokens(system_prompt),
        "history_msgs": len(history),
        "user_message": user_tokens,
        "response_rsv": RESPONSE_BUDGET,
        "grand_total":  total,
        "over_budget":  max(0, total - MAX_PROMPT_TOKENS),
    }

    return system_prompt, context, history, stats


# ─────────────────────────────────────────────
# 26.5 QUICK USAGE REPORT (for debug/logging)
# ─────────────────────────────────────────────

def token_report(stats: Dict[str, int]) -> str:
    over = stats.get("over_budget", 0)
    status = "✅ within budget" if over == 0 else f"⚠️ over by {over} tokens"
    return (
        f"Token Report | {status}\n"
        f"  System:   {stats.get('system', 0):>5} / {SYSTEM_BUDGET}\n"
        f"  Context:  {stats.get('total_context', 0):>5} / {CONTEXT_BUDGET}\n"
        f"  History:  {sum(0 for _ in range(stats.get('history_msgs', 0))):>5} msgs\n"
        f"  User msg: {stats.get('user_message', 0):>5}\n"
        f"  Reserved: {RESPONSE_BUDGET:>5} (response)\n"
        f"  ─────────────────────\n"
        f"  Total:    {stats.get('grand_total', 0):>5} / {MAX_PROMPT_TOKENS}"
    )