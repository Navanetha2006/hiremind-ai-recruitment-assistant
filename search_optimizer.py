"""
STEP 25 — OPTIMIZE SEARCH PERFORMANCE
Embedding caching · faster retrieval · optimized ranking
"""

import hashlib
import time
import numpy as np
from functools import lru_cache
from typing import List, Dict, Any, Optional

# ─────────────────────────────────────────────
# 25.1 EMBEDDING CACHE
# ─────────────────────────────────────────────

_embedding_cache: Dict[str, List[float]] = {}

def _cache_key(text: str) -> str:
    return hashlib.md5(text.strip().lower().encode()).hexdigest()

def get_cached_embedding(text: str, generator_fn) -> List[float]:
    """Return cached embedding or generate + cache a new one."""
    key = _cache_key(text)
    if key in _embedding_cache:
        return _embedding_cache[key]
    embedding = generator_fn(text)
    _embedding_cache[key] = embedding
    return embedding

def cache_stats() -> Dict[str, int]:
    return {"cached_embeddings": len(_embedding_cache)}

def clear_embedding_cache():
    _embedding_cache.clear()


# ─────────────────────────────────────────────
# 25.2 FASTER RETRIEVAL — HNSW-style pre-filter
# ─────────────────────────────────────────────

def cosine_similarity(a: List[float], b: List[float]) -> float:
    a, b = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))

def fast_search(
    query_embedding: List[float],
    candidates: List[Dict[str, Any]],
    top_k: int = 10,
    pre_filter_k: int = 50,
) -> List[Dict[str, Any]]:
    """
    Two-stage retrieval:
    1. Cheap dot-product pre-filter to reduce candidate pool (pre_filter_k).
    2. Exact cosine re-score on the shortlist.
    """
    q = np.array(query_embedding, dtype=np.float32)
    q_norm = q / (np.linalg.norm(q) + 1e-9)

    # Stage 1 — approximate dot product (fast numpy batch)
    embeddings = np.array(
        [c["embedding"] for c in candidates], dtype=np.float32
    )  # shape (N, D)
    dot_scores = embeddings @ q_norm          # shape (N,)
    pre_idx = np.argsort(dot_scores)[::-1][:pre_filter_k]

    # Stage 2 — exact cosine on shortlist
    shortlist = [candidates[i] for i in pre_idx]
    scored = []
    for c in shortlist:
        sim = cosine_similarity(query_embedding, c["embedding"])
        scored.append({**c, "_similarity": sim})

    scored.sort(key=lambda x: x["_similarity"], reverse=True)
    return scored[:top_k]


# ─────────────────────────────────────────────
# 25.3 OPTIMIZED RANKING
# ─────────────────────────────────────────────

def _bm25_score(query_terms: List[str], candidate_text: str) -> float:
    """Lightweight BM25-inspired keyword overlap score (k1=1.5, b=0.75)."""
    k1, b, avg_dl = 1.5, 0.75, 500
    words = candidate_text.lower().split()
    dl = len(words)
    score = 0.0
    for term in query_terms:
        tf = words.count(term.lower())
        idf = 1.0  # simplified — uniform IDF
        score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avg_dl))
    return score

def _recency_decay(uploaded_date: Optional[str], half_life_days: int = 60) -> float:
    """More recent uploads score slightly higher (max boost +5%)."""
    if not uploaded_date:
        return 1.0
    try:
        from datetime import datetime
        delta = (datetime.now() - datetime.strptime(uploaded_date, "%b %d, %Y")).days
        return 1.0 + 0.05 * (0.5 ** (delta / half_life_days))
    except Exception:
        return 1.0

def _mmr_diverse(
    scored: List[Dict[str, Any]],
    lambda_param: float = 0.7,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Maximal Marginal Relevance — balance relevance vs diversity.
    Avoids returning near-duplicate candidates.
    """
    if not scored:
        return []
    selected, remaining = [scored[0]], scored[1:]
    while remaining and len(selected) < top_k:
        best, best_score = None, -1.0
        for c in remaining:
            rel = c.get("_similarity", 0)
            redundancy = max(
                cosine_similarity(c["embedding"], s["embedding"])
                for s in selected
            )
            mmr = lambda_param * rel - (1 - lambda_param) * redundancy
            if mmr > best_score:
                best, best_score = c, mmr
        if best:
            selected.append(best)
            remaining.remove(best)
    return selected

def optimized_rank(
    query_embedding: List[float],
    query_text: str,
    candidates: List[Dict[str, Any]],
    top_k: int = 5,
    use_mmr: bool = True,
) -> List[Dict[str, Any]]:
    """
    Full ranking pipeline:
    vector similarity + BM25 keyword score + recency decay → MMR diversity filter.
    """
    query_terms = query_text.lower().split()

    for c in candidates:
        vec_score = cosine_similarity(query_embedding, c.get("embedding", []))
        kw_score  = _bm25_score(query_terms, c.get("text", "")) / 10  # normalise
        recency   = _recency_decay(c.get("uploaded"))
        combined  = (0.65 * vec_score + 0.25 * min(kw_score, 1.0) + 0.10) * recency
        c["_similarity"]    = vec_score
        c["_bm25"]          = kw_score
        c["_combined_score"] = round(min(combined, 1.0), 4)

    candidates.sort(key=lambda x: x["_combined_score"], reverse=True)

    if use_mmr:
        return _mmr_diverse(candidates, top_k=top_k)
    return candidates[:top_k]