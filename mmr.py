"""
MMR — Maximal Marginal Relevance re-ranking for retrieval diversification.

Pure-vector top-K often returns five near-duplicate records (the same fact
restated across memory files). MMR re-orders candidates to balance *relevance to
the query* against *novelty vs already-selected results*, so the returned set
covers more ground instead of repeating one point.

Selection rule (Carbonell & Goldstein, 1998):

    MMR = argmax_{d in candidates} [ lambda * sim(d, query)
                                     - (1 - lambda) * max_{s in selected} sim(d, s) ]

lambda=1.0 -> pure relevance (original order); lambda=0.0 -> pure diversity.
Default lambda=0.5 balances the two.

Inputs are plain Python lists of floats (the embeddings ChromaDB already returns
when you include 'embeddings' in the query). Cosine similarity is computed with
numpy if available, else a pure-Python fallback — no hard numpy dependency for
the math, though numpy is present in this project.
"""

from __future__ import annotations

import math
from typing import Sequence

try:  # numpy is available in this project but keep a pure-python fallback.
    import numpy as _np
except Exception:  # pragma: no cover - numpy is installed here
    _np = None


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity between two vectors (0 if either is zero-length)."""
    if _np is not None:
        av = _np.asarray(a, dtype=float)
        bv = _np.asarray(b, dtype=float)
        na = float(_np.linalg.norm(av))
        nb = float(_np.linalg.norm(bv))
        if na == 0.0 or nb == 0.0:
            return 0.0
        return float(_np.dot(av, bv) / (na * nb))
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def mmr_rank(
    query_embedding: Sequence[float],
    doc_embeddings: Sequence[Sequence[float]],
    lambda_mult: float = 0.5,
    top_k: int | None = None,
) -> list[int]:
    """Return candidate indices re-ordered by Maximal Marginal Relevance.

    Args:
        query_embedding: the query vector.
        doc_embeddings: candidate document vectors (parallel to your results).
        lambda_mult: trade-off in [0, 1]; 1.0 = pure relevance, 0.0 = pure
            diversity. Default 0.5.
        top_k: how many indices to select (default: all candidates).

    Returns:
        list of indices into doc_embeddings, in MMR-selected order.
    """
    n = len(doc_embeddings)
    if n == 0:
        return []
    if top_k is None or top_k > n:
        top_k = n

    # Precompute query relevance for each candidate.
    rel = [_cosine(query_embedding, d) for d in doc_embeddings]

    selected: list[int] = []
    remaining = set(range(n))

    # Seed with the most relevant candidate.
    first = max(remaining, key=lambda i: rel[i])
    selected.append(first)
    remaining.discard(first)

    # Cache pairwise sims lazily.
    sim_cache: dict[tuple[int, int], float] = {}

    def pair_sim(i: int, j: int) -> float:
        key = (i, j) if i < j else (j, i)
        if key not in sim_cache:
            sim_cache[key] = _cosine(doc_embeddings[i], doc_embeddings[j])
        return sim_cache[key]

    while remaining and len(selected) < top_k:
        best_idx = None
        best_score = -math.inf
        for i in remaining:
            max_sim_to_selected = max(pair_sim(i, s) for s in selected)
            score = lambda_mult * rel[i] - (1 - lambda_mult) * max_sim_to_selected
            if score > best_score:
                best_score = score
                best_idx = i
        selected.append(best_idx)
        remaining.discard(best_idx)

    return selected
