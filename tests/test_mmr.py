"""Tests for mmr — Maximal Marginal Relevance diversification."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mmr


def test_empty_returns_empty():
    assert mmr.mmr_rank([1.0, 0.0], []) == []


def test_lambda_one_is_pure_relevance():
    query = [1.0, 0.0]
    # doc0 most relevant, doc1 less, doc2 least.
    docs = [[1.0, 0.0], [0.8, 0.2], [0.2, 0.8]]
    order = mmr.mmr_rank(query, docs, lambda_mult=1.0)
    assert order[0] == 0  # most relevant first


def test_diversity_avoids_near_duplicates():
    query = [1.0, 0.0, 0.0]
    # doc0 and doc1 are near-duplicates; doc2 is relevant but in a new direction.
    docs = [
        [1.0, 0.0, 0.0],    # top relevance
        [0.95, 0.31, 0.0],  # near-duplicate of doc0
        [0.6, 0.0, 0.8],    # different direction
    ]
    # Diversity-leaning lambda: the new-direction doc2 should beat the
    # near-duplicate doc1 for the second slot.
    order = mmr.mmr_rank(query, docs, lambda_mult=0.3)
    assert order[0] == 0
    assert order[1] == 2
    assert order[2] == 1


def test_top_k_limits_output():
    query = [1.0, 0.0]
    docs = [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]]
    order = mmr.mmr_rank(query, docs, top_k=2)
    assert len(order) == 2


def test_returns_permutation_of_indices():
    query = [1.0, 1.0]
    docs = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]]
    order = mmr.mmr_rank(query, docs)
    assert sorted(order) == [0, 1, 2, 3]
