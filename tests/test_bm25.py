"""Tests for bm25 — tokenization, ranking, and reciprocal rank fusion."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import bm25


# --- tokenization --------------------------------------------------------
def test_tokenize_lowercases():
    assert bm25.tokenize("Auth Service") == ['auth', 'service']


def test_tokenize_keeps_identifiers():
    toks = bm25.tokenize("connect to port 50052 for BUG-123 v1.2.3")
    assert '50052' in toks
    assert 'bug-123' in toks
    assert 'v1.2.3' in toks


def test_tokenize_empty():
    assert bm25.tokenize('') == []
    assert bm25.tokenize(None) == []


# --- ranking -------------------------------------------------------------
def test_exact_token_ranks_top():
    corpus = [
        "the deployment process for the web application",
        "auth service listens on port 50052 for grpc",
        "billing api migration is sixty percent complete",
    ]
    b = bm25.BM25(corpus)
    top = b.top_n("port 50052", n=3)
    assert top, "expected at least one BM25 hit"
    assert top[0][0] == 1  # the doc containing 'port 50052'


def test_scores_parallel_to_corpus():
    corpus = ["alpha beta", "gamma delta", "alpha alpha alpha"]
    b = bm25.BM25(corpus)
    scores = b.get_scores("alpha")
    assert len(scores) == 3
    # doc 2 has the most 'alpha' occurrences -> highest score
    assert scores[2] > scores[0]
    assert scores[1] == 0.0  # no 'alpha'


def test_unknown_term_yields_zero_scores():
    b = bm25.BM25(["one two", "three four"])
    assert b.top_n("zzzznotpresent") == []


def test_empty_corpus_is_safe():
    b = bm25.BM25([])
    assert b.get_scores("anything") == []
    assert b.top_n("anything") == []


# --- reciprocal rank fusion ----------------------------------------------
def test_rrf_rewards_consensus():
    # 'b' appears high in both rankings; 'a' only tops one.
    r1 = ['a', 'b', 'c']
    r2 = ['b', 'x', 'y']
    fused = bm25.reciprocal_rank_fusion([r1, r2])
    # b is rank1 in r1 and rank0 in r2 -> should beat a (rank0 in r1 only).
    assert fused['b'] > fused['a']
    assert fused['b'] > fused['x']


def test_rrf_single_ranking_preserves_order():
    fused = bm25.reciprocal_rank_fusion([['a', 'b', 'c']])
    ordered = sorted(fused, key=lambda x: fused[x], reverse=True)
    assert ordered == ['a', 'b', 'c']
