"""Tests for the shared canonical core (memcore) — Steno copy.

Covers the pure helpers (no heavy deps): frontmatter parsing, cosine/distance
math, and store/collection resolution. The embedder loader + ChromaDB
collection helpers are exercised indirectly by the heavier suites; here we only
verify the dep-free plumbing.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, ROOT)

import memcore  # noqa: E402


def test_parse_frontmatter_basic():
    meta, body = memcore.parse_frontmatter('---\ntype: project\nname: svc\n---\nhello body\n')
    assert meta == {'type': 'project', 'name': 'svc'}
    assert body.strip() == 'hello body'


def test_parse_frontmatter_missing():
    meta, body = memcore.parse_frontmatter('no frontmatter here')
    assert meta == {}
    assert body == 'no frontmatter here'


def test_parse_frontmatter_invalid_yaml_returns_empty():
    text = '---\n: : bad : yaml :\n---\nbody'
    meta, body = memcore.parse_frontmatter(text)
    assert meta == {}
    # On invalid YAML the full text is returned unchanged.
    assert body == text


def test_parse_frontmatter_non_dict_returns_empty():
    meta, body = memcore.parse_frontmatter('---\n- just\n- a\n- list\n---\nbody')
    assert meta == {}


def test_sim_from_distance():
    assert memcore.sim_from_distance(0.0) == 1.0
    assert memcore.sim_from_distance(0.2) == 0.8
    assert abs(memcore.sim_from_distance(1.0)) < 1e-9


def test_cosine_matrix_identity_and_orthogonal():
    # Identical vectors -> sim 1.0; orthogonal -> sim ~0.0.
    embs = [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    m = memcore.cosine_matrix(embs)
    assert abs(m[0][1] - 1.0) < 1e-5
    assert abs(m[0][2] - 0.0) < 1e-5


def test_resolve_collection_precedence(monkeypatch):
    monkeypatch.delenv('MEMORY_COLLECTION', raising=False)
    assert memcore.resolve_collection() == 'agent_memory'
    assert memcore.resolve_collection('explicit') == 'explicit'
    monkeypatch.setenv('MEMORY_COLLECTION', 'envcol')
    assert memcore.resolve_collection() == 'envcol'
    # Explicit still wins over env.
    assert memcore.resolve_collection('explicit') == 'explicit'


def test_resolve_store_precedence(monkeypatch):
    monkeypatch.delenv('STENO_STORE', raising=False)
    monkeypatch.delenv('MEMORY_STORE', raising=False)
    # explicit wins
    assert memcore.resolve_store('/x/explicit', 'STENO_STORE', 'MEMORY_STORE') == Path('/x/explicit')
    # first env var wins over second
    monkeypatch.setenv('STENO_STORE', '/x/steno')
    monkeypatch.setenv('MEMORY_STORE', '/x/mem')
    assert memcore.resolve_store(None, 'STENO_STORE', 'MEMORY_STORE') == Path('/x/steno')
    monkeypatch.delenv('STENO_STORE', raising=False)
    assert memcore.resolve_store(None, 'STENO_STORE', 'MEMORY_STORE') == Path('/x/mem')
    # falls back to default when nothing set
    monkeypatch.delenv('MEMORY_STORE', raising=False)
    assert memcore.resolve_store(None, 'STENO_STORE', 'MEMORY_STORE', default='/x/def') == Path('/x/def')
    # None when no default and nothing resolves
    assert memcore.resolve_store(None, 'STENO_STORE') is None
