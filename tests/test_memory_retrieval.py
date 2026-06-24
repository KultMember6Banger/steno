"""Tests for memory_retrieval with chromadb + sentence_transformers MOCKED.

These run in environments without the heavy deps installed. We inject fake
modules into sys.modules BEFORE importing memory_index / memory_retrieval, then
exercise the query() logic — especially:
  - access-tracking fix #1: only records that PASS the min_score filter get
    their access_count bumped (NOT the first-N raw results).
  - cosine/score math: score = (1 - distance) * health_score.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest import mock

ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, ROOT)


# ---------------------------------------------------------------------------
# Inject fake chromadb + sentence_transformers so the modules import.
# ---------------------------------------------------------------------------
def _install_fake_heavy_deps():
    fake_chromadb = types.ModuleType('chromadb')

    class _ClientAPI:  # type annotations reference these
        pass

    class _Collection:
        pass

    fake_chromadb.ClientAPI = _ClientAPI
    fake_chromadb.Collection = _Collection
    fake_chromadb.PersistentClient = mock.MagicMock()
    sys.modules['chromadb'] = fake_chromadb

    fake_st = types.ModuleType('sentence_transformers')

    class _FakeST:
        def __init__(self, *a, **k):
            pass

        def encode(self, texts, show_progress_bar=False):
            import array
            # Return an object with .tolist() like numpy would.
            class _Vecs:
                def tolist(self_inner):
                    return [[0.1, 0.2, 0.3] for _ in texts]
            return _Vecs()

    fake_st.SentenceTransformer = _FakeST
    sys.modules['sentence_transformers'] = fake_st


_install_fake_heavy_deps()

import memory_retrieval as mr  # noqa: E402


def _make_raw_results(items):
    """Build a ChromaDB-shaped results dict from (id, distance, meta) tuples."""
    return {
        'ids': [[i[0] for i in items]],
        'distances': [[i[1] for i in items]],
        'documents': [[f"doc for {i[0]}" for i in items]],
        'metadatas': [[i[2] for i in items]],
    }


def _patch_query(monkeypatch, raw_results):
    """Patch chromadb client/collection + model so query() runs against fixtures."""
    fake_collection = mock.MagicMock()
    fake_collection.query.return_value = raw_results

    fake_client = mock.MagicMock()
    fake_client.get_collection.return_value = fake_collection

    monkeypatch.setattr(mr.chromadb, 'PersistentClient', lambda path: fake_client)
    # SentenceTransformer already faked at module level.
    return fake_collection


def test_score_math_similarity_times_health(monkeypatch):
    # distance 0.2 -> similarity 0.8; health 0.5 -> score 0.4
    raw = _make_raw_results([
        ('r1', 0.2, {'health_score': 0.5, 'source_file': 'f', 'record_type': '@T', 'memory_type': 'project'}),
    ])
    _patch_query(monkeypatch, raw)
    results = mr.query("anything", min_score=0.0)
    assert len(results) == 1
    assert abs(results[0].score - 0.4) < 1e-6


def test_min_score_filters_out_low(monkeypatch):
    raw = _make_raw_results([
        ('hi', 0.1, {'source_file': 'a'}),   # sim 0.9
        ('lo', 0.8, {'source_file': 'b'}),   # sim 0.2 -> filtered at min 0.3
    ])
    _patch_query(monkeypatch, raw)
    results = mr.query("q", min_score=0.3)
    ids = [r.id for r in results]
    assert ids == ['hi']


def test_access_tracking_bumps_only_passing_records(monkeypatch):
    """The bug fix: a high-ranked hit filtered out must NOT be bumped, and a
    later record that passes MUST be bumped — not just the first-N raw rows."""
    raw = _make_raw_results([
        ('rank0_filtered', 0.95, {'health_score': 1.0, 'source_file': 'x'}),  # sim 0.05 -> FILTERED
        ('rank1_passes', 0.1, {'health_score': 1.0, 'source_file': 'y'}),     # sim 0.9 -> PASSES
        ('rank2_passes', 0.2, {'health_score': 1.0, 'source_file': 'z'}),     # sim 0.8 -> PASSES
    ])
    fake_collection = _patch_query(monkeypatch, raw)

    results = mr.query("q", min_score=0.3)
    returned_ids = {r.id for r in results}
    assert returned_ids == {'rank1_passes', 'rank2_passes'}

    # Exactly one update call; it must bump ONLY the passing ids.
    assert fake_collection.update.call_count == 1
    _, kwargs = fake_collection.update.call_args
    bumped_ids = kwargs['ids']
    assert set(bumped_ids) == {'rank1_passes', 'rank2_passes'}
    assert 'rank0_filtered' not in bumped_ids

    # And the bumped metas have access_count incremented to 1 + last_accessed set.
    for meta in kwargs['metadatas']:
        assert meta['access_count'] == 1
        assert meta['last_accessed']  # non-empty iso string


def test_no_update_when_nothing_passes(monkeypatch):
    raw = _make_raw_results([
        ('lo', 0.9, {'source_file': 'a'}),  # sim 0.1 -> filtered
    ])
    fake_collection = _patch_query(monkeypatch, raw)
    results = mr.query("q", min_score=0.5)
    assert results == []
    fake_collection.update.assert_not_called()


def test_empty_query_returns_empty(monkeypatch):
    assert mr.query("   ") == []


def test_collection_name_threaded(monkeypatch):
    raw = _make_raw_results([('r1', 0.1, {'source_file': 'a'})])
    fake_collection = mock.MagicMock()
    fake_collection.query.return_value = raw
    fake_client = mock.MagicMock()
    fake_client.get_collection.return_value = fake_collection
    monkeypatch.setattr(mr.chromadb, 'PersistentClient', lambda path: fake_client)

    mr.query("q", collection_name='custom_coll', min_score=0.0)
    fake_client.get_collection.assert_called_once_with('custom_coll')


def test_default_min_score_is_030():
    import inspect
    sig = inspect.signature(mr.query)
    assert sig.parameters['min_score'].default == 0.30
