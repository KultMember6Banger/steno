"""Tests for memory_index with chromadb + sentence_transformers MOCKED.

Covers the metadata schema (access_count/last_accessed defaults, source_file,
record_type, memory_type) and store-dir / collection precedence resolution.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest import mock

ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, ROOT)


def _install_fake_heavy_deps():
    fake_chromadb = types.ModuleType('chromadb')
    fake_chromadb.ClientAPI = type('ClientAPI', (), {})
    fake_chromadb.Collection = type('Collection', (), {})
    fake_chromadb.PersistentClient = mock.MagicMock()
    sys.modules['chromadb'] = fake_chromadb

    fake_st = types.ModuleType('sentence_transformers')

    class _FakeST:
        def __init__(self, *a, **k):
            pass

        def encode(self, texts, show_progress_bar=False):
            class _Vecs:
                def tolist(self_inner):
                    return [[0.1, 0.2, 0.3] for _ in texts]
            return _Vecs()

    fake_st.SentenceTransformer = _FakeST
    sys.modules['sentence_transformers'] = fake_st


_install_fake_heavy_deps()

import memory_index as mi  # noqa: E402
from steno_parser import Record  # noqa: E402


def test_sanitize_metadata_schema_defaults():
    rec = Record(
        id='f:p:0',
        source_file='f',
        record_type='chunk',
        text='hi',
        metadata={'memory_type': 'feedback', 'format': 'prose'},
    )
    m = mi._sanitize_metadata(rec)
    assert m['source_file'] == 'f'
    assert m['record_type'] == 'chunk'
    assert m['memory_type'] == 'feedback'
    assert m['access_count'] == 0
    assert m['last_accessed'] == ''
    # health_score is WRITTEN by Vigil, not by Steno — must NOT be stripped/added here.
    assert 'health_score' not in m


def test_sanitize_metadata_optional_scope_name():
    rec = Record('id', 'src', '@T', 'txt', {'scope': 'proj', 'name': 'svc'})
    m = mi._sanitize_metadata(rec)
    assert m['scope'] == 'proj'
    assert m['name'] == 'svc'


def test_default_collection_is_agent_memory():
    assert mi.COLLECTION_NAME == 'agent_memory'


def test_get_collection_uses_default(monkeypatch):
    client = mock.MagicMock()
    mi.get_collection(client)
    args, kwargs = client.get_or_create_collection.call_args
    assert kwargs['name'] == 'agent_memory'


def test_get_collection_respects_override():
    client = mock.MagicMock()
    mi.get_collection(client, 'custom')
    _, kwargs = client.get_or_create_collection.call_args
    assert kwargs['name'] == 'custom'


def test_store_dir_precedence(monkeypatch):
    # STENO_STORE wins over MEMORY_STORE.
    monkeypatch.setenv('STENO_STORE', '/tmp/steno_store')
    monkeypatch.setenv('MEMORY_STORE', '/tmp/memory_store')
    assert mi._default_store_dir() == Path('/tmp/steno_store')

    # MEMORY_STORE used as fallback when STENO_STORE unset.
    monkeypatch.delenv('STENO_STORE', raising=False)
    assert mi._default_store_dir() == Path('/tmp/memory_store')

    # Neither set -> ./chroma_store next to module.
    monkeypatch.delenv('MEMORY_STORE', raising=False)
    assert mi._default_store_dir().name == 'chroma_store'


def test_build_index_indexes_records_and_writes_metadata(monkeypatch, tmp_path):
    # One memory file with parseable prose content.
    (tmp_path / "note.md").write_text(
        "---\ntype: project\n---\n# Heading\n"
        "Enough prose content here to make a real chunk record for indexing.\n"
    )

    fake_collection = mock.MagicMock()
    fake_client = mock.MagicMock()
    fake_client.get_or_create_collection.return_value = fake_collection

    store = tmp_path / "store"

    def _fake_get_client(store_dir=None):
        # Mirror the real get_client side effect: ensure the store dir exists
        # (real impl calls store_dir.mkdir(parents=True, exist_ok=True)).
        if store_dir is not None:
            Path(store_dir).mkdir(parents=True, exist_ok=True)
        return fake_client

    monkeypatch.setattr(mi, 'get_client', _fake_get_client)

    stats = mi.build_index(tmp_path, store_dir=store, rebuild=True)

    assert stats['records_indexed'] >= 1
    assert fake_collection.add.called
    _, kwargs = fake_collection.add.call_args
    metas = kwargs['metadatas']
    assert metas[0]['source_file'] == 'note'
    assert metas[0]['memory_type'] == 'project'
    assert metas[0]['access_count'] == 0
