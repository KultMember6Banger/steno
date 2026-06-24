"""Tests for the self-curating loop (steno curate).

Exercises the orchestration WITHOUT heavy deps by stubbing build_index and the
Vigil scanner entry points. Covers:
  - Vigil ABSENT: gate + scoring are skipped; compress + index still run.
  - Vigil PRESENT: gate runs (CRITICAL conflicts skip indexing unless --yes)
    and health scoring runs.

The "present" case uses the real soft-import path (Vigil is on PYTHONPATH via a
sibling ../vigil/src checkout); only the expensive scanner calls are stubbed.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, ROOT)

import steno_curate as sc  # noqa: E402


def _write(d: Path, name: str, content: str) -> Path:
    p = d / name
    p.write_text(content, encoding='utf-8')
    return p


def _stub_build_index(monkeypatch):
    """Stub memory_index.build_index to record which files exist at index time."""
    seen = {}

    def fake_build_index(memory_dir, store_dir=None, collection_name=None, **kw):
        files = sorted(f.stem for f in Path(memory_dir).glob('*.md'))
        seen['files'] = files
        return {'records_indexed': len(files), 'records_unchanged': 0}

    import memory_index
    monkeypatch.setattr(memory_index, 'build_index', fake_build_index)
    return seen


# ---------------------------------------------------------------------------
# Vigil ABSENT
# ---------------------------------------------------------------------------
def test_curate_without_vigil_compress_and_index(monkeypatch, tmp_path):
    # Force the soft import to fail -> Vigil unavailable.
    monkeypatch.setattr(sc, '_try_import_vigil', lambda: None)
    seen = _stub_build_index(monkeypatch)

    _write(tmp_path, 'a.md', '---\ntype: project\n---\nThe service is currently running on the host.\n')

    summary = sc.curate(tmp_path, compress=True, gate=True, store_dir=tmp_path / 'store', verbose=False)

    assert summary['vigil_available'] is False
    assert summary['compressed'] == 1          # prose compressed
    assert summary['gated'] == 0               # gate skipped (no vigil)
    assert summary['scored'] == 0              # scoring skipped (no vigil)
    assert summary['indexed'] == 1             # indexed anyway
    assert seen['files'] == ['a']
    # File was rewritten with format: steno frontmatter.
    assert 'format: steno' in (tmp_path / 'a.md').read_text()


def test_curate_without_vigil_skips_already_steno(monkeypatch, tmp_path):
    monkeypatch.setattr(sc, '_try_import_vigil', lambda: None)
    _stub_build_index(monkeypatch)

    _write(tmp_path, 'a.md', '---\nformat: steno\n---\nalready compressed body\n')
    summary = sc.curate(tmp_path, compress=True, store_dir=tmp_path / 'store', verbose=False)
    assert summary['compressed'] == 0          # steno file skipped


# ---------------------------------------------------------------------------
# Vigil PRESENT (real soft-import; scanner calls stubbed for speed)
# ---------------------------------------------------------------------------
def _require_vigil():
    if not sc.vigil_available():
        pytest.skip('vigil not importable (expected via sibling ../vigil/src or pip)')


def _make_issue(category, severity, files):
    vigil = sc._try_import_vigil()
    from vigil.scanner import Issue
    return Issue(severity=severity, category=category, message='stub', files=files, details={})


def test_curate_with_vigil_gates_critical(monkeypatch, tmp_path):
    _require_vigil()
    seen = _stub_build_index(monkeypatch)

    import vigil.scanner as vs
    # bad.md trips a CRITICAL pre_write_conflict; good.md is clean.
    def fake_pre_write_check(text, store_dir, source_file='', collection_name=None, **kw):
        if source_file == 'bad':
            return [_make_issue('pre_write_conflict', 'CRITICAL', ['existing'])]
        return []
    monkeypatch.setattr(vs, 'pre_write_check', fake_pre_write_check)
    # Make scoring a no-op so we don't need a real ChromaDB store.
    monkeypatch.setattr(vs, 'full_scan', lambda *a, **k: {})
    monkeypatch.setattr(vs, 'compute_health_scores', lambda results: {})
    monkeypatch.setattr(vs, 'update_health_scores', lambda scores, store_dir, **k: 7)

    _write(tmp_path, 'good.md', '---\ntype: project\n---\nGood content.\n')
    _write(tmp_path, 'bad.md', '---\ntype: project\n---\nBad content.\n')

    summary = sc.curate(tmp_path, gate=True, store_dir=tmp_path / 'store', verbose=False)

    assert summary['vigil_available'] is True
    assert summary['gated'] == 1               # bad.md gated
    assert 'bad' not in seen['files']          # gated file NOT indexed
    assert 'good' in seen['files']
    assert summary['scored'] == 7              # health scoring ran
    # Gated file restored after indexing.
    assert (tmp_path / 'bad.md').exists()
    assert not (tmp_path / 'bad.md.gated').exists()


def test_curate_with_vigil_yes_overrides_gate(monkeypatch, tmp_path):
    _require_vigil()
    seen = _stub_build_index(monkeypatch)

    import vigil.scanner as vs
    monkeypatch.setattr(vs, 'pre_write_check',
                        lambda text, store_dir, source_file='', collection_name=None, **kw:
                        [_make_issue('pre_write_conflict', 'CRITICAL', ['existing'])])
    monkeypatch.setattr(vs, 'full_scan', lambda *a, **k: {})
    monkeypatch.setattr(vs, 'compute_health_scores', lambda results: {})
    monkeypatch.setattr(vs, 'update_health_scores', lambda scores, store_dir, **k: 0)

    _write(tmp_path, 'bad.md', '---\ntype: project\n---\nBad content.\n')

    summary = sc.curate(tmp_path, gate=True, yes=True, store_dir=tmp_path / 'store', verbose=False)
    # --yes: flagged but NOT skipped from indexing.
    assert summary['gated'] == 0
    assert 'bad' in seen['files']
