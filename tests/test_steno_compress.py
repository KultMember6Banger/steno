"""Tests for steno_compress — article dropping, abbreviation, dates, preservation."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import steno_compress as sc


# --- article dropping ----------------------------------------------------
def test_drops_articles():
    out = sc.compress("the cat sat on a mat near an oak")
    assert 'the' not in out.split()
    assert 'a' not in out.split()
    assert 'an' not in out.split()
    assert 'cat' in out and 'mat' in out and 'oak' in out


def test_articles_kept_in_headings():
    out = sc.compress("# The deployment guide\nthe body drops the article")
    lines = out.split('\n')
    assert lines[0] == '# The deployment guide'  # heading untouched re: articles
    assert 'the' not in lines[1].split()


def test_steno_m_level_keeps_articles_off_by_default_but_steno_drops():
    # Non-'steno' level should NOT drop articles (reserved/conservative).
    out = sc.compress("the quick brown fox", level='other')
    assert 'the' in out.split()


# --- abbreviation --------------------------------------------------------
def test_abbreviations_applied():
    out = sc.compress("authentication configuration verified in production environment")
    assert 'auth' in out
    assert 'cfg' in out
    assert 'vfd' in out
    assert 'prod' in out
    assert 'env' in out


def test_abbreviation_preserves_leading_capital():
    out = sc.compress("Configuration of the Database")
    assert 'Cfg' in out
    assert 'Db' in out


def test_abbreviation_dict_is_extendable():
    sc.ABBREVIATIONS['zzztestword'] = 'ztw'
    try:
        out = sc.compress("a zzztestword here")
        assert 'ztw' in out
    finally:
        del sc.ABBREVIATIONS['zzztestword']


# --- date compression ----------------------------------------------------
def test_iso_date_compression():
    out = sc.compress("shipped on 2026-04-11 after review")
    assert '04-11' in out
    assert '2026-04-11' not in out


# --- preservation --------------------------------------------------------
def test_inline_code_preserved():
    out = sc.compress("set the `configuration` value to true")
    assert '`configuration`' in out  # not abbreviated inside backticks


def test_urls_preserved():
    url = "https://example.com/the/authentication/path"
    out = sc.compress(f"visit {url} now")
    assert url in out  # 'the' and 'authentication' inside URL untouched


def test_email_preserved():
    out = sc.compress("contact the.admin@example.com about it")
    assert 'the.admin@example.com' in out


def test_fenced_code_block_preserved():
    text = "# Heading\nthe prose here\n```\nthe configuration code stays verbatim\n```\nthe trailing prose"
    out = sc.compress(text)
    assert 'the configuration code stays verbatim' in out  # untouched in fence


def test_whitespace_collapsed():
    out = sc.compress("word1     word2\t\tword3")
    assert '   ' not in out
    assert 'word1 word2 word3' in out


# --- ratio helper --------------------------------------------------------
def test_compression_ratio_positive_on_compressible_text():
    text = "the authentication configuration was verified in the production environment"
    out = sc.compress(text)
    assert sc.compression_ratio(text, out) > 0


def test_compression_ratio_empty():
    assert sc.compression_ratio('', '') == 0.0


# --- legend --------------------------------------------------------------
def test_expand_legend_is_inverse():
    legend = sc.expand_legend()
    assert legend['auth'] == 'authentication'
    assert legend['cfg'] in ('configuration', 'configure')


# --- llm hook ------------------------------------------------------------
def test_llm_hook_invoked_on_prose():
    calls = []

    def hook(s):
        calls.append(s)
        return s.upper()

    out = sc.compress("the authentication works", llm_hook=hook)
    assert calls  # hook was called
    assert 'AUTH WORKS' in out


def test_llm_hook_failure_is_swallowed():
    def bad_hook(s):
        raise RuntimeError("boom")

    out = sc.compress("the configuration here", llm_hook=bad_hook)
    assert 'cfg' in out  # rule-based output survives hook failure


# --- expand (best-effort inverse) ----------------------------------------
def test_expand_reverses_abbreviations():
    out = sc.expand("auth cfg vfd in prod env")
    assert 'authentication' in out
    assert 'configuration' in out
    assert 'verified' in out
    assert 'production' in out
    assert 'environment' in out


def test_expand_preserves_leading_capital():
    out = sc.expand("Cfg of the Db")
    assert 'Configuration' in out
    assert 'Database' in out


def test_expand_roundtrip_over_abbreviation_set():
    """compress then expand should recover the full word for every unambiguous
    abbreviation in the legend (cfg is the documented ambiguous exception)."""
    for full, abbr in sc.ABBREVIATIONS.items():
        compressed = sc.compress(full)  # single word, no articles to drop
        assert compressed.strip() == abbr, f"{full} did not compress to {abbr}"
        expanded = sc.expand(compressed).strip()
        # cfg maps from both configure/configuration; expansion picks one.
        if abbr == 'cfg':
            assert expanded == 'configuration'
        else:
            assert expanded == full, f"{abbr} expanded to {expanded}, wanted {full}"


def test_expand_preserves_inline_code():
    out = sc.expand("set `auth` to vfd")
    assert '`auth`' in out  # not expanded inside backticks
    assert 'verified' in out


def test_expand_preserves_fenced_code():
    text = "auth here\n```\nauth cfg verbatim\n```\nvfd there"
    out = sc.expand(text)
    assert 'auth cfg verbatim' in out  # untouched in fence
    assert 'authentication here' in out


def test_expand_does_not_restore_articles():
    # documented limitation: dropped articles are NOT recovered.
    compressed = sc.compress("the auth in the prod env")
    expanded = sc.expand(compressed)
    assert 'authentication' in expanded
    assert 'the' not in expanded.split()  # still gone


# --- verify / fidelity (real embeddings) ---------------------------------
import pytest


@pytest.fixture
def real_embeddings():
    """Ensure the REAL sentence_transformers is used.

    Sibling test modules inject a fake `sentence_transformers` into sys.modules
    to run without heavy deps; that fake leaks into this process. These fidelity
    tests need the real MiniLM model, so we drop any fake and let compute_fidelity
    re-import the genuine package. Skips if it isn't actually installed.
    """
    import importlib
    saved = sys.modules.get('sentence_transformers')
    if saved is not None and not hasattr(saved, '__file__'):
        del sys.modules['sentence_transformers']  # remove the fake
    try:
        importlib.import_module('sentence_transformers')
    except Exception:
        pytest.skip('real sentence_transformers not available')
    yield
    if saved is not None:
        sys.modules['sentence_transformers'] = saved  # restore the fake for others


def test_compress_verify_returns_tuple(real_embeddings):
    text = "the deployment pipeline runs nightly and the team reviews the results"
    out, fidelity = sc.compress(text, verify=True)
    assert isinstance(out, str)
    assert isinstance(fidelity, float)
    assert -1.0 <= fidelity <= 1.0


def test_verify_article_dropping_preserves_fidelity(real_embeddings):
    # Article-only compression (no obscure abbreviations) keeps the embedder's
    # view of meaning largely intact.
    text = "A quick brown fox jumps over the lazy dog while the cat watches from a fence"
    out, fidelity = sc.compress(text, verify=True)
    assert 'the' not in out.split() and 'a' not in out.split()
    assert fidelity >= sc.FIDELITY_THRESHOLD


def test_verify_flags_abbreviation_meaning_loss(real_embeddings):
    # Heavy abbreviation obscures meaning to MiniLM (it doesn't know the legend),
    # so fidelity drops below threshold — exactly the signal --verify surfaces.
    text = "the authentication configuration was verified in the production environment"
    out, fidelity = sc.compress(text, verify=True)
    assert fidelity < sc.FIDELITY_THRESHOLD


def test_compute_fidelity_identical_text_is_high(real_embeddings):
    f = sc.compute_fidelity("hello world this is a test sentence", "hello world this is a test sentence")
    assert f > 0.99
