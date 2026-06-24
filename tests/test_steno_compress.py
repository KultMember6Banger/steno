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
