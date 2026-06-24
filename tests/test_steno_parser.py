"""Tests for steno_parser — format detection, parsing, junk filtering, frontmatter."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import steno_parser as sp


# --- frontmatter ---------------------------------------------------------
def test_parse_frontmatter_extracts_yaml():
    content = "---\nname: Foo\ntype: feedback\n---\n\nbody text here"
    meta, body = sp.parse_frontmatter(content)
    assert meta == {'name': 'Foo', 'type': 'feedback'}
    assert body.strip() == 'body text here'


def test_parse_frontmatter_no_frontmatter():
    content = "no frontmatter here\njust body"
    meta, body = sp.parse_frontmatter(content)
    assert meta == {}
    assert body == content


def test_parse_frontmatter_malformed_yaml_is_safe():
    content = "---\n: : : not valid\n---\nbody"
    meta, body = sp.parse_frontmatter(content)
    assert meta == {}
    assert body == 'body'


# --- format detection ----------------------------------------------------
def test_detect_format_explicit_steno():
    assert sp.detect_format({'format': 'steno'}, 'whatever') == 'steno'


def test_detect_format_explicit_steno_m():
    assert sp.detect_format({'format': 'steno-m'}, 'whatever') == 'steno-m'


def test_detect_format_infers_steno_m_from_prefix():
    body = "#scope proj\n@T auth-service|active|primary auth"
    assert sp.detect_format({}, body) == 'steno-m'


def test_detect_format_defaults_to_prose():
    body = "# Heading\n\nSome ordinary prose with no special prefixes."
    assert sp.detect_format({}, body) == 'prose'


# --- junk filtering ------------------------------------------------------
def test_is_junk_separators():
    assert sp._is_junk('-----')
    assert sp._is_junk('====')
    assert sp._is_junk('   \n  ')


def test_is_junk_tiny_fragment():
    assert sp._is_junk('hi')  # < 15 alnum chars


def test_is_junk_false_for_real_content():
    assert not sp._is_junk('This is a real sentence with enough content.')


# --- steno-m parsing -----------------------------------------------------
def test_parse_steno_m_records_and_scope():
    body = (
        "#scope myproject\n"
        "#schemas @F @T\n"
        "@F BUG-123|open|high|auth-bypass|auth-service|evidence\n"
        "@T auth-service|active|go-grpc|primary auth\n"
    )
    recs = sp.parse_steno_m(body, 'findings', {'memory_type': 'project'})
    assert len(recs) == 2
    assert recs[0].record_type == '@F'
    assert recs[0].id == 'findings:@F:BUG-123'
    assert recs[0].metadata['scope'] == 'myproject'
    assert recs[1].record_type == '@T'
    assert recs[1].id == 'findings:@T:auth-service'


# --- steno parsing -------------------------------------------------------
def test_parse_steno_splits_on_headers():
    body = (
        "# First section heading line\n"
        "content for the first section that is long enough to keep around\n\n"
        "## Second section heading line\n"
        "content for the second section also long enough to be kept here\n"
    )
    recs = sp.parse_steno(body, 'mem', {})
    assert len(recs) == 2
    assert all(r.record_type == 'chunk' for r in recs)
    assert all(r.metadata['record_type'] == 'steno_block' for r in recs)


def test_parse_steno_headerless_falls_back_to_paragraphs():
    body = "\n\n".join(f"paragraph number {i} with enough text to be a real chunk here" for i in range(3))
    recs = sp.parse_steno(body, 'mem', {})
    assert len(recs) >= 1
    assert recs[0].record_type == 'chunk'


# --- prose parsing -------------------------------------------------------
def test_parse_prose_chunks_sections():
    body = (
        "# Heading one\nA decent amount of prose content under heading one.\n\n"
        "# Heading two\nMore prose content under heading two for completeness.\n"
    )
    recs = sp.parse_prose(body, 'doc', {})
    assert len(recs) == 2
    assert all(r.metadata['record_type'] == 'prose_section' for r in recs)


def test_parse_prose_large_section_splits_by_paragraph():
    big = "\n\n".join(["x" * 200 + " words here padding the paragraph out nicely"] * 5)
    body = "# Heading\n" + big
    recs = sp.parse_prose(body, 'doc', {}, chunk_size=300)
    assert len(recs) >= 2


# --- end to end via parse_file ------------------------------------------
def test_parse_file_prose(tmp_path):
    f = tmp_path / "note.md"
    f.write_text(
        "---\ntype: project\n---\n\n# Decision log\n"
        "We decided to ship the feature behind a flag for safety reasons here.\n"
    )
    recs = sp.parse_file(f)
    assert recs
    assert recs[0].metadata['format'] == 'prose'
    assert recs[0].metadata['memory_type'] == 'project'
    assert recs[0].source_file == 'note'


def test_parse_directory_skips_special_files(tmp_path):
    (tmp_path / "MEMORY.md").write_text("---\n---\nshould be skipped entirely here\n")
    (tmp_path / "README.md").write_text("---\n---\nalso skipped completely here\n")
    (tmp_path / "real.md").write_text(
        "---\ntype: user\n---\n# Real\nActual content that should be parsed into a record.\n"
    )
    recs = sp.parse_directory(tmp_path)
    assert recs
    assert all(r.source_file == 'real' for r in recs)


# --- nested metadata.type (Claude Code memory format) --------------------
def test_parse_file_resolves_nested_metadata_type(tmp_path):
    f = tmp_path / "nested.md"
    f.write_text(
        "---\nname: Nested\ndescription: d\nmetadata:\n  type: feedback\n---\n"
        "rule: integration tests must connect to a real database, never mock it\n"
    )
    recs = sp.parse_file(f)
    assert recs, "expected at least one record"
    assert recs[0].metadata.get("memory_type") == "feedback", "nested metadata.type should resolve"
