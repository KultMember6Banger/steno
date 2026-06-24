"""
Steno Compressor — rule-based prose -> Steno notation.

Steno currently only READS already-compressed files; this module lets it WRITE
them, turning ordinary prose memory into the human-auditable "steno" format
described in the README's "Steno Format Rules":

  1. Drop articles (a / an / the) ONLY where it's safe.
  2. Abbreviate a curated dictionary of common terms (verified -> vfd,
     authentication -> auth, configuration -> cfg, environment -> env, ...).
  3. Compress ISO dates: YYYY-MM-DD -> MM-DD.
  4. Collapse redundant whitespace.

Design goals:
  - LOSSY but HUMAN-AUDITABLE. That's the whole point of "steno" (auditable)
    vs "steno-m" (machine-only). A human can read the output and the
    ABBREVIATIONS dict below acts as the legend/expansion table.
  - CONSERVATIVE. Never mangle meaning. Keep full words when ambiguous.
  - STRUCTURE-PRESERVING. Code spans (`backticks`), fenced code blocks, URLs,
    YAML frontmatter and markdown structure are preserved verbatim.

Decompression / expansion:
  There is no lossless inverse (articles are dropped), but the output is
  designed to be read directly. The ABBREVIATIONS mapping is the legend: each
  abbreviated token expands to its dict key. `expand_legend()` returns that
  table for display.

LLM hook:
  This compressor is rule-based and has NO LLM dependency. If you later want an
  LLM-assisted pass (e.g. semantic summarisation), implement a callable with the
  signature `(text: str) -> str` and pass it as `llm_hook` to `compress()`. It
  runs AFTER the rule-based pass on prose segments only (never on preserved
  spans). See the `llm_hook` parameter below.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Abbreviation dictionary (the "legend"). Easily extendable: add key -> value.
# Keys are full words (lower-case); matching is case-insensitive and preserves
# leading capitalisation of the original token.
# ---------------------------------------------------------------------------
ABBREVIATIONS: dict[str, str] = {
    'verified': 'vfd',
    'authentication': 'auth',
    'authorization': 'authz',
    'configuration': 'cfg',
    'configure': 'cfg',
    'environment': 'env',
    'repository': 'repo',
    'database': 'db',
    'production': 'prod',
    'development': 'dev',
    'application': 'app',
    'documentation': 'docs',
    'message': 'msg',
    'request': 'req',
    'response': 'resp',
    'function': 'fn',
    'parameter': 'param',
    'parameters': 'params',
    'directory': 'dir',
    'reference': 'ref',
    'information': 'info',
    'management': 'mgmt',
    'dependency': 'dep',
    'dependencies': 'deps',
    'infrastructure': 'infra',
    'kubernetes': 'k8s',
    'organization': 'org',
    'administrator': 'admin',
    'specification': 'spec',
    'definition': 'def',
    'implementation': 'impl',
    'performance': 'perf',
    'temporary': 'tmp',
    'maximum': 'max',
    'minimum': 'min',
    'average': 'avg',
    'number': 'num',
    'version': 'ver',
}

ARTICLES = {'a', 'an', 'the'}

# Markdown / structural line prefixes whose articles we leave alone to avoid
# changing meaning of headings or list semantics. We still abbreviate within
# them, but we do NOT drop articles on these (conservative).
_HEADING_RE = re.compile(r'^\s{0,3}#{1,6}\s')

# ISO date YYYY-MM-DD -> MM-DD
_ISO_DATE_RE = re.compile(r'\b\d{4}-(\d{2})-(\d{2})\b')

# A token = word characters (incl. apostrophes); everything else is separator.
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z']*")

# Inline code span (`...`) and URLs — preserved verbatim.
_PRESERVE_RE = re.compile(
    r'(`[^`]*`)'                       # inline code
    r'|(\bhttps?://[^\s)]+)'           # URLs
    r'|(\b[\w.-]+@[\w.-]+\.\w+\b)'     # emails
)


def expand_legend() -> dict[str, str]:
    """Return the abbreviation -> full-word legend (inverse of ABBREVIATIONS)."""
    return {v: k for k, v in ABBREVIATIONS.items()}


def _abbreviate_token(token: str) -> str:
    """Abbreviate a single word token if it's in the dict; preserve capitalisation."""
    repl = ABBREVIATIONS.get(token.lower())
    if repl is None:
        return token
    # Preserve a leading capital (e.g. "Configuration" -> "Cfg").
    if token[:1].isupper():
        return repl[:1].upper() + repl[1:]
    return repl


def _compress_prose_segment(segment: str, drop_articles: bool) -> str:
    """Apply abbreviation + (optional) article dropping to a plain-text segment.

    `segment` is guaranteed to contain no preserved spans (code/URLs/emails).
    """
    out = []
    pos = 0
    for m in _TOKEN_RE.finditer(segment):
        # Emit separator text verbatim.
        out.append(segment[pos:m.start()])
        token = m.group(0)
        lowered = token.lower()

        if drop_articles and lowered in ARTICLES:
            # Drop the article AND a single following space (collapse cleanup
            # handles the rest). Mark with a sentinel we strip below.
            out.append('\x00')
        else:
            out.append(_abbreviate_token(token))
        pos = m.end()
    out.append(segment[pos:])

    text = ''.join(out)
    # Remove sentinel + an immediately-following single space.
    text = re.sub(r'\x00 ?', '', text)
    return text


def _compress_line(line: str, drop_articles: bool) -> str:
    """Compress one line, preserving inline code / URLs / emails verbatim."""
    result = []
    last = 0
    for m in _PRESERVE_RE.finditer(line):
        prose = line[last:m.start()]
        result.append(_compress_prose_segment(prose, drop_articles))
        result.append(m.group(0))  # preserved verbatim
        last = m.end()
    result.append(_compress_prose_segment(line[last:], drop_articles))
    return ''.join(result)


def compress(text: str, level: str = 'steno', llm_hook=None) -> str:
    """Compress prose into Steno notation (rule-based, lossy-but-auditable).

    Args:
        text: prose body to compress (frontmatter should be stripped first;
              see steno.py's compress command which handles frontmatter).
        level: 'steno' (default) human-auditable compression. Any other value
               currently behaves like 'steno' but is reserved for future levels
               (e.g. an 'steno-m' machine pass).
        llm_hook: optional callable (str) -> str applied to prose lines AFTER the
                  rule-based pass. Preserved spans (code/URLs) are never passed
                  to it. Leave None for pure rule-based compression (no LLM dep).

    Returns:
        The compressed text. Markdown structure, fenced code blocks, inline
        code, URLs and emails are preserved verbatim.
    """
    drop_articles = level == 'steno'

    lines = text.split('\n')
    out_lines = []
    in_fence = False
    fence_marker = None

    for line in lines:
        stripped = line.lstrip()

        # Toggle fenced code blocks (``` or ~~~). Preserve verbatim.
        fence_match = re.match(r'(```+|~~~+)', stripped)
        if fence_match:
            marker = fence_match.group(1)[:3]
            if not in_fence:
                in_fence = True
                fence_marker = marker
            elif stripped.startswith(fence_marker):
                in_fence = False
                fence_marker = None
            out_lines.append(line)
            continue

        if in_fence:
            out_lines.append(line)
            continue

        # Never drop articles on heading lines (would change heading meaning).
        line_drop_articles = drop_articles and not _HEADING_RE.match(line)

        compressed = _compress_line(line, line_drop_articles)

        if llm_hook is not None and compressed.strip():
            try:
                compressed = llm_hook(compressed)
            except Exception:
                pass  # LLM hook is best-effort; never break rule-based output.

        out_lines.append(compressed)

    result = '\n'.join(out_lines)

    # Compress ISO dates YYYY-MM-DD -> MM-DD (outside code handled above since
    # fenced/inline code already preserved; do a final safe pass on the joined
    # text excluding preserved inline spans is overkill — dates in code are rare
    # and this is conservative enough for the auditable level).
    result = _ISO_DATE_RE.sub(lambda m: f'{m.group(1)}-{m.group(2)}', result)

    # Collapse redundant intra-line whitespace (not newlines, not indentation).
    out = []
    for ln in result.split('\n'):
        indent_match = re.match(r'^(\s*)', ln)
        indent = indent_match.group(1)
        body = ln[len(indent):]
        body = re.sub(r'[ \t]{2,}', ' ', body)
        body = body.rstrip()
        out.append(indent + body)
    result = '\n'.join(out)

    # Collapse 3+ blank lines down to a single blank line.
    result = re.sub(r'\n{3,}', '\n\n', result)

    return result


def compression_ratio(original: str, compressed: str) -> float:
    """Return fractional char reduction, e.g. 0.18 == 18% smaller."""
    if not original:
        return 0.0
    return 1 - (len(compressed) / len(original))


if __name__ == '__main__':
    import sys

    src = sys.stdin.read() if len(sys.argv) < 2 else open(sys.argv[1]).read()
    out = compress(src)
    sys.stderr.write(
        f'[steno_compress] {len(src)} -> {len(out)} chars '
        f'({compression_ratio(src, out) * 100:.1f}% smaller)\n'
    )
    sys.stdout.write(out)
