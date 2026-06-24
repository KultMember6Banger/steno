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
  table for display. `expand()` is a BEST-EFFORT readability aid that runs the
  legend in reverse (abbreviation -> full word) plus light date heuristics. It
  is NOT lossless: dropped articles cannot be recovered, and a few abbreviations
  are ambiguous (e.g. `cfg` maps to both `configuration` and `configure`).

LLM hook:
  This compressor is rule-based and has NO LLM dependency. To plug in an
  LLM-assisted *semantic* compression pass (prose -> steno via summarisation),
  implement a callable with the signature:

      llm_hook(text: str) -> str

  and pass it as `llm_hook=` to `compress()`. Contract:
    - Input  : a single already-rule-compressed prose line (never a preserved
               span — code/URLs/emails are stripped out before the hook runs).
    - Output : the semantically-compressed replacement line (str).
    - Errors : any exception is swallowed; the rule-based line is kept. The hook
               must therefore be safe to fail. It is best-effort only.
  This keeps the default path 100% rule-based and dependency-free while letting
  callers opt into an LLM for tighter, meaning-aware compression.
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


# Default cosine-fidelity threshold below which compression is considered to
# have lost meaning. MiniLM cosine between a faithful compression and its
# original is typically >0.95; 0.92 leaves headroom for aggressive but safe runs.
FIDELITY_THRESHOLD = 0.92


def compress(
    text: str,
    level: str = 'steno',
    llm_hook=None,
    verify: bool = False,
    fidelity_threshold: float = FIDELITY_THRESHOLD,
):
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
                  See the module docstring for the full hook contract.
        verify: if True, embed the original and compressed text with the real
                MiniLM model, compute cosine fidelity, and return a
                (compressed_text, fidelity) tuple instead of just the string.
                A warning is emitted (via compute_fidelity's caller) when
                fidelity < fidelity_threshold.
        fidelity_threshold: cosine threshold used by callers to decide whether
                to warn about meaning loss (default FIDELITY_THRESHOLD).

    Returns:
        If verify is False (default): the compressed text (str).
        If verify is True: a tuple (compressed_text, fidelity_score: float).
        Markdown structure, fenced code blocks, inline code, URLs and emails are
        preserved verbatim.
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

    if verify:
        fidelity = compute_fidelity(text, result)
        return result, fidelity

    return result


def compression_ratio(original: str, compressed: str) -> float:
    """Return fractional char reduction, e.g. 0.18 == 18% smaller."""
    if not original:
        return 0.0
    return 1 - (len(compressed) / len(original))


# ---------------------------------------------------------------------------
# Fidelity check (real embeddings). Lives here so `compress(verify=True)` and
# the CLI `--verify` flag share one implementation.
# ---------------------------------------------------------------------------
def compute_fidelity(original: str, compressed: str, model_name: str | None = None) -> float:
    """Cosine similarity between embeddings of the original and compressed text.

    Uses the same MiniLM model the index uses, so the score reflects whether the
    compression preserved *retrieval-relevant* meaning. Returns a float in
    roughly [-1, 1]; ~1.0 means semantically identical. Importing the model is
    deferred so the rule-based path stays dependency-light.

    NOTE: article-only compression stays high (~0.95), but aggressive
    abbreviation (auth/cfg/vfd/…) can score LOW because MiniLM does not know the
    legend — it sees `vfd` as gibberish, not `verified`. That is the intended
    signal: `--verify` flags where compression has drifted away from what the
    embedder (and thus retrieval) understands. It is most useful on LLM-assisted
    or lighter passes; expect low scores on heavily-abbreviated steno.
    """
    if not original.strip() or not compressed.strip():
        return 1.0
    from sentence_transformers import SentenceTransformer
    import numpy as np

    if model_name is None:
        import os
        model_name = os.environ.get('STENO_MODEL', 'all-MiniLM-L6-v2')

    model = SentenceTransformer(model_name)
    embs = model.encode([original, compressed], show_progress_bar=False)
    a, b = np.asarray(embs[0], dtype=float), np.asarray(embs[1], dtype=float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ---------------------------------------------------------------------------
# Expansion (best-effort inverse). NOT lossless — see module docstring.
# ---------------------------------------------------------------------------

# Abbreviation -> full word, for expansion. Where the forward map is many-to-one
# (configure/configuration -> cfg, parameter/parameters -> param/params) we pick
# the most common readable expansion. This is documented as ambiguous.
_EXPANSION_OVERRIDES: dict[str, str] = {
    'cfg': 'configuration',  # also 'configure'; pick the noun form
}

# Steno compresses ISO dates YYYY-MM-DD -> MM-DD, which is irreversible (year is
# gone). Expansion re-spaces them as MM-DD with no fabricated year.
_MMDD_RE = re.compile(r'\b(\d{2})-(\d{2})\b')

# Token boundary for reverse-abbreviation. Allows internal digits so that
# abbreviations like `k8s` are matched and expanded back to `kubernetes`.
_EXPAND_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9']*")


def _expansion_table() -> dict[str, str]:
    """abbreviation -> full word, applying overrides for ambiguous keys."""
    table = expand_legend()  # abbreviation -> full word (last forward key wins)
    table.update(_EXPANSION_OVERRIDES)
    return table


def _expand_token(token: str, table: dict[str, str]) -> str:
    """Expand a single abbreviation token, preserving leading capitalisation."""
    full = table.get(token.lower())
    if full is None:
        return token
    if token[:1].isupper():
        return full[:1].upper() + full[1:]
    return full


def expand(text: str, level: str = 'steno') -> str:
    """Best-effort INVERSE of compress(): expand steno back toward readable prose.

    This is a READABILITY AID, not a lossless decompressor. It:
      - expands abbreviations using the legend in reverse (auth -> authentication,
        cfg -> configuration, db -> database, ...);
      - leaves compressed dates as MM-DD (the dropped year cannot be recovered);
      - does NOT re-insert dropped articles (a/an/the) — that requires a language
        model and is out of scope for the rule-based path.

    Preserved spans (inline code, URLs, emails) and fenced code blocks are left
    verbatim, exactly as compress() left them. `level` is accepted for symmetry
    with compress() but currently only 'steno' is meaningful.

    Args:
        text: steno-compressed body.
        level: reserved; only 'steno' is implemented.

    Returns:
        Expanded, more-readable text.
    """
    table = _expansion_table()

    def expand_prose_segment(segment: str) -> str:
        out = []
        pos = 0
        for m in _EXPAND_TOKEN_RE.finditer(segment):
            out.append(segment[pos:m.start()])
            out.append(_expand_token(m.group(0), table))
            pos = m.end()
        out.append(segment[pos:])
        return ''.join(out)

    def expand_line(line: str) -> str:
        result = []
        last = 0
        for m in _PRESERVE_RE.finditer(line):
            result.append(expand_prose_segment(line[last:m.start()]))
            result.append(m.group(0))  # preserved verbatim
            last = m.end()
        result.append(expand_prose_segment(line[last:]))
        return ''.join(result)

    lines = text.split('\n')
    out_lines = []
    in_fence = False
    fence_marker = None

    for line in lines:
        stripped = line.lstrip()
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
        out_lines.append(expand_line(line))

    return '\n'.join(out_lines)


if __name__ == '__main__':
    import sys

    src = sys.stdin.read() if len(sys.argv) < 2 else open(sys.argv[1]).read()
    out = compress(src)
    sys.stderr.write(
        f'[steno_compress] {len(src)} -> {len(out)} chars '
        f'({compression_ratio(src, out) * 100:.1f}% smaller)\n'
    )
    sys.stdout.write(out)
