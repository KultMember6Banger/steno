"""Steno curate — the self-curating memory loop.

Ties the Steno runtime (compress / index / retrieve) together with the Vigil
auditor (contradiction gate / health scoring) into one pipeline that keeps a
memory directory healthy and retrieval health-weighted:

    compress (optional)  ->  gate (optional, Vigil)  ->  index  ->  score (Vigil)

Vigil is imported SOFTLY: every Vigil-dependent step degrades gracefully when
Vigil is not importable, so `curate --compress` (compress + index) works on a
bare Steno install, while the full loop lights up when Vigil is present.

Vigil discovery: a normal `import vigil` is tried first (pip-installed is the
expected path for full functionality); if that fails we also try a sibling
`../vigil/src` checkout next to this repo so a local dev layout Just Works.
"""

from __future__ import annotations

import sys
from pathlib import Path

import memcore
from steno_parser import parse_frontmatter as _steno_parse_frontmatter


# ---------------------------------------------------------------------------
# Vigil soft-import bridge
# ---------------------------------------------------------------------------
def _try_import_vigil():
    """Return the imported `vigil` package, or None if unavailable.

    Tries a normal import first (pip-installed). If that fails, inserts a
    sibling `../vigil/src` checkout onto sys.path and retries, so a local
    side-by-side dev layout works without installation.
    """
    try:
        import vigil  # noqa: F401
        return vigil
    except Exception:
        pass

    sibling = Path(__file__).resolve().parent.parent / 'vigil' / 'src'
    if sibling.is_dir():
        if str(sibling) not in sys.path:
            sys.path.insert(0, str(sibling))
        try:
            import vigil  # noqa: F401
            return vigil
        except Exception:
            return None
    return None


def vigil_available() -> bool:
    """True if the Vigil auditor can be imported (pip or sibling checkout)."""
    return _try_import_vigil() is not None


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------
def _is_already_steno(path: Path) -> bool:
    """True if the file's frontmatter already declares format: steno/steno-m."""
    try:
        content = path.read_text(encoding='utf-8')
    except Exception:
        return False
    meta, _ = _steno_parse_frontmatter(content)
    return meta.get('format') in ('steno', 'steno-m')


def _memory_files(memory_dir: Path) -> list[Path]:
    """Curatable memory files (.md), excluding MEMORY.md / README.md."""
    return [
        f for f in sorted(memory_dir.glob('*.md'))
        if f.name not in ('MEMORY.md', 'README.md')
    ]


def _compress_dir(memory_dir: Path) -> tuple[int, list[str]]:
    """Compress prose memories in place; skip files already in steno format.

    Returns (count_compressed, list_of_compressed_stems).
    """
    from steno_compress import compress
    import yaml

    compressed = 0
    names = []
    for f in _memory_files(memory_dir):
        if _is_already_steno(f):
            continue
        content = f.read_text(encoding='utf-8')
        meta, body = _steno_parse_frontmatter(content)
        compressed_body = compress(body, level='steno')

        if meta:
            meta = dict(meta)
            meta['format'] = 'steno'
            fm = yaml.safe_dump(
                meta, sort_keys=False, default_flow_style=False, allow_unicode=True
            ).strip()
            output = f'---\n{fm}\n---\n\n{compressed_body.lstrip(chr(10))}'
        else:
            output = compressed_body
        if not output.endswith('\n'):
            output += '\n'

        f.write_text(output, encoding='utf-8')
        compressed += 1
        names.append(f.stem)
    return compressed, names


def _gate_dir(memory_dir: Path, store_dir: Path, collection_name: str,
              changed_stems: set[str] | None, yes: bool) -> tuple[list[str], list[dict]]:
    """Run Vigil's pre-write contradiction gate on candidate files.

    For each candidate file, run pre_write_check against the EXISTING index. On
    a CRITICAL pre_write_conflict, the file is skipped from indexing (unless
    --yes). Returns (gated_stems, warnings) where warnings is a list of dicts.

    `changed_stems` limits the gate to new/changed files when known; None means
    gate every memory file.
    """
    vigil = _try_import_vigil()
    if vigil is None:
        return [], []
    from vigil.scanner import pre_write_check

    gated = []
    warnings = []
    for f in _memory_files(memory_dir):
        if changed_stems is not None and f.stem not in changed_stems:
            continue
        content = f.read_text(encoding='utf-8')
        _, body = _steno_parse_frontmatter(content)
        try:
            issues = pre_write_check(
                body, store_dir, source_file=f.stem,
                collection_name=collection_name,
            )
        except Exception:
            continue
        criticals = [i for i in issues
                     if i.category == 'pre_write_conflict' and i.severity == 'CRITICAL']
        if criticals:
            warnings.append({
                'file': f.stem,
                'conflicts': [
                    {'with': c.files[0] if c.files else '',
                     'message': c.message}
                    for c in criticals
                ],
            })
            if not yes:
                gated.append(f.stem)
    return gated, warnings


def curate(
    memory_dir: Path,
    compress: bool = False,
    gate: bool = False,
    store_dir: Path | None = None,
    collection_name: str | None = None,
    yes: bool = False,
    verbose: bool = True,
) -> dict:
    """Run the self-curating loop over a memory directory.

    Steps (each conditional / soft):
      1. compress (if compress): compress prose memories in place (skip steno).
      2. gate (if gate AND vigil): pre-write contradiction check vs the existing
         index; CRITICAL conflicts skip that file from indexing unless `yes`.
      3. index: build/update the shared ChromaDB index.
      4. score (if vigil): full_scan + compute/update health_scores so retrieval
         is health-weighted.

    Returns a summary dict: compressed, gated, indexed, scored, vigil_available,
    plus the raw index stats and any gate warnings.
    """
    from memory_index import build_index, DEFAULT_STORE_DIR

    memory_dir = Path(memory_dir)
    if store_dir is None:
        store_dir = DEFAULT_STORE_DIR
    store_dir = Path(store_dir)
    collection_name = memcore.resolve_collection(collection_name)
    have_vigil = vigil_available()

    def _log(msg):
        if verbose:
            print(msg)

    summary = {
        'compressed': 0,
        'gated': 0,
        'indexed': 0,
        'scored': 0,
        'vigil_available': have_vigil,
        'gate_warnings': [],
    }

    # --- 1. compress ---
    compressed_names: list[str] = []
    if compress:
        summary['compressed'], compressed_names = _compress_dir(memory_dir)
        _log(f'[curate] compressed {summary["compressed"]} prose file(s)')

    # --- 2. gate (soft) ---
    gated_stems: set[str] = set()
    if gate:
        if have_vigil:
            # Limit the gate to files that changed this run when we know them
            # (compressed files); otherwise gate all files.
            changed = set(compressed_names) if compress and compressed_names else None
            gated_list, warnings = _gate_dir(
                memory_dir, store_dir, collection_name, changed, yes)
            gated_stems = set(gated_list)
            summary['gated'] = len(gated_stems)
            summary['gate_warnings'] = warnings
            for w in warnings:
                status = 'SKIPPED' if w['file'] in gated_stems else 'allowed (--yes)'
                _log(f'[curate] CRITICAL contradiction in {w["file"]} -> {status}')
            _log(f'[curate] gated {summary["gated"]} file(s)')
        else:
            _log('[curate] vigil not installed — skipping contradiction gate')

    # --- 2b. quarantine gated files so the index step does not pick them up ---
    # build_index walks the directory; to skip a gated file we temporarily move
    # it aside, index, then restore it. This keeps build_index untouched.
    quarantined: list[tuple[Path, Path]] = []
    for stem in gated_stems:
        src = memory_dir / f'{stem}.md'
        if src.exists():
            dst = src.with_suffix('.md.gated')
            src.rename(dst)
            quarantined.append((src, dst))

    try:
        # --- 3. index ---
        stats = build_index(
            memory_dir, store_dir=store_dir, collection_name=collection_name)
        summary['indexed'] = stats.get('records_indexed', 0)
        summary['index_stats'] = stats
        _log(f'[curate] indexed {summary["indexed"]} record(s)')
    finally:
        for src, dst in quarantined:
            if dst.exists():
                dst.rename(src)

    # --- 4. score (soft) ---
    if have_vigil:
        try:
            from vigil.scanner import (
                full_scan, compute_health_scores, update_health_scores)
            results = full_scan(
                memory_dir, store_dir=store_dir, collection_name=collection_name)
            scores = compute_health_scores(results)
            summary['scored'] = update_health_scores(
                scores, store_dir, collection_name=collection_name)
            _log(f'[curate] scored {summary["scored"]} record(s) with health weights')
        except Exception as e:
            _log(f'[curate] health scoring skipped ({type(e).__name__}: {e})')
    else:
        _log('[curate] vigil not installed — skipping health scoring')

    return summary
