#!/usr/bin/env python3
"""
Steno — compressed memory notation with RAG retrieval for AI agents.

Usage:
  steno index [--rebuild] MEMORY_DIR     Index/re-index memory files
  steno query "search text" [TOP_K]      Semantic search over memory
  steno compress FILE [--level steno] [--write]   Compress prose -> steno
  steno expand FILE [--write]            Expand steno -> readable prose (best-effort)
  steno emit JSON_FILE [--scope NAME]    Emit Steno-M from structured JSON records
  steno curate MEMORY_DIR [--compress] [--gate] [--yes]   Self-curating loop
  steno stats                            Show index statistics
  steno parse [FILE_OR_DIR]              Parse and preview records (no indexing)

Filters for query:
  --type=@V          Filter by record type (@V, @F, @T, @C, @A, @L)
  --memory=feedback  Filter by memory type (user, feedback, project, reference)
  --min=0.3          Minimum similarity score (default 0.30)

Retrieval tuning for query:
  --budget=2000      Return best SET of results fitting a token budget (knapsack)
  --hybrid           Fuse semantic + BM25 keyword search (RRF) — better exact-token recall
  --mmr              Maximal Marginal Relevance re-ranking to diversify results
  --mmr-lambda=0.5   MMR relevance/diversity trade-off (1=relevance, 0=diversity)

Compression options:
  --level steno|steno-m   Compression level (steno default; steno-m = structured emit)
  --verify                Embed original vs compressed, report cosine fidelity, warn if low
  --health-aware --store S  Compress LOW-health memories aggressively, keep HIGH-health verbatim

Shared store / collection (for Vigil integration):
  --store=PATH       ChromaDB storage path (index/query/stats)
  --collection=NAME  ChromaDB collection name (default: agent_memory)

Environment variables:
  STENO_STORE        Path to ChromaDB storage (default: ./chroma_store)
  MEMORY_STORE       Fallback store path if STENO_STORE unset
  MEMORY_COLLECTION  ChromaDB collection name (default: agent_memory)
  STENO_MODEL        Embedding model name (default: all-MiniLM-L6-v2)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from steno_parser import parse_directory, parse_file


def _get_flag_value(flags: list[str], *names: str) -> str | None:
    """Return the value of --name=VALUE for the first matching name, else None."""
    for flag in flags:
        for name in names:
            prefix = name + '='
            if flag.startswith(prefix):
                return flag[len(prefix):]
    return None


def cmd_index(args: list[str]):
    from memory_index import build_index, DEFAULT_STORE_DIR

    positional = [a for a in args if not a.startswith('--')]
    flags = [a for a in args if a.startswith('--')]

    if not positional:
        print('Error: memory directory required')
        print('Usage: steno index [--rebuild] MEMORY_DIR [--store=PATH] [--collection=NAME]')
        sys.exit(1)

    memory_dir = Path(positional[0])
    rebuild = '--rebuild' in flags
    store_val = _get_flag_value(flags, '--store')
    store_dir = Path(store_val) if store_val else DEFAULT_STORE_DIR
    collection_name = _get_flag_value(flags, '--collection')

    if not memory_dir.is_dir():
        print(f'Error: {memory_dir} is not a directory')
        sys.exit(1)

    print(f'Indexing {memory_dir}...')
    if rebuild:
        print('(rebuilding from scratch)')

    stats = build_index(
        memory_dir,
        store_dir=store_dir,
        rebuild=rebuild,
        collection_name=collection_name,
    )
    print(f'Done in {stats["time_sec"]}s')
    print(f'  Indexed:   {stats["records_indexed"]} records')
    print(f'  Updated:   {stats["records_updated"]} files re-indexed')
    print(f'  Unchanged: {stats["records_unchanged"]} files skipped')
    print(f'  Deleted:   {stats["files_deleted"]} files cleaned up')
    print(f'  Total:     {stats["total_files"]} files tracked')


def cmd_query(args: list[str]):
    from memory_retrieval import query
    from memory_index import DEFAULT_STORE_DIR

    if not args or args[0].startswith('--'):
        print('Error: query text required')
        print('Usage: steno query "search text" [top_k] [--type=@V] [--memory=feedback]')
        sys.exit(1)

    query_text = args[0]
    top_k = 10
    record_type = None
    memory_type = None
    min_score = 0.30  # unified default (matches memory_retrieval.query)
    store_dir = DEFAULT_STORE_DIR
    collection_name = None
    token_budget = None
    hybrid = False
    mmr = False
    mmr_lambda = 0.5

    for arg in args[1:]:
        if arg.startswith('--type='):
            record_type = arg[7:]
        elif arg.startswith('--memory='):
            memory_type = arg[9:]
        elif arg.startswith('--min='):
            min_score = float(arg[6:])
        elif arg.startswith('--store='):
            store_dir = Path(arg[len('--store='):])
        elif arg.startswith('--collection='):
            collection_name = arg[len('--collection='):]
        elif arg.startswith('--budget='):
            token_budget = int(arg[len('--budget='):])
        elif arg == '--hybrid':
            hybrid = True
        elif arg == '--mmr':
            mmr = True
        elif arg.startswith('--mmr-lambda='):
            mmr_lambda = float(arg[len('--mmr-lambda='):])
        elif arg.isdigit():
            top_k = int(arg)

    results = query(
        query_text,
        top_k=top_k,
        record_type=record_type,
        memory_type=memory_type,
        min_score=min_score,
        store_dir=store_dir,
        collection_name=collection_name,
        token_budget=token_budget,
        hybrid=hybrid,
        mmr=mmr,
        mmr_lambda=mmr_lambda,
    )

    print(f'\nQuery: "{query_text}"')
    modes = []
    if hybrid:
        modes.append('hybrid')
    if mmr:
        modes.append(f'mmr(λ={mmr_lambda})')
    if token_budget:
        modes.append(f'budget={token_budget}')
    mode_str = f' [{", ".join(modes)}]' if modes else ''
    print(f'Results: {len(results)}{mode_str}\n')

    if token_budget:
        from memory_retrieval import _estimate_tokens
        used = sum(_estimate_tokens(r.text) for r in results)
        print(f'Token budget: {used}/{token_budget} estimated tokens used\n')

    for r in results:
        print(f'[{r.score:.4f}] {r.id}')
        print(f'  source: {r.source_file} | type: {r.record_type} | memory: {r.memory_type}')
        preview = r.text[:150].replace('\n', ' ')
        print(f'  {preview}')
        print()


def cmd_stats(args: list[str]):
    from memory_index import DEFAULT_STORE_DIR, COLLECTION_NAME, get_client, get_collection, _load_mtimes

    flags = [a for a in args if a.startswith('--')]
    store_val = _get_flag_value(flags, '--store')
    store_dir = Path(store_val) if store_val else DEFAULT_STORE_DIR
    collection_name = _get_flag_value(flags, '--collection') or COLLECTION_NAME

    client = get_client(store_dir)
    try:
        collection = get_collection(client, collection_name)
        count = collection.count()
    except Exception:
        print('No index found. Run: steno index MEMORY_DIR')
        sys.exit(1)

    mtimes = _load_mtimes(store_dir)
    store_size = sum(f.stat().st_size for f in store_dir.rglob('*') if f.is_file())

    print(f'Steno Memory Index')
    print(f'  Records:    {count}')
    print(f'  Files:      {len(mtimes)}')
    print(f'  Collection: {collection_name}')
    print(f'  Store size: {store_size / 1024:.0f} KB')
    print(f'  Store path: {store_dir}')


def cmd_parse(args: list[str]):
    if not args:
        print('Error: file or directory required')
        print('Usage: steno parse FILE_OR_DIR')
        sys.exit(1)

    target = Path(args[0])

    if target.is_file():
        records = parse_file(target)
    elif target.is_dir():
        records = parse_directory(target)
    else:
        print(f'Error: {target} not found')
        sys.exit(1)

    formats = {}
    for r in records:
        fmt = r.metadata.get('format', 'unknown')
        formats[fmt] = formats.get(fmt, 0) + 1

    print(f'Parsed {len(records)} records')
    print(f'  Formats: {formats}')
    print(f'\nFirst 10 records:')
    for r in records[:10]:
        preview = r.text[:100].replace('\n', ' ')
        print(f'  [{r.record_type}] {r.id}')
        print(f'    {preview}')


def _health_for_source(store_dir: Path, collection_name: str | None, source_file: str):
    """Return the (min) health_score for a source file's records, or None.

    Health-aware compression policy reads the *minimum* health across a file's
    records so that any decayed/contradicted memory pulls the whole file toward
    aggressive compression. Returns None if the store/collection/file is absent.
    """
    from memory_index import COLLECTION_NAME, get_client, get_collection
    if collection_name is None:
        collection_name = COLLECTION_NAME
    try:
        client = get_client(store_dir)
        collection = get_collection(client, collection_name)
        got = collection.get(where={'source_file': source_file}, include=['metadatas'])
    except Exception:
        return None
    metas = got.get('metadatas') or []
    healths = [float(m.get('health_score', 1.0)) for m in metas if m is not None]
    if not healths:
        return None
    return min(healths)


def cmd_compress(args: list[str]):
    from steno_compress import compress, compression_ratio, FIDELITY_THRESHOLD
    from steno_parser import parse_frontmatter, to_steno_m
    from memory_index import DEFAULT_STORE_DIR
    import json
    import yaml

    # Support both "--level steno" (space-separated) and "--level=steno".
    level = 'steno'
    write = False
    verify = False
    health_aware = False
    store_dir = DEFAULT_STORE_DIR
    collection_name = None
    file_arg = None
    i = 0
    while i < len(args):
        a = args[i]
        if a == '--level':
            if i + 1 < len(args) and not args[i + 1].startswith('--'):
                level = args[i + 1]
                i += 2
                continue
            i += 1
        elif a.startswith('--level='):
            level = a[len('--level='):]
            i += 1
        elif a == '--write':
            write = True
            i += 1
        elif a == '--verify':
            verify = True
            i += 1
        elif a == '--health-aware':
            health_aware = True
            i += 1
        elif a.startswith('--store='):
            store_dir = Path(a[len('--store='):])
            i += 1
        elif a == '--store':
            if i + 1 < len(args):
                store_dir = Path(args[i + 1])
                i += 2
                continue
            i += 1
        elif a.startswith('--collection='):
            collection_name = a[len('--collection='):]
            i += 1
        elif a.startswith('--'):
            i += 1  # ignore unknown flags
        else:
            if file_arg is None:
                file_arg = a
            i += 1

    if file_arg is None:
        print('Error: file required')
        print('Usage: steno compress FILE [--level steno] [--write] [--verify] [--health-aware --store S]')
        sys.exit(1)

    target = Path(file_arg)
    if not target.is_file():
        print(f'Error: {target} not found')
        sys.exit(1)

    # --level steno-m: treat the input file as JSON structured records and emit
    # Steno-M. (See `steno emit` for the same path with explicit --scope.)
    if level == 'steno-m':
        raw = json.loads(target.read_text(encoding='utf-8'))
        records = raw.get('records', raw) if isinstance(raw, dict) else raw
        scope = raw.get('scope') if isinstance(raw, dict) else None
        output = to_steno_m(records, scope=scope)
        if write:
            target.write_text(output, encoding='utf-8')
            print(f'Emitted Steno-M to {target} ({len(records)} records)')
        else:
            sys.stdout.write(output)
        return

    content = target.read_text(encoding='utf-8')
    meta, body = parse_frontmatter(content)

    # Health-aware policy: aggressive (article-dropping 'steno') for LOW-health or
    # missing memories; HIGH-health (>=0.7) memories stay closer to verbatim
    # (abbreviation-only via a non-'steno' level that keeps articles).
    effective_level = level
    health_note = ''
    if health_aware:
        health = _health_for_source(store_dir, collection_name, target.stem)
        if health is None:
            effective_level = 'steno'
            health_note = ' (health-aware: no health found -> aggressive)'
        elif health >= 0.7:
            effective_level = 'steno-light'  # non-'steno' => keeps articles, abbreviations only
            health_note = f' (health-aware: health={health:.2f} HIGH -> conservative)'
        else:
            effective_level = 'steno'
            health_note = f' (health-aware: health={health:.2f} LOW -> aggressive)'

    if verify:
        compressed_body, fidelity = compress(body, level=effective_level, verify=True)
    else:
        compressed_body = compress(body, level=effective_level)
        fidelity = None

    # Preserve frontmatter, set format: steno.
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

    reduction = compression_ratio(content, output) * 100

    if write:
        target.write_text(output, encoding='utf-8')
        print(f'Compressed {target} in place{health_note}')
        print(f'  {len(content)} -> {len(output)} chars ({reduction:.1f}% smaller)')
    else:
        sys.stdout.write(output)
        sys.stderr.write(
            f'\n[steno compress] {len(content)} -> {len(output)} chars '
            f'({reduction:.1f}% smaller){health_note}\n'
        )

    if fidelity is not None:
        msg = f'[steno compress] semantic fidelity (cosine): {fidelity:.4f}\n'
        sys.stderr.write(msg)
        if fidelity < FIDELITY_THRESHOLD:
            sys.stderr.write(
                f'[steno compress] WARNING: fidelity {fidelity:.4f} below '
                f'threshold {FIDELITY_THRESHOLD} — possible meaning loss.\n'
            )


def cmd_expand(args: list[str]):
    from steno_compress import expand
    from steno_parser import parse_frontmatter
    import yaml

    write = False
    file_arg = None
    for a in args:
        if a == '--write':
            write = True
        elif a.startswith('--'):
            continue
        elif file_arg is None:
            file_arg = a

    if file_arg is None:
        print('Error: file required')
        print('Usage: steno expand FILE [--write]')
        sys.exit(1)

    target = Path(file_arg)
    if not target.is_file():
        print(f'Error: {target} not found')
        sys.exit(1)

    content = target.read_text(encoding='utf-8')
    meta, body = parse_frontmatter(content)
    expanded_body = expand(body)

    if meta:
        meta = dict(meta)
        meta.pop('format', None)  # no longer steno after expansion
        fm = yaml.safe_dump(
            meta, sort_keys=False, default_flow_style=False, allow_unicode=True
        ).strip()
        output = f'---\n{fm}\n---\n\n{expanded_body.lstrip(chr(10))}'
    else:
        output = expanded_body

    if not output.endswith('\n'):
        output += '\n'

    if write:
        target.write_text(output, encoding='utf-8')
        print(f'Expanded {target} in place (best-effort readability aid)')
    else:
        sys.stdout.write(output)
        sys.stderr.write(
            '\n[steno expand] best-effort: abbreviations expanded; dropped '
            'articles are NOT restored (not lossless).\n'
        )


def cmd_curate(args: list[str]):
    """Self-curating loop: compress -> gate (Vigil) -> index -> score (Vigil).

    Usage: steno curate MEMORY_DIR [--compress] [--gate] [--store S]
           [--collection C] [--yes]

    --compress  compress prose memories in place first (skip steno files)
    --gate      run Vigil's pre-write contradiction gate; CRITICAL conflicts
                skip that file from indexing unless --yes
    --yes       allow flagged files through the gate (index them anyway)

    Vigil steps (gate + health scoring) are soft: they activate only when Vigil
    is importable (pip-installed, or a sibling ../vigil/src checkout). Without
    Vigil, curate runs compress + index only.
    """
    from steno_curate import curate
    from memory_index import DEFAULT_STORE_DIR

    positional = [a for a in args if not a.startswith('--')]
    flags = [a for a in args if a.startswith('--')]

    if not positional:
        print('Error: memory directory required')
        print('Usage: steno curate MEMORY_DIR [--compress] [--gate] '
              '[--store=PATH] [--collection=NAME] [--yes]')
        sys.exit(1)

    memory_dir = Path(positional[0])
    if not memory_dir.is_dir():
        print(f'Error: {memory_dir} is not a directory')
        sys.exit(1)

    do_compress = '--compress' in flags
    do_gate = '--gate' in flags
    yes = '--yes' in flags
    store_val = _get_flag_value(flags, '--store')
    store_dir = Path(store_val) if store_val else DEFAULT_STORE_DIR
    collection_name = _get_flag_value(flags, '--collection')

    print(f'Curating {memory_dir}...')
    summary = curate(
        memory_dir,
        compress=do_compress,
        gate=do_gate,
        store_dir=store_dir,
        collection_name=collection_name,
        yes=yes,
    )

    print('\n--- curate summary ---')
    print(f'  compressed:      {summary["compressed"]}')
    print(f'  gated:           {summary["gated"]}')
    print(f'  indexed:         {summary["indexed"]}')
    print(f'  scored:          {summary["scored"]}')
    print(f'  vigil_available: {summary["vigil_available"]}')


def cmd_emit(args: list[str]):
    """Emit Steno-M text from a JSON file of structured records.

    JSON input shape (two accepted forms):
      {"scope": "proj", "records": [{"type": "@T", "fields": ["svc", "active"]}]}
      [{"type": "@T", "fields": ["svc", "active"]}, ...]   (no scope)
    """
    from steno_parser import to_steno_m
    import json

    scope = None
    file_arg = None
    i = 0
    while i < len(args):
        a = args[i]
        if a == '--scope':
            if i + 1 < len(args):
                scope = args[i + 1]
                i += 2
                continue
            i += 1
        elif a.startswith('--scope='):
            scope = a[len('--scope='):]
            i += 1
        elif a.startswith('--'):
            i += 1
        else:
            if file_arg is None:
                file_arg = a
            i += 1

    if file_arg is None:
        print('Error: JSON file required')
        print('Usage: steno emit JSON_FILE [--scope NAME]')
        sys.exit(1)

    target = Path(file_arg)
    if not target.is_file():
        print(f'Error: {target} not found')
        sys.exit(1)

    raw = json.loads(target.read_text(encoding='utf-8'))
    if isinstance(raw, dict):
        records = raw.get('records', [])
        if scope is None:
            scope = raw.get('scope')
    else:
        records = raw

    output = to_steno_m(records, scope=scope)
    sys.stdout.write(output)


def main():
    if len(sys.argv) < 2:
        print(__doc__.strip())
        sys.exit(0)

    cmd = sys.argv[1]
    args = sys.argv[2:]

    commands = {
        'index': cmd_index,
        'query': cmd_query,
        'compress': cmd_compress,
        'expand': cmd_expand,
        'emit': cmd_emit,
        'curate': cmd_curate,
        'stats': cmd_stats,
        'parse': cmd_parse,
    }

    if cmd in ('-h', '--help', 'help'):
        print(__doc__.strip())
    elif cmd in commands:
        commands[cmd](args)
    else:
        print(f'Unknown command: {cmd}')
        print(f'Available: {", ".join(commands.keys())}')
        sys.exit(1)


if __name__ == '__main__':
    main()
