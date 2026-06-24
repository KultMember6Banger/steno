#!/usr/bin/env python3
"""
Steno — compressed memory notation with RAG retrieval for AI agents.

Usage:
  steno index [--rebuild] MEMORY_DIR     Index/re-index memory files
  steno query "search text" [TOP_K]      Semantic search over memory
  steno compress FILE [--level steno] [--write]   Compress prose -> steno
  steno stats                            Show index statistics
  steno parse [FILE_OR_DIR]              Parse and preview records (no indexing)

Filters for query:
  --type=@V          Filter by record type (@V, @F, @T, @C, @A, @L)
  --memory=feedback  Filter by memory type (user, feedback, project, reference)
  --min=0.3          Minimum similarity score (default 0.30)

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
    )

    print(f'\nQuery: "{query_text}"')
    print(f'Results: {len(results)}\n')

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


def cmd_compress(args: list[str]):
    from steno_compress import compress, compression_ratio
    from steno_parser import parse_frontmatter
    import yaml

    # Support both "--level steno" (space-separated) and "--level=steno".
    level = 'steno'
    write = False
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
        elif a.startswith('--'):
            i += 1  # ignore unknown flags
        else:
            if file_arg is None:
                file_arg = a
            i += 1

    if file_arg is None:
        print('Error: file required')
        print('Usage: steno compress FILE [--level steno] [--write]')
        sys.exit(1)

    target = Path(file_arg)
    if not target.is_file():
        print(f'Error: {target} not found')
        sys.exit(1)

    content = target.read_text(encoding='utf-8')
    meta, body = parse_frontmatter(content)

    compressed_body = compress(body, level=level)

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
        print(f'Compressed {target} in place')
        print(f'  {len(content)} -> {len(output)} chars ({reduction:.1f}% smaller)')
    else:
        sys.stdout.write(output)
        sys.stderr.write(
            f'\n[steno compress] {len(content)} -> {len(output)} chars '
            f'({reduction:.1f}% smaller)\n'
        )


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
