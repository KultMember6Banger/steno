#!/usr/bin/env python3
"""
Steno — compressed memory notation with RAG retrieval for AI agents.

Usage:
  steno index [--rebuild] MEMORY_DIR     Index/re-index memory files
  steno query "search text" [TOP_K]      Semantic search over memory
  steno stats                            Show index statistics
  steno parse [FILE_OR_DIR]              Parse and preview records (no indexing)

Filters for query:
  --type=@V          Filter by record type (@V, @F, @T, @C, @A, @L)
  --memory=feedback  Filter by memory type (user, feedback, project, reference)
  --min=0.6          Minimum similarity score

Environment variables:
  STENO_STORE        Path to ChromaDB storage (default: ./chroma_store)
  STENO_MODEL        Embedding model name (default: all-MiniLM-L6-v2)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from steno_parser import parse_directory, parse_file


def cmd_index(args: list[str]):
    from memory_index import build_index

    positional = [a for a in args if not a.startswith('--')]
    flags = [a for a in args if a.startswith('--')]

    if not positional:
        print('Error: memory directory required')
        print('Usage: steno index [--rebuild] MEMORY_DIR')
        sys.exit(1)

    memory_dir = Path(positional[0])
    rebuild = '--rebuild' in flags

    if not memory_dir.is_dir():
        print(f'Error: {memory_dir} is not a directory')
        sys.exit(1)

    print(f'Indexing {memory_dir}...')
    if rebuild:
        print('(rebuilding from scratch)')

    stats = build_index(memory_dir, rebuild=rebuild)
    print(f'Done in {stats["time_sec"]}s')
    print(f'  Indexed:   {stats["records_indexed"]} records')
    print(f'  Updated:   {stats["records_updated"]} files re-indexed')
    print(f'  Unchanged: {stats["records_unchanged"]} files skipped')
    print(f'  Deleted:   {stats["files_deleted"]} files cleaned up')
    print(f'  Total:     {stats["total_files"]} files tracked')


def cmd_query(args: list[str]):
    from memory_retrieval import query

    if not args or args[0].startswith('--'):
        print('Error: query text required')
        print('Usage: steno query "search text" [top_k] [--type=@V] [--memory=feedback]')
        sys.exit(1)

    query_text = args[0]
    top_k = 10
    record_type = None
    memory_type = None
    min_score = 0.55

    for arg in args[1:]:
        if arg.startswith('--type='):
            record_type = arg[7:]
        elif arg.startswith('--memory='):
            memory_type = arg[9:]
        elif arg.startswith('--min='):
            min_score = float(arg[6:])
        elif arg.isdigit():
            top_k = int(arg)

    results = query(
        query_text,
        top_k=top_k,
        record_type=record_type,
        memory_type=memory_type,
        min_score=min_score,
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
    import json
    from memory_index import DEFAULT_STORE_DIR, get_client, get_collection, _load_mtimes

    client = get_client()
    try:
        collection = get_collection(client)
        count = collection.count()
    except Exception:
        print('No index found. Run: steno index MEMORY_DIR')
        sys.exit(1)

    mtimes = _load_mtimes(DEFAULT_STORE_DIR)
    store_size = sum(f.stat().st_size for f in DEFAULT_STORE_DIR.rglob('*') if f.is_file())

    print(f'Steno Memory Index')
    print(f'  Records:    {count}')
    print(f'  Files:      {len(mtimes)}')
    print(f'  Store size: {store_size / 1024:.0f} KB')
    print(f'  Store path: {DEFAULT_STORE_DIR}')


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


def main():
    if len(sys.argv) < 2:
        print(__doc__.strip())
        sys.exit(0)

    cmd = sys.argv[1]
    args = sys.argv[2:]

    commands = {
        'index': cmd_index,
        'query': cmd_query,
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
