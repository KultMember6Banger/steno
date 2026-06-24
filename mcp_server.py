#!/usr/bin/env python3
"""
Steno MCP server — exposes Steno as native AI-agent tools over stdio JSON-RPC.

Implements the Model Context Protocol (protocolVersion "2024-11-05") with the
standard handlers: initialize, notifications/initialized, tools/list,
tools/call, ping.

Tools exposed:
  - steno_query    : semantic search over the memory index
  - steno_index    : index / re-index a memory directory
  - steno_compress : compress prose text -> Steno notation
  - memory_curate  : self-curating loop (compress -> gate -> index -> score)

Transport: newline-delimited JSON-RPC 2.0 messages on stdin/stdout (one JSON
object per line). Logs/diagnostics go to stderr only.

Run:
  python3 mcp_server.py

Note: steno_query / steno_index import chromadb + sentence-transformers lazily
(only when the tool is actually called) so the server can start and list tools
even in environments where those heavy deps aren't installed. steno_compress
has no heavy deps.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

PROTOCOL_VERSION = '2024-11-05'
SERVER_INFO = {'name': 'steno', 'version': '0.4.0'}

TOOLS = [
    {
        'name': 'steno_query',
        'description': (
            'Semantic search over the Steno memory index (ChromaDB). Returns the '
            'top-K most relevant memory records, health-weighted when Vigil is in use.'
        ),
        'inputSchema': {
            'type': 'object',
            'properties': {
                'query': {'type': 'string', 'description': 'Natural-language search text'},
                'top_k': {'type': 'integer', 'description': 'Max results (default 10)'},
                'record_type': {'type': 'string', 'description': 'Filter by @V/@F/@T/@C/@A/@L'},
                'memory_type': {'type': 'string', 'description': 'Filter: user/feedback/project/reference'},
                'min_score': {'type': 'number', 'description': 'Min similarity (default 0.30)'},
                'store': {'type': 'string', 'description': 'ChromaDB store path'},
                'collection': {'type': 'string', 'description': 'Collection name (default agent_memory)'},
            },
            'required': ['query'],
        },
    },
    {
        'name': 'steno_index',
        'description': 'Index or re-index a directory of memory files into the Steno/ChromaDB store.',
        'inputSchema': {
            'type': 'object',
            'properties': {
                'memory_dir': {'type': 'string', 'description': 'Directory of .md memory files'},
                'rebuild': {'type': 'boolean', 'description': 'Drop and rebuild from scratch'},
                'store': {'type': 'string', 'description': 'ChromaDB store path'},
                'collection': {'type': 'string', 'description': 'Collection name (default agent_memory)'},
            },
            'required': ['memory_dir'],
        },
    },
    {
        'name': 'steno_compress',
        'description': 'Compress prose text into human-auditable Steno notation (rule-based, lossy).',
        'inputSchema': {
            'type': 'object',
            'properties': {
                'text': {'type': 'string', 'description': 'Prose text to compress'},
                'level': {'type': 'string', 'description': "Compression level (default 'steno')"},
            },
            'required': ['text'],
        },
    },
    {
        'name': 'memory_curate',
        'description': (
            'Self-curating loop over a memory directory: compress -> gate '
            '(Vigil contradiction check) -> index -> health-score (Vigil). '
            'Vigil steps are soft and activate only when Vigil is importable.'
        ),
        'inputSchema': {
            'type': 'object',
            'properties': {
                'memory_dir': {'type': 'string', 'description': 'Directory of .md memory files'},
                'compress': {'type': 'boolean', 'description': 'Compress prose memories in place first'},
                'gate': {'type': 'boolean', 'description': 'Run Vigil pre-write contradiction gate'},
                'yes': {'type': 'boolean', 'description': 'Index gate-flagged files anyway'},
                'store': {'type': 'string', 'description': 'ChromaDB store path'},
                'collection': {'type': 'string', 'description': 'Collection name (default agent_memory)'},
            },
            'required': ['memory_dir'],
        },
    },
]


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------
def _tool_steno_query(args: dict) -> str:
    from memory_retrieval import query
    from memory_index import DEFAULT_STORE_DIR

    store = args.get('store')
    results = query(
        args['query'],
        top_k=int(args.get('top_k', 10)),
        record_type=args.get('record_type'),
        memory_type=args.get('memory_type'),
        min_score=float(args.get('min_score', 0.30)),
        store_dir=Path(store) if store else DEFAULT_STORE_DIR,
        collection_name=args.get('collection'),
    )
    payload = [
        {
            'id': r.id,
            'score': r.score,
            'source_file': r.source_file,
            'record_type': r.record_type,
            'memory_type': r.memory_type,
            'text': r.text,
        }
        for r in results
    ]
    return json.dumps({'query': args['query'], 'count': len(payload), 'results': payload}, indent=2)


def _tool_steno_index(args: dict) -> str:
    from memory_index import build_index, DEFAULT_STORE_DIR

    store = args.get('store')
    stats = build_index(
        Path(args['memory_dir']),
        store_dir=Path(store) if store else DEFAULT_STORE_DIR,
        rebuild=bool(args.get('rebuild', False)),
        collection_name=args.get('collection'),
    )
    return json.dumps(stats, indent=2)


def _tool_steno_compress(args: dict) -> str:
    from steno_compress import compress, compression_ratio

    text = args['text']
    level = args.get('level', 'steno')
    out = compress(text, level=level)
    return json.dumps(
        {
            'level': level,
            'original_chars': len(text),
            'compressed_chars': len(out),
            'reduction_pct': round(compression_ratio(text, out) * 100, 1),
            'compressed': out,
        },
        indent=2,
    )


def _tool_memory_curate(args: dict) -> str:
    from steno_curate import curate
    from memory_index import DEFAULT_STORE_DIR

    store = args.get('store')
    summary = curate(
        Path(args['memory_dir']),
        compress=bool(args.get('compress', False)),
        gate=bool(args.get('gate', False)),
        store_dir=Path(store) if store else DEFAULT_STORE_DIR,
        collection_name=args.get('collection'),
        yes=bool(args.get('yes', False)),
        verbose=False,
    )
    return json.dumps(summary, indent=2)


TOOL_IMPLS = {
    'steno_query': _tool_steno_query,
    'steno_index': _tool_steno_index,
    'steno_compress': _tool_steno_compress,
    'memory_curate': _tool_memory_curate,
}


# ---------------------------------------------------------------------------
# JSON-RPC plumbing
# ---------------------------------------------------------------------------
def _result(req_id, result):
    return {'jsonrpc': '2.0', 'id': req_id, 'result': result}


def _error(req_id, code, message):
    return {'jsonrpc': '2.0', 'id': req_id, 'error': {'code': code, 'message': message}}


def handle_request(req: dict):
    """Handle one JSON-RPC request. Returns a response dict, or None for notifications."""
    method = req.get('method')
    req_id = req.get('id')
    params = req.get('params') or {}

    # Notifications (no id) — never reply.
    if method == 'notifications/initialized' or (method and method.startswith('notifications/')):
        return None

    if method == 'initialize':
        return _result(req_id, {
            'protocolVersion': PROTOCOL_VERSION,
            'capabilities': {'tools': {}},
            'serverInfo': SERVER_INFO,
        })

    if method == 'ping':
        return _result(req_id, {})

    if method == 'tools/list':
        return _result(req_id, {'tools': TOOLS})

    if method == 'tools/call':
        name = params.get('name')
        arguments = params.get('arguments') or {}
        impl = TOOL_IMPLS.get(name)
        if impl is None:
            return _error(req_id, -32602, f'Unknown tool: {name}')
        try:
            text = impl(arguments)
            return _result(req_id, {'content': [{'type': 'text', 'text': text}]})
        except Exception as e:  # surface tool errors as isError content
            return _result(req_id, {
                'content': [{'type': 'text', 'text': f'Error: {type(e).__name__}: {e}'}],
                'isError': True,
            })

    if req_id is None:
        return None
    return _error(req_id, -32601, f'Method not found: {method}')


def serve(stdin=None, stdout=None):
    """Run the stdio JSON-RPC loop (one JSON object per line)."""
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout

    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            resp = _error(None, -32700, 'Parse error')
            stdout.write(json.dumps(resp) + '\n')
            stdout.flush()
            continue

        resp = handle_request(req)
        if resp is not None:
            stdout.write(json.dumps(resp) + '\n')
            stdout.flush()


if __name__ == '__main__':
    serve()
