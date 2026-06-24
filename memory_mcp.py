#!/usr/bin/env python3
"""Unified memory MCP server — one stdio entry point for the whole memory system.

A single Model Context Protocol (protocolVersion "2024-11-05") stdio JSON-RPC
server that exposes the WHOLE Steno + Vigil memory stack behind one command.

Always-available Steno tools:
  - memory_query    : semantic search (with hybrid / mmr / budget options)
  - memory_index    : index / re-index a memory directory
  - memory_compress : compress prose text -> Steno notation
  - memory_expand   : expand Steno notation -> readable prose (best-effort)
  - memory_curate   : self-curating loop (compress -> gate -> index -> score)

Vigil auditor tools — exposed ONLY when Vigil is importable (soft import):
  - memory_audit    : run a full/selective health scan
  - memory_check    : pre-write contradiction gate for proposed text
  - memory_fix      : build a (dry-run) resolution plan from a scan
  - memory_health   : full scan + write health scores into ChromaDB

`tools/list` reflects what is ACTUALLY available: when Vigil cannot be imported,
the Vigil tools are not advertised (and calling them returns an error).

Vigil discovery mirrors `steno curate`: a normal `import vigil` first (pip
install is the expected path), then a sibling `../vigil/src` checkout.

Transport: newline-delimited JSON-RPC 2.0 (one object per line) on stdin/stdout;
diagnostics go to stderr only.

Run:
  python3 memory_mcp.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from steno_curate import _try_import_vigil  # soft Vigil import (pip or sibling)

PROTOCOL_VERSION = '2024-11-05'
SERVER_INFO = {'name': 'memory', 'version': '0.4.0'}


# ---------------------------------------------------------------------------
# Steno tool schemas (always available)
# ---------------------------------------------------------------------------
STENO_TOOLS = [
    {
        'name': 'memory_query',
        'description': (
            'Semantic search over the shared memory index (ChromaDB). Returns '
            'the top-K most relevant records, health-weighted when Vigil scores '
            'are present. Supports hybrid (BM25 fusion), MMR diversification, '
            'and token-budget knapsack selection.'
        ),
        'inputSchema': {
            'type': 'object',
            'properties': {
                'query': {'type': 'string', 'description': 'Natural-language search text'},
                'top_k': {'type': 'integer', 'description': 'Max results (default 10)'},
                'record_type': {'type': 'string', 'description': 'Filter by @V/@F/@T/@C/@A/@L'},
                'memory_type': {'type': 'string', 'description': 'Filter: user/feedback/project/reference'},
                'min_score': {'type': 'number', 'description': 'Min similarity (default 0.30)'},
                'hybrid': {'type': 'boolean', 'description': 'Fuse semantic + BM25 (RRF)'},
                'mmr': {'type': 'boolean', 'description': 'Diversify via Maximal Marginal Relevance'},
                'mmr_lambda': {'type': 'number', 'description': 'MMR relevance/diversity trade-off (default 0.5)'},
                'budget': {'type': 'integer', 'description': 'Token budget (knapsack selection)'},
                'store': {'type': 'string', 'description': 'ChromaDB store path'},
                'collection': {'type': 'string', 'description': 'Collection name (default agent_memory)'},
            },
            'required': ['query'],
        },
    },
    {
        'name': 'memory_index',
        'description': 'Index or re-index a directory of memory files into the shared ChromaDB store.',
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
        'name': 'memory_compress',
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
        'name': 'memory_expand',
        'description': 'Expand Steno notation back into readable prose (best-effort; not lossless).',
        'inputSchema': {
            'type': 'object',
            'properties': {
                'text': {'type': 'string', 'description': 'Steno text to expand'},
            },
            'required': ['text'],
        },
    },
    {
        'name': 'memory_curate',
        'description': (
            'Self-curating loop over a memory directory: compress -> gate '
            '(Vigil contradiction check) -> index -> health-score (Vigil). '
            'Vigil steps activate only when Vigil is importable.'
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
# Vigil tool schemas (only advertised when Vigil is importable)
# ---------------------------------------------------------------------------
VIGIL_TOOLS = [
    {
        'name': 'memory_audit',
        'description': (
            'Run a Vigil health scan over a memory directory: contradictions, '
            'duplicates, isolated entries, staleness, orphan refs, provenance. '
            'Returns the found issues as JSON.'
        ),
        'inputSchema': {
            'type': 'object',
            'properties': {
                'memory_dir': {'type': 'string', 'description': 'Directory of .md memory files'},
                'checks': {
                    'type': 'array',
                    'items': {'type': 'string',
                              'enum': ['contradictions', 'duplicates', 'isolated',
                                       'stale', 'orphans', 'provenance']},
                    'description': 'Subset of checks to run. Omit to run all.',
                },
                'store': {'type': 'string', 'description': 'ChromaDB store path'},
                'collection': {'type': 'string', 'description': 'Collection name (default agent_memory)'},
            },
            'required': ['memory_dir'],
        },
    },
    {
        'name': 'memory_check',
        'description': (
            'Pre-write contradiction gate. Given proposed new memory text, '
            'returns existing memories it would contradict or supersede. Call '
            'BEFORE writing a new memory.'
        ),
        'inputSchema': {
            'type': 'object',
            'properties': {
                'memory_dir': {'type': 'string', 'description': 'Directory (locates the store)'},
                'text': {'type': 'string', 'description': 'Proposed new memory text'},
                'source': {'type': 'string', 'description': 'Source file stem to exclude (optional)'},
                'store': {'type': 'string', 'description': 'ChromaDB store path'},
                'collection': {'type': 'string', 'description': 'Collection name (default agent_memory)'},
            },
            'required': ['memory_dir', 'text'],
        },
    },
    {
        'name': 'memory_fix',
        'description': (
            'Build a Vigil resolution PLAN from a health scan (dry run; never '
            'edits or deletes). Returns the planned actions (archive stale, '
            'dedupe, manual-edit orphans) as a readable plan + structured JSON.'
        ),
        'inputSchema': {
            'type': 'object',
            'properties': {
                'memory_dir': {'type': 'string', 'description': 'Directory of .md memory files'},
                'store': {'type': 'string', 'description': 'ChromaDB store path'},
                'collection': {'type': 'string', 'description': 'Collection name (default agent_memory)'},
            },
            'required': ['memory_dir'],
        },
    },
    {
        'name': 'memory_health',
        'description': (
            'Run a full health scan AND write per-file health scores into '
            'ChromaDB metadata so retrieval can deprioritize unhealthy memories. '
            'Returns issues, per-file scores, and records updated.'
        ),
        'inputSchema': {
            'type': 'object',
            'properties': {
                'memory_dir': {'type': 'string', 'description': 'Directory of .md memory files'},
                'store': {'type': 'string', 'description': 'ChromaDB store path'},
                'collection': {'type': 'string', 'description': 'Collection name (default agent_memory)'},
            },
            'required': ['memory_dir'],
        },
    },
]


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------
def vigil_available() -> bool:
    """True if the Vigil auditor can be imported."""
    return _try_import_vigil() is not None


def available_tools() -> list[dict]:
    """Steno tools always; Vigil tools only when importable."""
    tools = list(STENO_TOOLS)
    if vigil_available():
        tools = tools + list(VIGIL_TOOLS)
    return tools


# ---------------------------------------------------------------------------
# Steno tool implementations
# ---------------------------------------------------------------------------
def _store_or_default(store):
    from memory_index import DEFAULT_STORE_DIR
    return Path(store) if store else DEFAULT_STORE_DIR


def _tool_memory_query(args: dict) -> str:
    from memory_retrieval import query

    results = query(
        args['query'],
        top_k=int(args.get('top_k', 10)),
        record_type=args.get('record_type'),
        memory_type=args.get('memory_type'),
        min_score=float(args.get('min_score', 0.30)),
        store_dir=_store_or_default(args.get('store')),
        collection_name=args.get('collection'),
        token_budget=int(args['budget']) if args.get('budget') else None,
        hybrid=bool(args.get('hybrid', False)),
        mmr=bool(args.get('mmr', False)),
        mmr_lambda=float(args.get('mmr_lambda', 0.5)),
    )
    payload = [
        {'id': r.id, 'score': r.score, 'source_file': r.source_file,
         'record_type': r.record_type, 'memory_type': r.memory_type, 'text': r.text}
        for r in results
    ]
    return json.dumps({'query': args['query'], 'count': len(payload), 'results': payload}, indent=2)


def _tool_memory_index(args: dict) -> str:
    from memory_index import build_index

    stats = build_index(
        Path(args['memory_dir']),
        store_dir=_store_or_default(args.get('store')),
        rebuild=bool(args.get('rebuild', False)),
        collection_name=args.get('collection'),
    )
    return json.dumps(stats, indent=2)


def _tool_memory_compress(args: dict) -> str:
    from steno_compress import compress, compression_ratio

    text = args['text']
    level = args.get('level', 'steno')
    out = compress(text, level=level)
    return json.dumps({
        'level': level,
        'original_chars': len(text),
        'compressed_chars': len(out),
        'reduction_pct': round(compression_ratio(text, out) * 100, 1),
        'compressed': out,
    }, indent=2)


def _tool_memory_expand(args: dict) -> str:
    from steno_compress import expand

    text = args['text']
    out = expand(text)
    return json.dumps({'original_chars': len(text), 'expanded_chars': len(out),
                       'expanded': out}, indent=2)


def _tool_memory_curate(args: dict) -> str:
    from steno_curate import curate

    summary = curate(
        Path(args['memory_dir']),
        compress=bool(args.get('compress', False)),
        gate=bool(args.get('gate', False)),
        store_dir=_store_or_default(args.get('store')),
        collection_name=args.get('collection'),
        yes=bool(args.get('yes', False)),
        verbose=False,
    )
    return json.dumps(summary, indent=2)


# ---------------------------------------------------------------------------
# Vigil tool implementations (only reachable when Vigil imported)
# ---------------------------------------------------------------------------
def _vigil_store(args: dict, memory_dir: Path) -> Path:
    from vigil.indexer import default_store_dir
    store = args.get('store')
    return Path(store) if store else default_store_dir(memory_dir)


def _issue_to_dict(issue) -> dict:
    return {'severity': issue.severity, 'category': issue.category,
            'message': issue.message, 'files': issue.files, 'details': issue.details}


def _results_to_dict(results: dict) -> dict:
    return {cat: [_issue_to_dict(i) for i in issues] for cat, issues in results.items()}


def _tool_memory_audit(args: dict) -> str:
    from vigil.scanner import full_scan

    memory_dir = Path(args['memory_dir'])
    results = full_scan(
        memory_dir=memory_dir,
        store_dir=_vigil_store(args, memory_dir),
        checks=args.get('checks'),
        collection_name=args.get('collection'),
    )
    total = sum(len(v) for v in results.values())
    return json.dumps({'total_issues': total, 'issues': _results_to_dict(results)},
                      indent=2, default=str)


def _tool_memory_check(args: dict) -> str:
    from vigil.scanner import pre_write_check

    memory_dir = Path(args['memory_dir'])
    issues = pre_write_check(
        args.get('text', ''),
        store_dir=_vigil_store(args, memory_dir),
        source_file=args.get('source', ''),
        collection_name=args.get('collection'),
    )
    return json.dumps({'conflict_count': len(issues),
                       'conflicts': [_issue_to_dict(i) for i in issues]},
                      indent=2, default=str)


def _tool_memory_fix(args: dict) -> str:
    from vigil.scanner import full_scan
    from vigil.fix import build_plan, format_plan
    from vigil.config import VigilConfig

    memory_dir = Path(args['memory_dir'])
    store_dir = _vigil_store(args, memory_dir)
    coll = args.get('collection')
    config = VigilConfig.load(memory_dir)

    results = full_scan(memory_dir=memory_dir, store_dir=store_dir, collection_name=coll)
    actions = build_plan(results, memory_dir, store_dir, config, collection_name=coll)
    plan_text = format_plan(actions, applied=False,
                            archive_dir_name=getattr(config, 'archive_dir_name', 'archive'))
    payload = {
        'action_count': len(actions),
        'plan': plan_text,
        'actions': [
            {'category': a.category, 'action': a.action, 'target': a.target,
             'reason': a.reason, 'details': a.details}
            for a in actions
        ],
    }
    return json.dumps(payload, indent=2, default=str)


def _tool_memory_health(args: dict) -> str:
    from vigil.scanner import full_scan, compute_health_scores, update_health_scores

    memory_dir = Path(args['memory_dir'])
    store_dir = _vigil_store(args, memory_dir)
    coll = args.get('collection')

    results = full_scan(memory_dir=memory_dir, store_dir=store_dir, collection_name=coll)
    scores = compute_health_scores(results)
    n_updated = update_health_scores(scores, store_dir, collection_name=coll)
    total = sum(len(v) for v in results.values())
    return json.dumps({
        'total_issues': total,
        'records_scored': n_updated,
        'scores': {k: round(v, 3) for k, v in sorted(scores.items())},
        'issues': _results_to_dict(results),
    }, indent=2, default=str)


STENO_IMPLS = {
    'memory_query': _tool_memory_query,
    'memory_index': _tool_memory_index,
    'memory_compress': _tool_memory_compress,
    'memory_expand': _tool_memory_expand,
    'memory_curate': _tool_memory_curate,
}

VIGIL_IMPLS = {
    'memory_audit': _tool_memory_audit,
    'memory_check': _tool_memory_check,
    'memory_fix': _tool_memory_fix,
    'memory_health': _tool_memory_health,
}


def _impl_for(name: str):
    """Resolve a tool impl, honoring Vigil availability for Vigil tools."""
    if name in STENO_IMPLS:
        return STENO_IMPLS[name]
    if name in VIGIL_IMPLS and vigil_available():
        return VIGIL_IMPLS[name]
    return None


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

    if method and method.startswith('notifications/'):
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
        return _result(req_id, {'tools': available_tools()})

    if method == 'tools/call':
        name = params.get('name')
        arguments = params.get('arguments') or {}
        impl = _impl_for(name)
        if impl is None:
            return _error(req_id, -32602, f'Unknown or unavailable tool: {name}')
        try:
            text = impl(arguments)
            return _result(req_id, {'content': [{'type': 'text', 'text': text}]})
        except Exception as e:
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
            stdout.write(json.dumps(_error(None, -32700, 'Parse error')) + '\n')
            stdout.flush()
            continue

        resp = handle_request(req)
        if resp is not None:
            stdout.write(json.dumps(resp) + '\n')
            stdout.flush()


def main():
    serve()


if __name__ == '__main__':
    main()
