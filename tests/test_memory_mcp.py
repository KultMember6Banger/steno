"""Tests for the unified memory MCP server (memory_mcp.py).

Verifies the JSON-RPC handshake and that tools/list reflects Vigil availability:
  - Vigil PRESENT  -> Steno + Vigil tools advertised; Vigil tool callable.
  - Vigil ABSENT   -> only Steno tools advertised; Vigil tool returns an error.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, ROOT)

import memory_mcp as mm  # noqa: E402

STENO_TOOL_NAMES = {'memory_query', 'memory_index', 'memory_compress',
                    'memory_expand', 'memory_curate'}
VIGIL_TOOL_NAMES = {'memory_audit', 'memory_check', 'memory_fix', 'memory_health'}


def test_initialize_handshake():
    resp = mm.handle_request({'jsonrpc': '2.0', 'id': 1, 'method': 'initialize'})
    assert resp['result']['protocolVersion'] == '2024-11-05'
    assert resp['result']['serverInfo']['name'] == 'memory'


def test_notifications_get_no_response():
    assert mm.handle_request({'jsonrpc': '2.0', 'method': 'notifications/initialized'}) is None


def test_tools_list_always_includes_steno():
    resp = mm.handle_request({'jsonrpc': '2.0', 'id': 2, 'method': 'tools/list'})
    names = {t['name'] for t in resp['result']['tools']}
    assert STENO_TOOL_NAMES <= names


def test_tools_list_with_vigil_present(monkeypatch):
    monkeypatch.setattr(mm, 'vigil_available', lambda: True)
    resp = mm.handle_request({'jsonrpc': '2.0', 'id': 3, 'method': 'tools/list'})
    names = {t['name'] for t in resp['result']['tools']}
    assert STENO_TOOL_NAMES <= names
    assert VIGIL_TOOL_NAMES <= names


def test_tools_list_without_vigil_hides_vigil_tools(monkeypatch):
    monkeypatch.setattr(mm, 'vigil_available', lambda: False)
    resp = mm.handle_request({'jsonrpc': '2.0', 'id': 4, 'method': 'tools/list'})
    names = {t['name'] for t in resp['result']['tools']}
    assert STENO_TOOL_NAMES <= names
    assert not (VIGIL_TOOL_NAMES & names)


def test_calling_vigil_tool_when_absent_errors(monkeypatch):
    monkeypatch.setattr(mm, 'vigil_available', lambda: False)
    resp = mm.handle_request({
        'jsonrpc': '2.0', 'id': 5, 'method': 'tools/call',
        'params': {'name': 'memory_audit', 'arguments': {'memory_dir': '/x'}},
    })
    assert 'error' in resp
    assert 'unavailable' in resp['error']['message'].lower()


def test_steno_compress_tool_call_roundtrips():
    # memory_compress has no heavy deps — exercise the full tools/call path.
    resp = mm.handle_request({
        'jsonrpc': '2.0', 'id': 6, 'method': 'tools/call',
        'params': {'name': 'memory_compress',
                   'arguments': {'text': 'The service is currently running on the host.'}},
    })
    assert 'result' in resp
    assert resp['result']['content'][0]['type'] == 'text'
    import json
    payload = json.loads(resp['result']['content'][0]['text'])
    assert payload['original_chars'] > 0
    assert 'compressed' in payload


def test_unknown_tool_errors():
    resp = mm.handle_request({
        'jsonrpc': '2.0', 'id': 7, 'method': 'tools/call',
        'params': {'name': 'nope', 'arguments': {}},
    })
    assert 'error' in resp
