# Steno

Compressed memory notation with RAG retrieval for AI agents.

Steno solves the AI memory problem: agents accumulate knowledge across sessions, but loading everything into context every time is expensive, noisy, and causes drift. Steno compresses memories into a dense notation format and retrieves only what's relevant using semantic search.

## The Problem

AI coding agents (Claude Code, Cursor, Copilot) build up memory files over time — user preferences, project context, past decisions, feedback. The default approach is brute-force: load all memory into every session. This wastes tokens, pollutes context with irrelevant information, and causes the agent to act on stale facts.

## The Solution

**Two-tier notation:**
- **Steno** — human-auditable compressed format. Drop articles, abbreviate common terms, use key-value pairs. Readable by humans, efficient for AI.
- **Steno-M** — AI-only format. Fixed schemas, positional fields, no labels. Maximum density for machine-to-machine communication.

**RAG retrieval:**
- Parse memory files into structured records
- Embed with a lightweight model (all-MiniLM-L6-v2, 80MB, runs on CPU)
- Store in ChromaDB (local, no server needed)
- Query semantically — only relevant memories enter the context window

## Quick Start

```bash
# Clone
git clone https://github.com/KultMember6Banger/steno.git
cd steno

# Install (pip-installable — provides the `steno` console script)
python3 -m venv .venv
source .venv/bin/activate
pip install .            # or: pip install -e .   (editable)
# (or, without packaging: pip install -r requirements.txt and use `python steno.py ...`)

# Index your memory files
steno index ./examples

# Search
steno query "database testing rules" 5

# Compress prose into Steno notation
steno compress notes.md

# Check index stats
steno stats
```

Requires Python 3.9+. (The code uses `from __future__ import annotations` so the
PEP-604 `X | None` type hints work on 3.9.)

## Writing Steno Memory Files

Memory files are markdown with YAML frontmatter:

```markdown
---
name: Integration Tests Must Hit Real DB
description: No mocking database in integration tests
type: feedback
format: steno
---

rule: integration tests connect to real PostgreSQL, never mock DB layer

**Why:** Q1 migration failure — mocked tests passed but prod migration broke.

**How to apply:**
  tests/integration/ → always uses test_db fixture (real PostgreSQL)
  tests/unit/ → mocks are fine (testing logic, not persistence)
```

### Memory Types

| Type | What | When to Save |
|---|---|---|
| `user` | Role, goals, preferences | When you learn about the user |
| `feedback` | Corrections, confirmed approaches | When the user corrects or validates |
| `project` | Ongoing work, decisions, deadlines | When you learn project context |
| `reference` | Pointers to external resources | When you discover useful external info |

### Steno Format Rules

1. Drop articles (a, the, an)
2. Abbreviate common terms: `verified` → `vfd`, `authentication` → `auth`, `configuration` → `cfg`
3. Compress dates: `2026-04-11` → `04-11`
4. Key-value for metadata: `key: value`
5. Indentation for hierarchy
6. Full words for anything ambiguous

### Steno-M Format (AI-only)

For machine-to-machine communication, Steno-M uses fixed schemas with positional fields:

```
#scope myproject
#schemas @F @T

@F BUG-123|open|high|auth-bypass-on-reset|auth-service|evidence/bug123
@T auth-service|active|go-grpc|primary auth, port 50052
@T billing-api|migrating|python-rest|gRPC migration 60%
```

Record types: `@V` (vulnerability), `@F` (finding), `@T` (target), `@C` (credential), `@A` (message), `@L` (lead)

## CLI Reference

```
steno index [--rebuild] MEMORY_DIR    Index memory files (incremental by default)
steno query "text" [N]                Semantic search, top N results
steno compress FILE [--write]         Compress prose -> Steno notation
steno stats                           Show index statistics
steno parse FILE_OR_DIR               Parse and preview records
```

`index`, `query` and `stats` all accept `--store=PATH` and `--collection=NAME`
to target a specific ChromaDB store/collection (see [Integration](#integration)).

### Compressing prose into Steno

Steno can WRITE the compressed notation, not just read it. The compressor is
rule-based (no LLM dependency) and lossy-but-auditable:

```bash
# Print compressed version to stdout (reports char-count reduction on stderr)
steno compress notes.md

# Rewrite the file in place (frontmatter preserved, format: steno set)
steno compress notes.md --write

# Choose level (default: steno)
steno compress notes.md --level steno
```

What it does (the README's "Steno Format Rules"):

- Drops articles (`a`/`an`/`the`) only where safe (never in headings).
- Abbreviates a curated, extendable dictionary (`authentication`→`auth`,
  `configuration`→`cfg`, `environment`→`env`, `repository`→`repo`,
  `database`→`db`, `production`→`prod`, `verified`→`vfd`, …). The dictionary is
  the legend — `steno_compress.expand_legend()` returns the expansion table.
- Compresses ISO dates `YYYY-MM-DD` → `MM-DD`.
- Collapses redundant whitespace.
- Preserves verbatim: inline code (`` `...` ``), fenced code blocks, URLs,
  emails, YAML frontmatter, and markdown structure.

An LLM-assisted pass can be plugged in later via the documented `llm_hook`
parameter of `steno_compress.compress()` — the default path uses no LLM.

### MCP server

Steno ships an stdio JSON-RPC MCP server (protocol `2024-11-05`) exposing the
tools `steno_query`, `steno_index`, and `steno_compress`:

```bash
python mcp_server.py
```

Register it with an MCP-capable agent (e.g. Claude Code):

```json
{
  "mcpServers": {
    "steno": { "command": "python", "args": ["/path/to/steno/mcp_server.py"] }
  }
}
```

Tool results are returned as `{"content": [{"type": "text", "text": <json>}]}`.

### Query Filters

```bash
# Filter by memory type
steno query "deployment process" --memory=project

# Filter by record type (Steno-M)
steno query "auth service" --type=@T

# Set minimum similarity score (default: 0.30)
steno query "testing" --min=0.6

# Target a specific shared store / collection
steno query "testing" --store=/path/to/store --collection=agent_memory
```

The minimum similarity score defaults to **0.30** consistently across the
library (`memory_retrieval.query`) and the CLI.

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `STENO_STORE` | `./chroma_store` | ChromaDB storage path |
| `MEMORY_STORE` | — | Fallback store path (used only if `STENO_STORE` is unset) |
| `MEMORY_COLLECTION` | `agent_memory` | ChromaDB collection name |
| `STENO_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformers model |

Store-dir precedence: explicit `--store` / arg > `STENO_STORE` > `MEMORY_STORE` >
`./chroma_store`. Collection precedence: explicit `--collection` / arg >
`MEMORY_COLLECTION` > `agent_memory`.

## How It Works

```
Memory files (Steno/Steno-M/prose)
    ↓ parse (steno_parser.py)
Structured records with metadata
    ↓ embed (memory_index.py)
ChromaDB vector store (local)
    ↓ query (memory_retrieval.py)
Top-K relevant records
    ↓ inject
AI agent context window
```

**Health-weighted scoring:** When used with [Vigil](https://github.com/KultMember6Banger/vigil), retrieval scores are multiplied by each memory's `health_score` (`score = similarity * health_score`, where `similarity = 1 - cosine_distance`). Stale, contradicted, or orphaned memories are automatically deprioritized without manual curation. Steno *reads* `health_score`; Vigil *writes* it.

**Access tracking:** Every retrieval updates `access_count` and `last_accessed` in ChromaDB metadata — and only for the records that actually pass the `min_score` filter and are returned (a high-ranked raw hit that gets filtered out is **not** bumped). Vigil uses this signal to apply Ebbinghaus retention curves — frequently accessed memories resist staleness decay.

**Incremental indexing:** Steno tracks file modification times. Only changed files are re-embedded on re-index. Unchanged files are skipped in ~0.1s.

**Junk filtering:** Empty lines, separators (`---`), and content-free fragments are automatically filtered out during parsing.

## Integration

### As a Python library

```python
from memory_index import build_index
from memory_retrieval import query

# Index
build_index(Path('./memories'))

# Query
results = query("what are the deployment rules", top_k=5)
for r in results:
    print(f"[{r.score:.2f}] {r.source_file}: {r.text[:100]}")
```

### With Claude Code

Place your memory files in `~/.claude/projects/YOUR_PROJECT/memory/` and run:

```bash
steno index ~/.claude/projects/YOUR_PROJECT/memory/
steno query "relevant context for current task"
```

### With other AI agents

The `query_formatted()` function returns a pre-formatted string ready for context injection:

```python
from memory_retrieval import query_formatted

context = query_formatted("auth service architecture", top_k=5)
# Returns formatted block ready to inject into agent prompt
```

### With Vigil (shared ChromaDB)

Steno and [Vigil](https://github.com/KultMember6Banger/vigil) operate on the
**same ChromaDB store and collection** so Vigil can audit exactly what Steno
indexes. The shared default collection is `agent_memory`.

```bash
# 1. Steno indexes memory into the shared store/collection
steno index ./memories --store ./shared_store --collection agent_memory

# 2. Vigil scores the same store (writes health_score onto each record)
#    (run Vigil pointed at ./shared_store / agent_memory)

# 3. Steno queries — retrieval is now health-weighted via health_score
steno query "deployment rules" --store ./shared_store --collection agent_memory
```

Metadata contract every record carries:

| Field | Written by | Notes |
|---|---|---|
| `source_file` | Steno | file stem |
| `record_type` | Steno | `@V`/`@F`/…/`chunk` |
| `memory_type` | Steno | user/feedback/project/reference |
| `access_count` | Steno | int, default 0; bumped on retrieval |
| `last_accessed` | Steno | ISO timestamp; set on retrieval |
| `health_score` | **Vigil** | float, default 1.0; Steno reads & multiplies into score |

## Performance

Tested on 85 memory files (466 KB total):

| Operation | Time |
|---|---|
| Full rebuild | ~20s |
| Incremental (no changes) | 0.08s |
| Single file re-index | ~3s |
| Query | ~2s (includes model load) |

Model: all-MiniLM-L6-v2 (384-dim, 80MB, CPU-only). No GPU required.

## License

MIT
