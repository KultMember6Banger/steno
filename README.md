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
git clone https://github.com/YOUR_USERNAME/steno.git
cd steno

# Install
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Index your memory files
python steno.py index ./examples

# Search
python steno.py query "database testing rules" 5

# Check index stats
python steno.py stats
```

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
steno stats                           Show index statistics
steno parse FILE_OR_DIR               Parse and preview records
```

### Query Filters

```bash
# Filter by memory type
steno query "deployment process" --memory=project

# Filter by record type (Steno-M)
steno query "auth service" --type=@T

# Set minimum similarity score
steno query "testing" --min=0.6
```

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `STENO_STORE` | `./chroma_store` | ChromaDB storage path |
| `STENO_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformers model |

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
