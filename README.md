# Steno

Compressed memory notation with RAG retrieval for AI agents.

Steno solves the AI memory problem: agents accumulate knowledge across sessions, but loading everything into context every time is expensive, noisy, and causes drift. Steno compresses memories into a dense notation format and retrieves only what's relevant using semantic search.

## The Problem

AI coding agents (Claude Code, Cursor, Copilot) build up memory files over time — user preferences, project context, past decisions, feedback. The default approach is brute-force: load all memory into every session. This wastes tokens, pollutes context with irrelevant information, and causes the agent to act on stale facts.

## The Solution

**Two-tier notation:**
- **Steno** — human-auditable compressed format. Drop articles, abbreviate common terms, use key-value pairs. Readable by humans, efficient for AI.
- **Steno-M** — AI-only format. Fixed schemas, positional fields, no labels. Maximum density for machine-to-machine communication. Steno both reads and **writes** Steno-M (see [`steno emit`](#writing-steno-m-from-structured-records-steno-emit)).

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
steno expand FILE [--write]           Expand Steno -> readable prose (best-effort)
steno emit JSON_FILE [--scope NAME]   Emit Steno-M from structured JSON records
steno curate MEMORY_DIR [--compress] [--gate] [--yes]   Self-curating loop
steno stats                           Show index statistics
steno parse FILE_OR_DIR               Parse and preview records
```

`index`, `query`, `stats` (and `compress --health-aware`) accept `--store=PATH`
and `--collection=NAME` to target a specific ChromaDB store/collection (see
[Integration](#integration)).

### Advanced retrieval (`query`)

```bash
# Token-budget retrieval — return the best SET of results that fits a budget,
# not a fixed top-K. Greedy knapsack by score-per-token (~4 chars/token est).
steno query "auth architecture" --budget=2000

# Hybrid search — fuse semantic ranking with pure-Python BM25 keyword ranking
# via Reciprocal Rank Fusion. Fixes recall on EXACT tokens that vector search
# misses: names, ports, IDs, error codes (e.g. "port 50052", "BUG-123").
steno query "port 50052" --hybrid

# MMR re-ranking — Maximal Marginal Relevance diversifies the top-K so you don't
# get five near-duplicate records. --mmr-lambda tunes relevance vs diversity
# (1.0 = pure relevance, 0.0 = pure diversity; default 0.5).
steno query "deployment" --mmr --mmr-lambda=0.5

# Combine them:
steno query "auth service port" --hybrid --mmr --budget=2000
```

| Flag | Effect |
|---|---|
| `--budget=N` | Return best result SET whose estimated tokens ≤ N (knapsack) |
| `--hybrid` | Semantic + BM25 keyword fusion (Reciprocal Rank Fusion) |
| `--mmr` | Diversify top-K via Maximal Marginal Relevance |
| `--mmr-lambda=F` | MMR relevance/diversity trade-off in [0,1] (default 0.5) |

These are additive and back-compatible: with none set, `query` behaves exactly
as before (semantic top-K).

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

#### LLM-assisted compression (optional hook)

The default path is 100% rule-based with no LLM dependency. To plug in a
*semantic* compression pass (prose → steno via summarisation), pass a callable
to `steno_compress.compress(..., llm_hook=...)`:

```python
from steno_compress import compress

def my_llm(line: str) -> str:
    # `line` is one already-rule-compressed prose line (never a code/URL span).
    return call_your_model(f"Compress to steno, keep meaning: {line}")

out = compress(prose, llm_hook=my_llm)   # rule-based first, then your model
```

Hook contract: `llm_hook(text: str) -> str`. Input is a single rule-compressed
prose line; preserved spans (inline code, URLs, emails) are stripped out before
the hook runs. Any exception is swallowed and the rule-based line is kept, so the
hook is always safe to fail.

#### Expanding Steno back to prose (`steno expand`)

`expand` is the **best-effort inverse** of `compress`: it runs the abbreviation
legend in reverse to recover readability. It is a readability aid, **not a
lossless decompressor**:

- Expands abbreviations (`auth`→`authentication`, `cfg`→`configuration`,
  `db`→`database`, `k8s`→`kubernetes`, …).
- Dropped articles (`a`/`an`/`the`) are **not** restored (that needs a language
  model).
- Compressed dates stay `MM-DD` (the year is gone — irreversible).
- A few abbreviations are ambiguous (`cfg` maps from both `configuration` and
  `configure`; expansion picks `configuration`).

```bash
steno expand notes.md            # print expanded prose to stdout
steno expand notes.md --write    # rewrite in place (drops format: steno)
```

#### Fidelity check (`steno compress --verify`)

`--verify` embeds the original and the compressed text with the real MiniLM
model, prints the cosine **fidelity**, and **warns** if it drops below the
threshold (default `0.92`):

```bash
steno compress notes.md --verify
# ...
# [steno compress] semantic fidelity (cosine): 0.95
```

Use it to catch meaning loss. Note: article-only compression scores high
(~0.95), but **aggressive abbreviation can score low** because MiniLM doesn't
know the legend — it sees `vfd` as gibberish, not `verified`. That low score is
the intended signal: it flags where compression has drifted from what the
embedder (and therefore retrieval) understands. `--verify` is most useful on
LLM-assisted or lighter passes. Available in the library as
`compress(text, verify=True)` (returns `(text, fidelity)`).

#### Health-aware compression (`--health-aware`)

Reads each memory's `health_score` from the shared store (written by
[Vigil](#with-vigil-shared-chromadb)) and compresses by health:

- **LOW health (`< 0.7`) or no health found** → aggressive (`steno`: drop
  articles + abbreviate). Decayed/contradicted memories shrink hard.
- **HIGH health (`≥ 0.7`)** → conservative (abbreviate only, keep articles), so
  trusted memories stay close to verbatim.

The policy keys off the **minimum** health across a file's records (any decayed
record pulls the file toward aggressive compression).

```bash
steno compress old_note.md --health-aware --store ./shared_store
# [steno compress] ... (health-aware: health=0.30 LOW -> aggressive)
```

#### Writing Steno-M from structured records (`steno emit`)

Steno-M was previously read-only. `to_steno_m()` (and the `steno emit` /
`steno compress --level steno-m` CLI paths) now **write** the positional
`@V/@F/@T/@C/@A/@L` format from structured dicts — the inverse of
`parse_steno_m`. JSON input shape:

```json
{
  "scope": "myproject",
  "records": [
    {"type": "@F", "fields": ["BUG-123", "open", "high", "auth-bypass"]},
    {"type": "@T", "fields": ["auth-service", "active", "primary auth, port 50052"]}
  ]
}
```

`fields[0]` is the record id. A long form `{"type": "@T", "id": "svc",
"fields": [...]}` is also accepted. `parse_steno_m(to_steno_m(x))` round-trips.

```bash
steno emit records.json --scope myproject
steno compress records.json --level steno-m   # same emit path
```

Output:

```
#scope myproject
#schemas @F @T

@F BUG-123|open|high|auth-bypass
@T auth-service|active|primary auth, port 50052
```

### MCP servers

Steno ships **two** stdio JSON-RPC MCP servers (protocol `2024-11-05`).

**1. `mcp_server.py` — Steno-only.** Exposes `steno_query`, `steno_index`,
`steno_compress`, and `memory_curate`:

```bash
python mcp_server.py
```

**2. `memory_mcp.py` — the unified memory server (recommended).** One entry
point for the *whole* memory system. It always exposes the Steno tools
`memory_query` (with `hybrid` / `mmr` / `budget` options), `memory_index`,
`memory_compress`, `memory_expand`, and `memory_curate`. When **Vigil is
importable** (pip-installed, or a sibling `../vigil/src` checkout) it *also*
exposes the auditor tools `memory_audit`, `memory_check`, `memory_fix`, and
`memory_health`. `tools/list` reflects what is actually available — Vigil tools
are not advertised when Vigil is absent.

```bash
python memory_mcp.py      # or the `memory-mcp` console script after install
```

Register either with an MCP-capable agent (e.g. Claude Code):

```json
{
  "mcpServers": {
    "memory": { "command": "python", "args": ["/path/to/steno/memory_mcp.py"] }
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

### Self-curating memory loop (`steno curate`)

`steno curate` ties the runtime (Steno) and the auditor (Vigil) into a single
self-maintaining pipeline so memory stays compressed, contradiction-free, and
health-weighted with one command:

```
compress  →  gate  →  index  →  score  →  health-weighted retrieve
(Steno)     (Vigil)   (Steno)   (Vigil)    (Steno)
```

```bash
# Full loop (Vigil installed): compress prose, gate contradictions, index, score.
steno curate ./memories --compress --gate --store ./shared_store

# Bare Steno (Vigil absent): compress + index only — Vigil steps are skipped.
steno curate ./memories --compress
```

Pipeline, per run:

1. **compress** (`--compress`) — compress prose memories in place; already-Steno
   files are skipped.
2. **gate** (`--gate`) — soft-imports Vigil and runs the pre-write contradiction
   check for each new/changed file against the existing index. A **CRITICAL**
   contradiction warns and **skips indexing that file** (use `--yes` to index it
   anyway). If Vigil isn't installed, a notice is printed and gating is skipped.
3. **index** — build/update the shared ChromaDB store/collection.
4. **score** — soft-imports Vigil's `full_scan` + `compute_health_scores` +
   `update_health_scores` and writes `health_score` onto each record, so the very
   next `steno query` is health-weighted. Skipped if Vigil is absent.

The summary prints `compressed`, `gated`, `indexed`, `scored`, and
`vigil_available`. The loop **works whether or not Vigil is importable.**

**Vigil discovery:** `curate` first tries a normal `import vigil` (pip-installed
is the expected path for full functionality), then falls back to a sibling
`../vigil/src` checkout next to the Steno repo. The same loop is available as the
`memory_curate` MCP tool on both MCP servers.

> Shared plumbing (frontmatter parsing, the cached embedder, the
> cosine↔distance conversion, ChromaDB client/collection setup, and
> store/collection resolution) lives in **`memcore.py`** — a canonical core
> vendored identically into both Steno and Vigil. A published `memcore` package
> is its eventual home.

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
