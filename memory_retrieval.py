"""
Memory Retrieval — query the Steno memory index via semantic search.

Returns top-K relevant records from ChromaDB, ranked by cosine similarity.
Supports filtering by record type, memory type, and source file.

Access tracking: each retrieval updates access_count and last_accessed
in ChromaDB metadata, enabling Vigil's access-frequency decay scoring.
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime, timezone

import chromadb
from sentence_transformers import SentenceTransformer

from memory_index import DEFAULT_STORE_DIR, COLLECTION_NAME, EMBED_MODEL


@dataclass
class Result:
    """Single retrieval result."""
    id: str
    text: str
    score: float          # cosine similarity (0-1, higher = more relevant)
    source_file: str
    record_type: str
    memory_type: str


def query(
    query_text: str,
    top_k: int = 10,
    record_type: str | None = None,
    memory_type: str | None = None,
    source_file: str | None = None,
    min_score: float = 0.55,
    store_dir: Path = DEFAULT_STORE_DIR,
    model_name: str = EMBED_MODEL,
) -> list[Result]:
    """Query the memory index.

    Args:
        query_text: natural language query
        top_k: max results to return
        record_type: filter by @V, @F, @T, etc.
        memory_type: filter by user, feedback, project, reference
        source_file: filter by source file stem
        min_score: minimum cosine similarity threshold (default 0.55)
        store_dir: path to ChromaDB storage
        model_name: embedding model name

    Returns:
        list of Result, sorted by score descending
    """
    if not query_text or not query_text.strip():
        return []

    client = chromadb.PersistentClient(path=str(store_dir))
    collection = client.get_collection(COLLECTION_NAME)
    model = SentenceTransformer(model_name)

    where = {}
    conditions = []
    if record_type:
        conditions.append({'record_type': record_type})
    if memory_type:
        conditions.append({'memory_type': memory_type})
    if source_file:
        conditions.append({'source_file': source_file})

    if len(conditions) == 1:
        where = conditions[0]
    elif len(conditions) > 1:
        where = {'$and': conditions}

    query_embedding = model.encode([query_text], show_progress_bar=False).tolist()

    kwargs = {
        'query_embeddings': query_embedding,
        'n_results': top_k,
        'include': ['documents', 'metadatas', 'distances'],
    }
    if where:
        kwargs['where'] = where

    results = collection.query(**kwargs)

    output = []
    for i in range(len(results['ids'][0])):
        distance = results['distances'][0][i]
        raw_score = 1 - (distance / 2)

        if raw_score < min_score:
            continue

        meta = results['metadatas'][0][i]
        health = float(meta.get('health_score', 1.0))
        score = raw_score * health

        output.append(Result(
            id=results['ids'][0][i],
            text=results['documents'][0][i],
            score=round(score, 4),
            source_file=meta.get('source_file', ''),
            record_type=meta.get('record_type', ''),
            memory_type=meta.get('memory_type', ''),
        ))

    # Update access tracking for returned records
    if output:
        _update_access_counts(collection, results, len(output))

    return output


def _update_access_counts(collection, results, n_returned: int):
    """Bump access_count and last_accessed for retrieved records."""
    now_iso = datetime.now(timezone.utc).isoformat()
    batch_ids = []
    batch_metas = []

    for i in range(n_returned):
        rec_id = results['ids'][0][i]
        meta = dict(results['metadatas'][0][i])
        meta['access_count'] = int(meta.get('access_count', 0)) + 1
        meta['last_accessed'] = now_iso
        batch_ids.append(rec_id)
        batch_metas.append(meta)

    try:
        collection.update(ids=batch_ids, metadatas=batch_metas)
    except Exception:
        pass  # access tracking is best-effort, never block retrieval


def query_formatted(query_text: str, top_k: int = 10, **kwargs) -> str:
    """Query and return formatted string output for context injection."""
    results = query(query_text, top_k=top_k, **kwargs)
    if not results:
        return '(no relevant memories found)'

    lines = [f'--- Memory retrieval: "{query_text}" ({len(results)} results) ---']
    for r in results:
        lines.append(f'[{r.score:.2f}] {r.source_file} ({r.record_type}/{r.memory_type})')
        for tline in r.text.split('\n'):
            lines.append(f'  {tline}')
        lines.append('')

    return '\n'.join(lines)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: memory_retrieval.py "query text" [top_k] [--type=@V] [--memory=feedback]')
        sys.exit(1)

    query_text = sys.argv[1]
    top_k = 10
    record_type = None
    memory_type = None

    for arg in sys.argv[2:]:
        if arg.startswith('--type='):
            record_type = arg[7:]
        elif arg.startswith('--memory='):
            memory_type = arg[9:]
        elif arg.isdigit():
            top_k = int(arg)

    results = query(query_text, top_k=top_k, record_type=record_type, memory_type=memory_type)

    print(f'\nQuery: "{query_text}"')
    print(f'Results: {len(results)}\n')

    for r in results:
        print(f'[{r.score:.4f}] {r.id}')
        print(f'  source: {r.source_file} | type: {r.record_type} | memory: {r.memory_type}')
        preview = r.text[:120].replace('\n', ' ')
        print(f'  {preview}...')
        print()
