"""
Memory Retrieval — query the Steno memory index via semantic search.

Returns top-K relevant records from ChromaDB, ranked by cosine similarity.
Supports filtering by record type, memory type, and source file.

Access tracking: each retrieval updates access_count and last_accessed
in ChromaDB metadata, enabling Vigil's access-frequency decay scoring.
"""

from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime, timezone

import math

import chromadb
from sentence_transformers import SentenceTransformer

from memory_index import DEFAULT_STORE_DIR, COLLECTION_NAME, EMBED_MODEL
from bm25 import BM25, reciprocal_rank_fusion
from mmr import mmr_rank


@dataclass
class Result:
    """Single retrieval result."""
    id: str
    text: str
    score: float          # cosine similarity (0-1, higher = more relevant)
    source_file: str
    record_type: str
    memory_type: str


def _estimate_tokens(text: str) -> int:
    """Cheap token-count estimate (no tiktoken dependency).

    Uses the common ~4-chars-per-token heuristic. Good enough for budgeting how
    many retrieved records fit a context window; never returns less than 1 for
    non-empty text.
    """
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


def _knapsack_by_budget(results: list[Result], token_budget: int) -> list[Result]:
    """Greedily pick the best SET of results whose tokens fit token_budget.

    Lightweight knapsack: sort by value-density (score per token) descending and
    take items while they fit. This favours high-relevance, low-cost records so
    the budget is spent where it matters, rather than a fixed top_k. Returned in
    score-descending order for stable presentation.
    """
    if token_budget is None or token_budget <= 0:
        return results

    ranked = sorted(
        results,
        key=lambda r: (r.score / max(1, _estimate_tokens(r.text))),
        reverse=True,
    )
    chosen: list[Result] = []
    used = 0
    for r in ranked:
        cost = _estimate_tokens(r.text)
        if used + cost <= token_budget:
            chosen.append(r)
            used += cost
    chosen.sort(key=lambda r: r.score, reverse=True)
    return chosen


def query(
    query_text: str,
    top_k: int = 10,
    record_type: str | None = None,
    memory_type: str | None = None,
    source_file: str | None = None,
    min_score: float = 0.30,
    store_dir: Path = DEFAULT_STORE_DIR,
    model_name: str = EMBED_MODEL,
    collection_name: str | None = None,
    token_budget: int | None = None,
    hybrid: bool = False,
    mmr: bool = False,
    mmr_lambda: float = 0.5,
) -> list[Result]:
    """Query the memory index.

    Args:
        query_text: natural language query
        top_k: max results to return
        record_type: filter by @V, @F, @T, etc.
        memory_type: filter by user, feedback, project, reference
        source_file: filter by source file stem
        min_score: minimum cosine similarity threshold (default 0.30)
        store_dir: path to ChromaDB storage
        model_name: embedding model name
        collection_name: ChromaDB collection (default: COLLECTION_NAME / 'agent_memory')
        token_budget: if set, return the best SET of results whose estimated
            combined token cost fits the budget (greedy knapsack by score/token),
            instead of a fixed top_k. See _estimate_tokens / _knapsack_by_budget.
        hybrid: if True, fuse semantic ranking with a pure-Python BM25 keyword
            ranking over the collection's documents via Reciprocal Rank Fusion.
            Improves recall on exact tokens (names, ports, IDs, error codes).
        mmr: if True, re-rank results with Maximal Marginal Relevance over the
            embeddings ChromaDB returns, to diversify the top-K (avoid
            near-duplicates).
        mmr_lambda: MMR relevance/diversity trade-off in [0,1] (default 0.5).

    Returns:
        list of Result, sorted by score descending
    """
    if not query_text or not query_text.strip():
        return []

    if collection_name is None:
        collection_name = COLLECTION_NAME

    client = chromadb.PersistentClient(path=str(store_dir))
    collection = client.get_collection(collection_name)
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

    # When hybrid or mmr is on we over-fetch a candidate pool so fusion/diversity
    # has something to work with, then trim to top_k (or the budget) at the end.
    fetch_k = top_k
    if hybrid or mmr or token_budget:
        fetch_k = max(top_k * 4, 40)

    include = ['documents', 'metadatas', 'distances']
    if mmr:
        include.append('embeddings')

    kwargs = {
        'query_embeddings': query_embedding,
        'n_results': fetch_k,
        'include': include,
    }
    if where:
        kwargs['where'] = where

    results = collection.query(**kwargs)

    ids_row = results['ids'][0]
    n = len(ids_row)

    # ChromaDB cosine distance = 1 - cosine_similarity, range [0, 2]
    # Convert back: similarity = 1 - distance. score = similarity * health_score.
    semantic_scores: list[float] = []
    for i in range(n):
        distance = results['distances'][0][i]
        raw_score = 1 - distance
        meta = results['metadatas'][0][i]
        health = float(meta.get('health_score', 1.0))
        semantic_scores.append(raw_score * health)

    # Order of candidate indices we ultimately keep, before min_score filtering.
    order = list(range(n))  # chroma already returns by ascending distance

    if hybrid and n:
        # Build BM25 over the candidate documents and fuse rankings via RRF.
        docs = [results['documents'][0][i] for i in range(n)]
        bm25 = BM25(docs)
        bm25_ranked = [ids_row[i] for i, _ in bm25.top_n(query_text, n=n)]
        semantic_ranked = [ids_row[i] for i in sorted(range(n), key=lambda j: semantic_scores[j], reverse=True)]
        fused = reciprocal_rank_fusion([semantic_ranked, bm25_ranked])
        id_to_idx = {ids_row[i]: i for i in range(n)}
        # Any id present in either ranking is in `fused`; sort by fused score.
        order = [id_to_idx[_id] for _id in sorted(fused, key=lambda x: fused[x], reverse=True)]

    if mmr and n:
        # Diversify using embeddings, restricted to the current candidate order.
        embeddings_row = results.get('embeddings')
        embs = embeddings_row[0] if embeddings_row is not None else None
        if embs is not None and len(embs):
            cand_embs = [embs[i] for i in order]
            mmr_order_local = mmr_rank(query_embedding[0], cand_embs, lambda_mult=mmr_lambda)
            order = [order[j] for j in mmr_order_local]

    # Build outputs in the chosen order, applying min_score and (when not budgeting)
    # the top_k cap. token_budget selection happens after, on the filtered set.
    output = []
    passed_indices = []
    for i in order:
        score = semantic_scores[i]
        if score < min_score:
            continue
        meta = results['metadatas'][0][i]
        passed_indices.append(i)
        output.append(Result(
            id=ids_row[i],
            text=results['documents'][0][i],
            score=round(score, 4),
            source_file=meta.get('source_file', ''),
            record_type=meta.get('record_type', ''),
            memory_type=meta.get('memory_type', ''),
        ))

    if token_budget:
        output = _knapsack_by_budget(output, token_budget)
        kept_ids = {r.id for r in output}
        passed_indices = [i for i in passed_indices if ids_row[i] in kept_ids]
    else:
        output = output[:top_k]
        kept_ids = {r.id for r in output}
        passed_indices = [i for i in passed_indices if ids_row[i] in kept_ids]

    # Update access tracking ONLY for the records that actually passed the filter
    # AND are returned (after budget/top_k trimming). This signal feeds Vigil's
    # staleness decay, so it must be accurate.
    if passed_indices:
        _update_access_counts(collection, results, passed_indices)

    return output


def _update_access_counts(collection, results, passed_indices):
    """Bump access_count and last_accessed for the records that passed the filter.

    Args:
        collection: ChromaDB collection
        results: raw ChromaDB query results
        passed_indices: list of raw result indices that survived min_score filtering
    """
    now_iso = datetime.now(timezone.utc).isoformat()
    batch_ids = []
    batch_metas = []

    for i in passed_indices:
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
