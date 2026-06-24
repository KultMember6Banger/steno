"""
Memory Index — embeds and stores Steno records in ChromaDB.

Uses all-MiniLM-L6-v2 (384-dim, CPU, ~80MB) for embeddings.
ChromaDB persists to disk. File mtimes tracked for incremental updates.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from steno_parser import Record, parse_directory, parse_file

# Defaults — override via env vars or function args.
#
# Store-dir precedence: explicit arg > STENO_STORE > MEMORY_STORE > ./chroma_store
# Collection precedence: explicit arg > MEMORY_COLLECTION > 'agent_memory'
#
# The shared 'agent_memory' collection name lets the sibling Vigil project audit
# exactly what Steno indexes by pointing at the same ChromaDB store + collection.
def _default_store_dir() -> Path:
    env = os.environ.get('STENO_STORE') or os.environ.get('MEMORY_STORE')
    if env:
        return Path(env)
    return Path(__file__).parent / 'chroma_store'


DEFAULT_STORE_DIR = _default_store_dir()
COLLECTION_NAME = os.environ.get('MEMORY_COLLECTION', 'agent_memory')
EMBED_MODEL = os.environ.get('STENO_MODEL', 'all-MiniLM-L6-v2')
BATCH_SIZE = 64
MTIME_FILE = 'file_mtimes.json'


def get_client(store_dir: Path = DEFAULT_STORE_DIR) -> chromadb.ClientAPI:
    """Get or create a persistent ChromaDB client."""
    store_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(store_dir))


def get_collection(client: chromadb.ClientAPI, name: str = None) -> chromadb.Collection:
    """Get or create the memory collection."""
    if name is None:
        name = COLLECTION_NAME
    return client.get_or_create_collection(
        name=name,
        metadata={'hnsw:space': 'cosine'}
    )


def load_model(model_name: str = EMBED_MODEL) -> SentenceTransformer:
    """Load the embedding model."""
    return SentenceTransformer(model_name)


def _mtime_path(store_dir: Path) -> Path:
    return store_dir / MTIME_FILE


def _load_mtimes(store_dir: Path) -> dict[str, float]:
    p = _mtime_path(store_dir)
    if p.exists():
        return json.loads(p.read_text())
    return {}


def _save_mtimes(store_dir: Path, mtimes: dict[str, float]):
    p = _mtime_path(store_dir)
    p.write_text(json.dumps(mtimes, indent=2))


def _sanitize_metadata(record: Record) -> dict:
    """Build ChromaDB-safe metadata (str/int/float/bool only)."""
    m = {
        'source_file': record.source_file,
        'record_type': record.record_type,
        'memory_type': record.metadata.get('memory_type', 'unknown'),
        'format': record.metadata.get('format', 'prose'),
        'access_count': 0,
        'last_accessed': '',
    }
    if record.metadata.get('scope'):
        m['scope'] = str(record.metadata['scope'])
    if record.metadata.get('name'):
        m['name'] = str(record.metadata['name'])
    return m


def _delete_file_records(collection: chromadb.Collection, file_stem: str):
    """Delete all records belonging to a specific source file."""
    try:
        collection.delete(where={'source_file': file_stem})
    except Exception:
        pass


def build_index(
    memory_dir: Path,
    store_dir: Path = DEFAULT_STORE_DIR,
    model_name: str = EMBED_MODEL,
    rebuild: bool = False,
    collection_name: str = None,
) -> dict:
    """Parse memory files, embed, and store in ChromaDB.

    Incremental by default: tracks file mtimes, only re-indexes changed files.

    Args:
        memory_dir: path to memory files
        store_dir: path to ChromaDB storage
        model_name: sentence-transformers model name
        rebuild: if True, drop and recreate collection
        collection_name: ChromaDB collection (default: COLLECTION_NAME / 'agent_memory')

    Returns:
        dict with stats
    """
    t0 = time.time()
    if collection_name is None:
        collection_name = COLLECTION_NAME

    client = get_client(store_dir)
    if rebuild:
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass
        mtime_p = _mtime_path(store_dir)
        if mtime_p.exists():
            mtime_p.unlink()

    collection = get_collection(client, collection_name)
    old_mtimes = {} if rebuild else _load_mtimes(store_dir)
    new_mtimes = {}

    changed_files = []
    unchanged_count = 0
    memory_files = sorted(memory_dir.glob('*.md'))

    for f in memory_files:
        if f.name in ('MEMORY.md', 'README.md'):
            continue
        mtime = f.stat().st_mtime
        new_mtimes[f.stem] = mtime

        if f.stem in old_mtimes and old_mtimes[f.stem] == mtime:
            unchanged_count += 1
        else:
            changed_files.append(f)

    deleted_stems = set(old_mtimes.keys()) - set(new_mtimes.keys())
    for stem in deleted_stems:
        _delete_file_records(collection, stem)

    if not changed_files and not deleted_stems:
        return {
            'records_parsed': 0,
            'records_indexed': 0,
            'records_updated': 0,
            'records_unchanged': unchanged_count,
            'files_deleted': len(deleted_stems),
            'total_files': len(new_mtimes),
            'time_sec': round(time.time() - t0, 2),
        }

    model = load_model(model_name)

    n_indexed = 0
    n_updated = 0

    for f in changed_files:
        if f.stem in old_mtimes:
            _delete_file_records(collection, f.stem)
            n_updated += 1

        try:
            records = parse_file(f)
        except Exception as e:
            print(f'WARN: failed to parse {f.name}: {e}')
            continue

        if not records:
            continue

        for i in range(0, len(records), BATCH_SIZE):
            batch = records[i:i + BATCH_SIZE]
            texts = [r.text for r in batch]
            ids = [r.id for r in batch]
            metadatas = [_sanitize_metadata(r) for r in batch]
            embeddings = model.encode(texts, show_progress_bar=False).tolist()

            collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )
            n_indexed += len(batch)

    _save_mtimes(store_dir, new_mtimes)

    return {
        'records_parsed': n_indexed,
        'records_indexed': n_indexed,
        'records_updated': n_updated,
        'records_unchanged': unchanged_count,
        'files_deleted': len(deleted_stems),
        'total_files': len(new_mtimes),
        'time_sec': round(time.time() - t0, 2),
    }


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    flags = [a for a in sys.argv[1:] if a.startswith('--')]

    memory_dir = Path(args[0]) if args else Path('.')
    rebuild = '--rebuild' in flags

    store_dir = DEFAULT_STORE_DIR
    collection_name = None
    for flag in flags:
        if flag.startswith('--store='):
            store_dir = Path(flag[len('--store='):])
        elif flag.startswith('--collection='):
            collection_name = flag[len('--collection='):]

    print(f'Indexing {memory_dir}...')
    if rebuild:
        print('(rebuilding from scratch)')

    stats = build_index(memory_dir, store_dir=store_dir, rebuild=rebuild,
                        collection_name=collection_name)
    print(f'Done in {stats["time_sec"]}s')
    print(f'  Indexed:   {stats["records_indexed"]} records')
    print(f'  Updated:   {stats["records_updated"]} files re-indexed')
    print(f'  Unchanged: {stats["records_unchanged"]} files skipped')
    print(f'  Deleted:   {stats["files_deleted"]} files cleaned up')
    print(f'  Total:     {stats["total_files"]} files tracked')
