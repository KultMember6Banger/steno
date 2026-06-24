"""
Steno Parser — extracts structured records from memory files.

Handles three formats:
  - prose (legacy markdown, no format tag)
  - steno (human-auditable compressed)
  - steno-m (AI-only, schema-based)

Output: list of Record dicts ready for embedding and indexing.
"""

from __future__ import annotations

import re
import yaml
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class Record:
    """Single indexable unit extracted from a memory file."""
    id: str                     # unique record ID (file stem + chunk index)
    source_file: str            # path to originating memory file
    record_type: str            # @V, @F, @T, @C, @A, @L, or 'chunk'
    text: str                   # the content to embed
    metadata: dict = field(default_factory=dict)


# Steno-M record prefixes
STENO_M_PREFIXES = {'@V', '@F', '@T', '@C', '@A', '@L'}

# Frontmatter pattern
FRONTMATTER_RE = re.compile(r'^---\s*\n(.*?)\n---\s*\n', re.DOTALL)


def parse_frontmatter(content: str) -> tuple[dict, str]:
    """Extract YAML frontmatter and return (metadata, body)."""
    m = FRONTMATTER_RE.match(content)
    if not m:
        return {}, content
    try:
        meta = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError:
        meta = {}
    body = content[m.end():]
    return meta, body


def detect_format(meta: dict, body: str) -> str:
    """Detect file format from frontmatter or content inspection."""
    fmt = meta.get('format', '')
    if fmt in ('steno', 'steno-m'):
        return fmt
    for line in body.split('\n')[:20]:
        stripped = line.strip()
        if any(stripped.startswith(p + ' ') or stripped.startswith(p + '\t') for p in STENO_M_PREFIXES):
            return 'steno-m'
    return 'prose'


def _is_junk(text: str) -> bool:
    """Filter out content-free chunks (separators, whitespace-only, tiny fragments)."""
    stripped = text.strip()
    if re.fullmatch(r'[-=_*\s]+', stripped):
        return True
    content = re.sub(r'[^a-zA-Z0-9]', '', stripped)
    return len(content) < 15


def parse_steno_m(body: str, file_stem: str, meta: dict) -> list[Record]:
    """Parse Steno-M formatted content into records."""
    records = []
    scope = None
    idx = 0

    for line in body.split('\n'):
        stripped = line.strip()
        if not stripped:
            continue

        if stripped.startswith('#scope '):
            scope = stripped[7:].strip()
            continue
        if stripped.startswith('#schemas '):
            continue

        prefix_match = None
        for p in STENO_M_PREFIXES:
            if stripped.startswith(p + ' ') or stripped.startswith(p + '\t'):
                prefix_match = p
                break

        if prefix_match:
            record_text = stripped[len(prefix_match):].strip()
            fields = record_text.split('|')
            record_id = fields[0].strip() if fields else f'{idx}'

            records.append(Record(
                id=f'{file_stem}:{prefix_match}:{record_id}',
                source_file=file_stem,
                record_type=prefix_match,
                text=stripped,
                metadata={
                    **meta,
                    'scope': scope,
                    'record_type': prefix_match,
                }
            ))
            idx += 1

    return records


def parse_steno(body: str, file_stem: str, meta: dict, min_chunk: int = 40) -> list[Record]:
    """Parse Steno (auditable) formatted content into records.

    Splits on markdown headers to keep logical sections together.
    Falls back to paragraph splitting for headerless content.
    """
    records = []
    idx = 0

    sections = re.split(r'\n(?=#{1,3}\s)', body)

    for section in sections:
        section = section.strip()
        if not section or len(section) < min_chunk or _is_junk(section):
            continue

        records.append(Record(
            id=f'{file_stem}:s:{idx}',
            source_file=file_stem,
            record_type='chunk',
            text=section,
            metadata={**meta, 'record_type': 'steno_block'}
        ))
        idx += 1

    if not records:
        blocks = re.split(r'\n\n+', body)
        buffer = []
        buffer_len = 0

        for block in blocks:
            block = block.strip()
            if not block:
                continue
            if buffer_len + len(block) > 500 and buffer:
                records.append(Record(
                    id=f'{file_stem}:s:{idx}',
                    source_file=file_stem,
                    record_type='chunk',
                    text='\n\n'.join(buffer),
                    metadata={**meta, 'record_type': 'steno_block'}
                ))
                idx += 1
                buffer = []
                buffer_len = 0
            buffer.append(block)
            buffer_len += len(block)

        if buffer:
            records.append(Record(
                id=f'{file_stem}:s:{idx}',
                source_file=file_stem,
                record_type='chunk',
                text='\n\n'.join(buffer),
                metadata={**meta, 'record_type': 'steno_block'}
            ))

    return records


def parse_prose(body: str, file_stem: str, meta: dict, chunk_size: int = 500) -> list[Record]:
    """Parse prose/legacy formatted content into chunks.

    Splits on markdown headers first, then by paragraph boundaries
    if sections are too large.
    """
    records = []
    idx = 0

    sections = re.split(r'\n(?=#{1,4}\s)', body)

    for section in sections:
        section = section.strip()
        if not section or _is_junk(section):
            continue

        if len(section) <= chunk_size:
            records.append(Record(
                id=f'{file_stem}:p:{idx}',
                source_file=file_stem,
                record_type='chunk',
                text=section,
                metadata={**meta, 'record_type': 'prose_section'}
            ))
            idx += 1
        else:
            paragraphs = re.split(r'\n\n+', section)
            buffer = []
            buffer_len = 0

            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                if buffer_len + len(para) > chunk_size and buffer:
                    records.append(Record(
                        id=f'{file_stem}:p:{idx}',
                        source_file=file_stem,
                        record_type='chunk',
                        text='\n\n'.join(buffer),
                        metadata={**meta, 'record_type': 'prose_section'}
                    ))
                    idx += 1
                    buffer = []
                    buffer_len = 0
                buffer.append(para)
                buffer_len += len(para)

            if buffer:
                records.append(Record(
                    id=f'{file_stem}:p:{idx}',
                    source_file=file_stem,
                    record_type='chunk',
                    text='\n\n'.join(buffer),
                    metadata={**meta, 'record_type': 'prose_section'}
                ))
                idx += 1

    return records


def parse_file(file_path: Path) -> list[Record]:
    """Parse a single memory file into records."""
    content = file_path.read_text(encoding='utf-8')
    meta, body = parse_frontmatter(content)
    file_stem = file_path.stem
    fmt = detect_format(meta, body)

    meta['format'] = fmt
    meta['file_stem'] = file_stem
    meta['memory_type'] = meta.get('type', 'unknown')

    if fmt == 'steno-m':
        return parse_steno_m(body, file_stem, meta)
    elif fmt == 'steno':
        return parse_steno(body, file_stem, meta)
    else:
        return parse_prose(body, file_stem, meta)


def parse_directory(dir_path: Path, pattern: str = '*.md') -> list[Record]:
    """Parse all memory files in a directory."""
    records = []
    for f in sorted(dir_path.glob(pattern)):
        if f.name in ('MEMORY.md', 'README.md'):
            continue
        try:
            records.extend(parse_file(f))
        except Exception as e:
            print(f'WARN: failed to parse {f.name}: {e}')
    return records


if __name__ == '__main__':
    import sys

    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('.')
    if target.is_file():
        recs = parse_file(target)
    else:
        recs = parse_directory(target)

    print(f'Parsed {len(recs)} records')
    for r in recs[:5]:
        print(f'  {r.id} [{r.record_type}] {r.text[:80]}...')
