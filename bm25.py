"""
BM25 — pure-Python Okapi BM25 keyword ranking over a corpus of documents.

Why this exists: pure vector (semantic) search reliably misses *exact tokens* —
names, ports, IDs, error codes, flags — because MiniLM embeds meaning, not
surface form. BM25 ranks by term frequency / inverse document frequency over the
literal tokens, so a query for "port 50052" or "BUG-123" lands the record that
contains that exact string. Steno fuses BM25 with the vector ranking via
Reciprocal Rank Fusion (see memory_retrieval.query(hybrid=True)).

No heavy dependency: this is a few dozen lines of standard-library Python. The
index is rebuilt from the collection's stored documents on each hybrid query —
the corpus is small (memory files), so this is cheap and always in sync.

Tokenization is deliberately simple and lossless-ish: lower-cased alphanumeric
runs, with hyphens/dots inside identifiers preserved as part of the token (so
`BUG-123`, `auth-service`, `v1.2.3`, `50052` stay intact).
"""

from __future__ import annotations

import math
import re

# A token is a run of word chars; we also keep internal '-' and '.' so that
# identifiers like auth-service, BUG-123 and v1.2.3 survive as single tokens.
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-.][A-Za-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """Lower-case alphanumeric tokenization, keeping identifier punctuation."""
    return [t.lower() for t in _TOKEN_RE.findall(text or '')]


class BM25:
    """Okapi BM25 over an in-memory corpus.

    Args:
        corpus: list of raw document strings (parallel to whatever ids you keep).
        k1: term-frequency saturation (default 1.5, standard).
        b: length-normalization strength (default 0.75, standard).
    """

    def __init__(self, corpus: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.doc_tokens = [tokenize(doc) for doc in corpus]
        self.doc_len = [len(toks) for toks in self.doc_tokens]
        self.avgdl = (sum(self.doc_len) / self.corpus_size) if self.corpus_size else 0.0

        # Document frequency: number of docs containing each term.
        self.df: dict[str, int] = {}
        # Per-doc term frequencies.
        self.tf: list[dict[str, int]] = []
        for toks in self.doc_tokens:
            freqs: dict[str, int] = {}
            for t in toks:
                freqs[t] = freqs.get(t, 0) + 1
            self.tf.append(freqs)
            for t in freqs:
                self.df[t] = self.df.get(t, 0) + 1

        # Precompute idf with the standard BM25 (Robertson) formula, floored at 0
        # so that very common terms can't push scores negative.
        self.idf: dict[str, float] = {}
        for term, freq in self.df.items():
            idf = math.log(1 + (self.corpus_size - freq + 0.5) / (freq + 0.5))
            self.idf[term] = idf

    def get_scores(self, query: str) -> list[float]:
        """Return a BM25 score for every document in the corpus (parallel list)."""
        q_terms = tokenize(query)
        scores = [0.0] * self.corpus_size
        if not self.corpus_size:
            return scores
        for term in q_terms:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i in range(self.corpus_size):
                freq = self.tf[i].get(term, 0)
                if freq == 0:
                    continue
                denom = freq + self.k1 * (
                    1 - self.b + self.b * (self.doc_len[i] / self.avgdl if self.avgdl else 0)
                )
                scores[i] += idf * (freq * (self.k1 + 1)) / denom
        return scores

    def top_n(self, query: str, n: int = 10) -> list[tuple[int, float]]:
        """Return up to n (doc_index, score) pairs sorted by score desc.

        Only documents with a strictly positive score are returned.
        """
        scored = [(i, s) for i, s in enumerate(self.get_scores(query)) if s > 0]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:n]


def reciprocal_rank_fusion(
    rankings: list[list[str]],
    k: int = 60,
) -> dict[str, float]:
    """Fuse several ranked id-lists into one score map via Reciprocal Rank Fusion.

    RRF score for a document = sum over rankings of 1 / (k + rank), where rank is
    0-based position in that ranking. Documents absent from a ranking contribute
    nothing from it. The constant k (default 60, the value from the original RRF
    paper) damps the influence of top ranks so no single retriever dominates.

    Args:
        rankings: list of ranked lists of ids (most-relevant first).
        k: RRF damping constant.

    Returns:
        dict mapping id -> fused score (higher is better).
    """
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    return fused
