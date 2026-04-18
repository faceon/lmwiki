"""Local vector DB using LanceDB + BM25 hybrid search with RRF fusion.

Architecture (gbrain-inspired):
  - LanceDB stores content chunks with embeddings (vector search)
  - BM25 over all page texts (keyword search)
  - RRF(k=60) fuses both ranked lists
  - Results are aggregated to page level before returning
"""

import re
import threading
from pathlib import Path
from typing import Callable

import lancedb
import pyarrow as pa
from rank_bm25 import BM25Okapi

from config import VECTORDB_DIR

EmbedFn = Callable[[list[str]], list[list[float]]]
RRF_K = 60
CHUNK_OVERLAP_CHARS = 100


def _get_embed_max_chars() -> int:
    """Return char limit per chunk from embedding model's context length (lazy, cached via embed.py)."""
    from embed import embed_max_chars
    return embed_max_chars()


# ── chunking ──────────────────────────────────────────────────────────────────

def _split_into_chunks(title: str, body: str) -> list[str]:
    """Split body into chunks by markdown sections, then by size if needed.

    Chunk size is derived from the embedding model's actual context length.
    """
    max_chars = _get_embed_max_chars()
    sections = re.split(r"\n(?=## )", body.strip())
    chunks: list[str] = []

    for section in sections:
        text = f"{title}: {section.strip()}"
        if len(text) <= max_chars:
            chunks.append(text)
        else:
            step = max(max_chars - CHUNK_OVERLAP_CHARS, 50)
            for start in range(0, len(text), step):
                chunks.append(text[start : start + max_chars])

    return chunks or [f"{title}: {body[:max_chars]}"]


# ── LanceDB table ──────────────────────────────────────────────────────────────

def _open_table(db_path: Path, dim: int) -> lancedb.table.Table:
    db = lancedb.connect(str(db_path))
    if "chunks" in db.table_names():
        return db.open_table("chunks")

    schema = pa.schema([
        pa.field("id", pa.utf8()),
        pa.field("title", pa.utf8()),
        pa.field("chunk_index", pa.int32()),
        pa.field("text", pa.utf8()),
        pa.field("vector", pa.list_(pa.float32(), dim)),
    ])
    return db.create_table("chunks", schema=schema)


_table: lancedb.table.Table | None = None
_table_lock = threading.Lock()


def get_table() -> lancedb.table.Table:
    global _table
    if _table is None:
        with _table_lock:
            if _table is None:  # re-check after acquiring lock
                from embed import embed_dim
                VECTORDB_DIR.mkdir(parents=True, exist_ok=True)
                _table = _open_table(VECTORDB_DIR, embed_dim())
    return _table


# ── upsert ────────────────────────────────────────────────────────────────────

def upsert_page(title: str, body: str, embed_fn: EmbedFn) -> int:
    """Chunk page body, embed, and upsert into LanceDB. Returns chunk count."""
    table = get_table()
    chunks = _split_into_chunks(title, body)
    embeddings = embed_fn(chunks)

    # Remove stale chunks for this page
    try:
        table.delete(f"title = '{title.replace(chr(39), chr(39)*2)}'")
    except Exception:
        pass

    rows = [
        {
            "id": f"{title}::{i}",
            "title": title,
            "chunk_index": i,
            "text": chunk,
            "vector": emb,
        }
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings))
    ]
    table.add(rows)
    return len(rows)


# ── BM25 helpers ──────────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """Whitespace tokenizer — works for Korean and English."""
    return text.lower().split()


def _gap_cutoff(scores: list[float], gap: float = 0.35, min_results: int = 1) -> int:
    """Return how many top-ranked items to keep based on the first relative score drop >= gap.

    Lower gap → triggers sooner → fewer results.
    Falls back to len(scores) if no gap found (hard cap n applies).
    """
    for i in range(min_results, len(scores)):
        prev, curr = scores[i - 1], scores[i]
        if prev > 0 and (prev - curr) / prev >= gap:
            return i
    return len(scores)


def _bm25_rank(all_pages: dict[str, str], query: str, limit: int) -> list[str]:
    """Return page titles ranked by BM25, cut at the first natural score gap."""
    if not all_pages:
        return []
    titles = list(all_pages.keys())
    corpus = [_tokenize(f"{t} {b}") for t, b in all_pages.items()]
    bm25 = BM25Okapi(corpus)
    raw = bm25.get_scores(_tokenize(query))
    max_score = max(raw) if any(s > 0 for s in raw) else 1.0

    ranked = sorted(range(len(raw)), key=lambda i: raw[i], reverse=True)
    # Only consider entries with positive score, up to limit
    candidates = [i for i in ranked[:limit] if raw[i] > 0]
    if not candidates:
        return []

    norm_scores = [raw[i] / max_score for i in candidates]
    cutoff = _gap_cutoff(norm_scores)
    return [titles[i] for i in candidates[:cutoff]]


# ── RRF fusion ────────────────────────────────────────────────────────────────

def _rrf_fuse(lists: list[list[str]], k: int = RRF_K) -> list[str]:
    """Reciprocal Rank Fusion over multiple title-ranked lists."""
    scores: dict[str, float] = {}
    for ranked in lists:
        for rank, title in enumerate(ranked):
            scores[title] = scores.get(title, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda t: scores[t], reverse=True)


def _rrf_fuse_with_scores(lists: list[list[str]], k: int = RRF_K) -> dict[str, float]:
    """RRF fusion returning normalized scores (0–1) keyed by title."""
    raw: dict[str, float] = {}
    for ranked in lists:
        for rank, title in enumerate(ranked):
            raw[title] = raw.get(title, 0.0) + 1.0 / (k + rank)
    if not raw:
        return {}
    max_score = max(raw.values())
    return {t: s / max_score for t, s in raw.items()}


# ── main search API ───────────────────────────────────────────────────────────

def find_related(
    all_pages: dict[str, str],
    source_text: str,
    n: int = 10,
    vec_distance_cap: float = 1.0,
    rrf_min_score: float = 0.5,
    embed_fn: EmbedFn | None = None,
) -> dict[str, tuple[str, float]]:
    """Hybrid search: vector + BM25 (each pre-filtered) → RRF → top N pages.

    Returns dict of title → (excerpt, similarity_score).
    similarity_score is cosine similarity (0–1) if vector search succeeded, else normalized RRF score.

    BM25 candidates are cut at the first natural score gap (relative drop ≥ 35%).
    Vector candidates are capped by cosine distance (0=identical, 2=opposite).
    After RRF fusion, only pages scoring ≥ rrf_min_score (normalized 0–1) are kept.
    """
    inner_limit = n * 3

    # ── keyword (BM25) — gap-filtered ─────────────────────────────────────────
    bm25_titles = _bm25_rank(all_pages, source_text, inner_limit)

    # ── vector — filtered by cosine distance ──────────────────────────────────
    vec_titles: list[str] = []
    vec_sims: dict[str, float] = {}
    vec_search_ran = False
    if embed_fn is not None:
        try:
            table = get_table()
            if table.count_rows() > 0:
                vec_search_ran = True
                query_vec = embed_fn([source_text[:_get_embed_max_chars()]])[0]
                rows = table.search(query_vec).limit(inner_limit).to_list()

                vec_candidates = []
                seen: set[str] = set()
                for row in rows:
                    t = row["title"]
                    d = row.get("_distance", 2.0)
                    if t not in seen and t in all_pages and d <= vec_distance_cap:
                        sim = max(0.01, 1.0 - (d / 2.0))
                        vec_candidates.append((t, sim))
                        if t not in vec_sims or sim > vec_sims[t]:
                            vec_sims[t] = sim
                        seen.add(t)

                if vec_candidates:
                    sims = [s for _, s in vec_candidates]
                    cutoff = _gap_cutoff(sims, gap=0.10, min_results=3)
                    vec_titles = [t for t, _ in vec_candidates[:cutoff]]
        except Exception:
            pass  # vector search failure is non-fatal

    # ── RRF over pre-filtered candidates ──────────────────────────────────────
    lists = [lst for lst in [vec_titles, bm25_titles] if lst]
    if not lists:
        return {}

    fused_scores = _rrf_fuse_with_scores(lists)

    fused_items = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    candidates = [(t, s) for t, s in fused_items if s >= rrf_min_score]

    # If vector search ran, require a vector match for inclusion.
    # Falls back to BM25-only when vector search was unavailable entirely.
    require_vec = vec_search_ran

    result: dict[str, tuple[str, float]] = {}
    for t, rrf_score in candidates[:n]:
        if t not in all_pages:
            continue
        if require_vec and t not in vec_sims:
            continue
        score = vec_sims.get(t, 0.0)
        result[t] = (all_pages[t][:1500], score)
    return result
