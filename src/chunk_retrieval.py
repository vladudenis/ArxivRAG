"""
Chunk retrieval: dense, hybrid (dense + BM25), optional cross-encoder re-ranking, MMR, neighbor expansion.
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Tuple

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity

from src.chunker import ChunkingStrategy
from src.embedder import PaperEmbedder
from src.rag_constants import (
    DEFAULT_CROSS_ENCODER_MODEL,
    HYBRID_DENSE_WEIGHT,
    RERANK_POOL_MAX,
    RERANK_POOL_MIN,
    RERANK_POOL_MULTIPLIER,
)

_cross_encoders: dict[str, Any] = {}


def _get_cross_encoder(model_name: str) -> Any:
    from sentence_transformers import CrossEncoder

    if model_name not in _cross_encoders:
        _cross_encoders[model_name] = CrossEncoder(model_name)
    return _cross_encoders[model_name]


def _mmr_select(
    vectors: np.ndarray,
    query_emb: np.ndarray,
    top_k: int,
    lambda_param: float = 0.7,
) -> List[int]:
    n = len(vectors)
    if n <= top_k:
        return list(range(n))

    sim_to_query = cosine_similarity(query_emb, vectors)[0]
    sim_matrix = cosine_similarity(vectors)

    selected: List[int] = []
    remaining = set(range(n))

    for _ in range(top_k):
        if not remaining:
            break

        best_score = float("-inf")
        best_idx = -1

        for idx in remaining:
            relevance = sim_to_query[idx]
            if not selected:
                mmr_score = relevance
            else:
                max_sim_to_selected = max(sim_matrix[idx, s] for s in selected)
                mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim_to_selected

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        if best_idx >= 0:
            selected.append(best_idx)
            remaining.discard(best_idx)

    return selected


def _expand_chunk_neighbors(
    selected: List[Tuple[str, Dict[str, Any]]],
    all_payloads: List[Dict[str, Any]],
) -> List[Tuple[str, Dict[str, Any]]]:
    if not selected or not all_payloads:
        return selected

    lookup: Dict[Tuple[str, int], Tuple[str, Dict[str, Any]]] = {}
    for p in all_payloads:
        if not p:
            continue
        paper_id = p.get("paper_id", "")
        position = p.get("position", 0)
        chunk_text = p.get("chunk_text", "")
        lookup[(paper_id, position)] = (chunk_text, p)

    expanded: List[Tuple[str, Dict[str, Any]]] = []
    seen: set[Tuple[str, int]] = set()

    for chunk_text, payload in selected:
        paper_id = payload.get("paper_id", "")
        pos = payload.get("position", 0)

        for neighbor_pos in (pos - 1, pos, pos + 1):
            key = (paper_id, neighbor_pos)
            if key in seen:
                continue
            if key in lookup:
                seen.add(key)
                expanded.append(lookup[key])

    return expanded


def _normalize_minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-12:
        return np.ones_like(x, dtype=np.float64)
    return (x - lo) / (hi - lo + 1e-12)


def retrieve_chunks_with_metadata(
    storage: Any,
    embedder: PaperEmbedder,
    query: str,
    strategy: ChunkingStrategy,
    top_k: int = 8,
    use_mmr: bool = True,
    mmr_lambda: float = 0.7,
    expand_neighbors: bool = True,
    retrieval_type: Literal["dense", "hybrid"] = "dense",
    re_ranking: bool = False,
    cross_encoder_model: str = DEFAULT_CROSS_ENCODER_MODEL,
) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Retrieve top-k chunks with metadata.

    hybrid: weighted combination of normalized dense cosine and BM25 scores.
    re_ranking: cross-encoder scores a candidate pool, then top_k is taken.
    use_mmr: when retrieval_type is dense and re_ranking is False, use MMR; ignored for hybrid path.
    """
    try:
        vectors, payloads = storage.fetch_embeddings(strategy.value)

        if not vectors or not payloads:
            return []

        chunk_texts = [str(p.get("chunk_text", "") or "") for p in payloads]

        query_emb = embedder.embed_texts([query])

        if isinstance(vectors[0], list):
            vectors = np.array(vectors)

        dense_sims = cosine_similarity(query_emb, vectors)[0]

        n = len(payloads)

        if re_ranking:
            pool = min(
                n,
                max(RERANK_POOL_MIN, min(RERANK_POOL_MAX, top_k * RERANK_POOL_MULTIPLIER)),
            )

            if retrieval_type == "dense":
                prelim = np.argsort(-dense_sims)[:pool]
            else:
                tokenized = [t.lower().split() for t in chunk_texts]
                if not any(tokenized):
                    bm25_scores = np.zeros(n)
                else:
                    bm25 = BM25Okapi(tokenized)
                    q_tok = query.lower().split()
                    bm25_scores = np.array(bm25.get_scores(q_tok), dtype=np.float64)
                d_n = _normalize_minmax(np.asarray(dense_sims))
                b_n = _normalize_minmax(bm25_scores)
                hybrid = HYBRID_DENSE_WEIGHT * d_n + (1.0 - HYBRID_DENSE_WEIGHT) * b_n
                prelim = np.argsort(-hybrid)[:pool]

            ce = _get_cross_encoder(cross_encoder_model)
            pairs = [[query, chunk_texts[int(i)]] for i in prelim]
            ce_scores = ce.predict(pairs, show_progress_bar=False)
            order = np.argsort(-np.asarray(ce_scores, dtype=np.float64))
            top_indices = [int(prelim[int(j)]) for j in order[:top_k]]
        else:
            if retrieval_type == "hybrid":
                tokenized = [t.lower().split() for t in chunk_texts]
                if not any(tokenized):
                    bm25_scores = np.zeros(n)
                else:
                    bm25 = BM25Okapi(tokenized)
                    q_tok = query.lower().split()
                    bm25_scores = np.array(bm25.get_scores(q_tok), dtype=np.float64)
                d_n = _normalize_minmax(np.asarray(dense_sims))
                b_n = _normalize_minmax(bm25_scores)
                hybrid = HYBRID_DENSE_WEIGHT * d_n + (1.0 - HYBRID_DENSE_WEIGHT) * b_n
                top_indices = np.argsort(-hybrid)[:top_k].tolist()
            elif use_mmr:
                top_indices = _mmr_select(vectors, query_emb, top_k=top_k, lambda_param=mmr_lambda)
            else:
                top_indices = np.argsort(-dense_sims)[:top_k].tolist()

        result: List[Tuple[str, Dict[str, Any]]] = []
        for i in top_indices:
            payload = payloads[i] or {}
            chunk_text = payload.get("chunk_text", "")
            result.append((chunk_text, payload))

        if expand_neighbors:
            result = _expand_chunk_neighbors(result, payloads)

        return result

    except Exception as e:
        print(f"Error retrieving chunks: {e}")
        import traceback

        traceback.print_exc()
        return []
