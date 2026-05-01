from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.embedder import PaperEmbedder
from src.rag_constants import PASSAGE_RELEVANCE_THRESHOLD


def chunk_is_relevant(
    chunk_text: str,
    paper_id: str,
    gold_docs: list[str],
    gold_passages: list[str],
    embedder: PaperEmbedder,
    passage_threshold: float = PASSAGE_RELEVANCE_THRESHOLD,
) -> bool:
    """Relevant if doc matches gold_docs OR max cos-sim to any gold_passage >= threshold."""
    if paper_id in gold_docs:
        return True
    if not gold_passages or not chunk_text.strip():
        return False

    c_emb = embedder.embed_texts([chunk_text])
    p_emb = embedder.embed_texts(gold_passages)
    if isinstance(c_emb[0], list):
        c_emb = np.array(c_emb)
    if isinstance(p_emb[0], list):
        p_emb = np.array(p_emb)
    sims = cosine_similarity(c_emb, p_emb)[0]
    return float(np.max(sims)) >= passage_threshold


def compute_retrieval_metrics_for_query(
    retrieved: list[tuple[str, dict[str, Any]]],
    gold_docs: list[str],
    gold_passages: list[str],
    embedder: PaperEmbedder,
    k: int,
    passage_threshold: float = PASSAGE_RELEVANCE_THRESHOLD,
) -> dict[str, float]:
    """
    IR metrics on the first k retrieved chunks (list should be pre-truncated to k for scoring).

    recall_at_k: paper-level recall in top-k chunks = |retrieved_paper_ids ∩ gold_docs| / |gold_docs|
    hit_at_k: 1.0 if at least one retrieved chunk in top-k is relevant, else 0.0
    precision_at_k: (# relevant in top-k) / k
    mrr: reciprocal rank of first relevant chunk, or 0.0
    """
    if k <= 0:
        return {"recall_at_k": 0.0, "hit_at_k": 0.0, "precision_at_k": 0.0, "mrr": 0.0}

    top = retrieved[:k]
    top_paper_ids = {
        str(payload.get("paper_id", "") or "")
        for _, payload in top
        if str(payload.get("paper_id", "") or "")
    }
    gold_doc_ids = {str(pid) for pid in gold_docs if str(pid)}
    recall_at_k = (
        len(top_paper_ids.intersection(gold_doc_ids)) / float(len(gold_doc_ids))
        if gold_doc_ids
        else 1.0
    )

    rel_flags: list[bool] = []
    for chunk_text, payload in top:
        pid = str(payload.get("paper_id", "") or "")
        rel_flags.append(
            chunk_is_relevant(chunk_text, pid, gold_docs, gold_passages, embedder, passage_threshold)
        )

    num_rel = sum(1 for r in rel_flags if r)
    hit_at_k = 1.0 if num_rel > 0 else 0.0
    precision_at_k = num_rel / float(k)

    mrr = 0.0
    for rank, is_rel in enumerate(rel_flags, start=1):
        if is_rel:
            mrr = 1.0 / float(rank)
            break

    return {
        "recall_at_k": recall_at_k,
        "hit_at_k": hit_at_k,
        "precision_at_k": precision_at_k,
        "mrr": mrr,
    }
