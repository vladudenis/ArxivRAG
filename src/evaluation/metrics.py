from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.embedder import PaperEmbedder
from src.rag_constants import PASSAGE_RELEVANCE_THRESHOLD

# Metric keys produced by compute_retrieval_metrics_for_query, in report order.
#   hit_at_k        : >=1 gold-relevant chunk in top-k (Success@k). Matches the metric
#                     the practical report labelled "Recall@k"; not capped by k/|gold|.
#   paper_coverage  : paper-level coverage = |retrieved paper IDs ∩ gold| / |gold|.
#                     Structurally capped at min(k, |gold|)/|gold|; report at k >= |gold|.
#   precision_at_k  : (# relevant chunks in top-k) / k.
#   mrr             : reciprocal rank of the first relevant chunk.
#   average_precision: AP over binary relevance (aggregates to MAP); rank-aware.
METRIC_KEYS = ("hit_at_k", "paper_coverage", "precision_at_k", "mrr", "average_precision")


def relevance_grades(
    top: list[tuple[str, dict[str, Any]]],
    gold_docs: list[str],
    gold_passages: list[str],
    judge_embedder: PaperEmbedder,
) -> list[float]:
    """
    Graded relevance in [0, 1] for each retrieved chunk.

    Grade = 1.0 if the chunk's paper ID is a gold document; otherwise the maximum
    cosine similarity between the chunk and any gold passage, clipped to [0, 1].
    Chunks with empty text or no gold passages receive 0.0.

    The judge embedder is intentionally decoupled from the retrieval embedder so
    relevance labels stay fixed when the retrieval embedding model is varied
    (embedding-model comparison). Gold passages are embedded once per call.
    """
    gold_doc_ids = {str(pid) for pid in gold_docs if str(pid)}
    grades: list[float] = [0.0] * len(top)

    pending_idx: list[int] = []
    pending_texts: list[str] = []
    for i, (chunk_text, payload) in enumerate(top):
        pid = str(payload.get("paper_id", "") or "")
        if pid and pid in gold_doc_ids:
            grades[i] = 1.0
            continue
        text = str(chunk_text or "").strip()
        if text and gold_passages:
            pending_idx.append(i)
            pending_texts.append(text)

    if pending_texts:
        c_emb = np.asarray(judge_embedder.embed_texts(pending_texts))
        p_emb = np.asarray(judge_embedder.embed_texts(gold_passages))
        sims = cosine_similarity(c_emb, p_emb)
        max_sims = sims.max(axis=1)
        for j, idx in enumerate(pending_idx):
            grades[idx] = float(np.clip(max_sims[j], 0.0, 1.0))

    return grades


def chunk_is_relevant(
    chunk_text: str,
    paper_id: str,
    gold_docs: list[str],
    gold_passages: list[str],
    embedder: PaperEmbedder,
    passage_threshold: float = PASSAGE_RELEVANCE_THRESHOLD,
) -> bool:
    """Relevant if doc matches gold_docs OR max cos-sim to any gold_passage >= threshold."""
    grade = relevance_grades([(chunk_text, {"paper_id": paper_id})], gold_docs, gold_passages, embedder)[0]
    return grade >= passage_threshold


def average_precision(rel_flags: list[bool]) -> float:
    """
    Average Precision over a binary-relevance ranked list.

    AP = mean of Precision@i over the ranks i where item i is relevant; 0 if none.
    Averaged across queries this yields MAP.
    """
    num_rel = 0
    ap_sum = 0.0
    for rank, is_rel in enumerate(rel_flags, start=1):
        if is_rel:
            num_rel += 1
            ap_sum += num_rel / float(rank)
    if num_rel == 0:
        return 0.0
    return float(ap_sum / num_rel)


def compute_retrieval_metrics_for_query(
    retrieved: list[tuple[str, dict[str, Any]]],
    gold_docs: list[str],
    gold_passages: list[str],
    embedder: PaperEmbedder,
    k: int,
    passage_threshold: float = PASSAGE_RELEVANCE_THRESHOLD,
    judge_embedder: PaperEmbedder | None = None,
) -> dict[str, float]:
    """
    IR metrics on the first k retrieved chunks (list should be pre-truncated to k).

    hit_at_k: 1.0 if >=1 relevant chunk in top-k, else 0.0 (Success@k). This is the
        metric the practical report labelled "Recall@k".
    paper_coverage: paper-level coverage = |retrieved_paper_ids ∩ gold_docs| / |gold_docs|
        (capped at min(k, |gold|)/|gold|).
    precision_at_k: (# relevant in top-k) / k, binary relevance at passage_threshold
    mrr: reciprocal rank of first relevant chunk, or 0.0
    average_precision: AP over binary relevance (aggregates to MAP)

    Binary relevance is derived by thresholding per-chunk graded relevance. The
    judge_embedder (if given) is used for grading instead of the retrieval embedder,
    keeping labels fixed across retrieval models.
    """
    if k <= 0:
        return {key: 0.0 for key in METRIC_KEYS}

    judge = judge_embedder or embedder
    top = retrieved[:k]

    top_paper_ids = {
        str(payload.get("paper_id", "") or "")
        for _, payload in top
        if str(payload.get("paper_id", "") or "")
    }
    gold_doc_ids = {str(pid) for pid in gold_docs if str(pid)}
    paper_coverage = (
        len(top_paper_ids.intersection(gold_doc_ids)) / float(len(gold_doc_ids))
        if gold_doc_ids
        else 1.0
    )

    grades = relevance_grades(top, gold_docs, gold_passages, judge)
    rel_flags = [g >= passage_threshold for g in grades]

    num_rel = sum(1 for r in rel_flags if r)
    hit_at_k = 1.0 if num_rel > 0 else 0.0
    precision_at_k = num_rel / float(k)

    mrr = 0.0
    for rank, is_rel in enumerate(rel_flags, start=1):
        if is_rel:
            mrr = 1.0 / float(rank)
            break

    return {
        "hit_at_k": hit_at_k,
        "paper_coverage": paper_coverage,
        "precision_at_k": precision_at_k,
        "mrr": mrr,
        "average_precision": average_precision(rel_flags),
    }
