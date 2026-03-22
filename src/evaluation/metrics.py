from __future__ import annotations

from typing import Any

import sacrebleu
from rouge_score import rouge_scorer


def compute_must_include_hit_rate(answer: str, must_include: list[str]) -> float:
    """Return ratio of required facts found in the answer (case-insensitive)."""
    if not must_include:
        return 1.0

    answer_norm = answer.lower()
    hits = sum(1 for fact in must_include if fact.lower() in answer_norm)
    return hits / len(must_include)


def compute_recall_at_k(sources: list[dict[str, Any]], relevant_papers: list[str]) -> float:
    """Paper-level Recall@k using cited sources as retrieved set."""
    if not relevant_papers:
        return 1.0

    retrieved_ids = {s.get("paper_id", "") for s in sources}
    relevant_ids = set(relevant_papers)
    hits = len(retrieved_ids.intersection(relevant_ids))
    return hits / len(relevant_ids) if relevant_ids else 1.0


def compute_text_overlap_metrics(reference_answer: str, predicted_answer: str) -> dict[str, float]:
    """Compute ROUGE-L F1 and BLEU; returns empty metrics if no reference exists."""
    if not reference_answer.strip():
        return {
            "rouge_l_f1": 0.0,
            "bleu": 0.0,
        }

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge = scorer.score(reference_answer, predicted_answer)["rougeL"].fmeasure

    bleu = sacrebleu.corpus_bleu([predicted_answer], [[reference_answer]]).score

    return {
        "rouge_l_f1": float(rouge),
        "bleu": float(bleu),
    }


def compute_simple_response_metrics(answer: str, sources: list[dict[str, Any]]) -> dict[str, float]:
    """Simple quality signals independent from references."""
    return {
        "answer_chars": float(len(answer)),
        "answer_words": float(len(answer.split())),
        "citation_count": float(len(sources)),
        "empty_answer": 1.0 if not answer.strip() else 0.0,
    }

