from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvaluationExample:
    """One dataset row loaded from JSONL."""

    example_id: str
    topics: str
    question: str
    reference_answer: str = ""
    must_include: list[str] = field(default_factory=list)
    relevant_papers: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class JudgeScores:
    """Structured judge output for one strategy response."""

    correctness: int
    groundedness: int
    completeness: int
    citation_quality: int
    hallucination: bool
    verdict: str
    reasoning: str

    @property
    def composite_score(self) -> float:
        # Weighted score aligned with RAG quality priorities.
        return (
            0.35 * self.groundedness
            + 0.30 * self.correctness
            + 0.20 * self.completeness
            + 0.15 * self.citation_quality
        )

