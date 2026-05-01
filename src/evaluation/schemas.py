from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EvaluationExample:
    """One dataset row loaded from JSONL for freezing."""

    example_id: str
    topics: str
    question: str
