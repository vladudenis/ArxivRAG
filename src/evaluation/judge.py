from __future__ import annotations

import json
from typing import Any

from src.llm_client import DeepSeekClient
from src.evaluation.schemas import JudgeScores


SYSTEM_PROMPT = """You are a strict evaluator for Retrieval-Augmented Generation (RAG) answers.
You must score the candidate answer using only the provided question, reference answer, retrieved context, and citations.
Output must be valid JSON only. No markdown, no prose outside JSON.
"""


def _extract_json_object(raw_text: str) -> dict[str, Any]:
    """Parse and return first JSON object found in the model output."""
    raw_text = raw_text.strip()

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        pass

    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Judge response does not contain JSON object.")

    return json.loads(raw_text[start : end + 1])


def _coerce_score(raw_value: Any, default_value: int = 1) -> int:
    """Convert a judge score into an integer between 1 and 5."""
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        return default_value
    return max(1, min(5, value))


class LLMJudge:
    """LLM-as-judge wrapper with strict structured output."""

    def __init__(self, model: str = "deepseek-chat"):
        self.client = DeepSeekClient(model=model, temperature=0.0, max_tokens=900)

    def judge(
        self,
        *,
        question: str,
        reference_answer: str,
        candidate_answer: str,
        retrieved_context: list[dict[str, Any]],
        sources: list[dict[str, Any]],
        must_include: list[str],
    ) -> JudgeScores:
        prompt = self._build_prompt(
            question=question,
            reference_answer=reference_answer,
            candidate_answer=candidate_answer,
            retrieved_context=retrieved_context,
            sources=sources,
            must_include=must_include,
        )

        response = self.client.generate_with_chat(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=900,
        )
        payload = _extract_json_object(response)

        hallucination_value = str(payload.get("hallucination", "false")).strip().lower()
        hallucination = hallucination_value in {"true", "1", "yes"}

        verdict = str(payload.get("verdict", "fail")).strip().lower()
        if verdict not in {"pass", "fail"}:
            verdict = "fail"

        return JudgeScores(
            correctness=_coerce_score(payload.get("correctness")),
            groundedness=_coerce_score(payload.get("groundedness")),
            completeness=_coerce_score(payload.get("completeness")),
            citation_quality=_coerce_score(payload.get("citation_quality")),
            hallucination=hallucination,
            verdict=verdict,
            reasoning=str(payload.get("reasoning", "")).strip(),
        )

    @staticmethod
    def _build_prompt(
        *,
        question: str,
        reference_answer: str,
        candidate_answer: str,
        retrieved_context: list[dict[str, Any]],
        sources: list[dict[str, Any]],
        must_include: list[str],
    ) -> str:
        rubric = """
Score each dimension from 1 (very poor) to 5 (excellent):
- correctness: matches the expected meaning of the reference answer.
- groundedness: claims are supported by retrieved context.
- completeness: addresses all major parts of the question.
- citation_quality: cited sources are relevant and sufficient.

Set hallucination=true if important claims are unsupported or contradicted by context.
Set verdict=fail if groundedness <= 2 or hallucination=true; otherwise verdict=pass.
"""

        output_contract = """
Return JSON exactly with keys:
{
  "correctness": 1-5,
  "groundedness": 1-5,
  "completeness": 1-5,
  "citation_quality": 1-5,
  "hallucination": true|false,
  "verdict": "pass"|"fail",
  "reasoning": "short explanation"
}
"""

        payload = {
            "question": question,
            "reference_answer": reference_answer,
            "candidate_answer": candidate_answer,
            "must_include": must_include,
            "sources": sources,
            "retrieved_context": retrieved_context,
        }

        return (
            f"{rubric}\n"
            f"{output_contract}\n"
            f"Evaluation payload JSON:\n{json.dumps(payload, ensure_ascii=True, indent=2)}"
        )

