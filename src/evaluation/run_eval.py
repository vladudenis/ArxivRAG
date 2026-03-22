from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

from src.evaluation.judge import LLMJudge
from src.evaluation.metrics import (
    compute_must_include_hit_rate,
    compute_recall_at_k,
    compute_simple_response_metrics,
    compute_text_overlap_metrics,
)
from src.evaluation.schemas import EvaluationExample


def _load_dataset(dataset_path: Path) -> list[EvaluationExample]:
    rows: list[EvaluationExample] = []
    with dataset_path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)

            example_id = data.get("id", f"example_{idx}")
            topics = str(data.get("topics", "")).strip()
            question = str(data.get("question", "")).strip()
            if not topics or not question:
                raise ValueError(
                    f"Dataset row {idx} must include non-empty 'topics' and 'question'."
                )

            rows.append(
                EvaluationExample(
                    example_id=example_id,
                    topics=topics,
                    question=question,
                    reference_answer=str(data.get("reference_answer", "")).strip(),
                    must_include=[str(x) for x in data.get("must_include", [])],
                    relevant_papers=[str(x) for x in data.get("relevant_papers", [])],
                    metadata=data.get("metadata", {}),
                )
            )
    return rows


def _check_api_health(base_url: str, timeout_s: float = 15.0) -> None:
    """Verify the API is reachable before running evaluation."""
    try:
        response = requests.get(
            f"{base_url.rstrip('/')}/health",
            timeout=(5.0, timeout_s),
        )
        response.raise_for_status()
    except requests.exceptions.Timeout as e:
        raise RuntimeError(
            f"API at {base_url} did not respond in time. It may be busy or overloaded. "
            "Ensure the API is idle and try again."
        ) from e
    except requests.exceptions.ConnectionError as e:
        raise RuntimeError(
            f"Cannot connect to API at {base_url}. Start it with: uvicorn src.api.main:app --reload --port 8000"
        ) from e
    except requests.RequestException as e:
        raise RuntimeError(f"Cannot reach API at {base_url}: {e}") from e


def _call_query_api(base_url: str, topics: str, question: str, timeout_s: float) -> dict[str, Any]:
    try:
        response = requests.post(
            f"{base_url.rstrip('/')}/query",
            json={
                "topics": topics,
                "query": question,
                "include_debug_context": True,
            },
            timeout=timeout_s,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ReadTimeout as e:
        raise RuntimeError(
            f"API read timed out after {timeout_s}s. Each RAG run (search + download + 4 LLM calls) can take several minutes. "
            f"Retry with: --timeout-s 600"
        ) from e


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["strategy"]].append(record)

    summary: dict[str, Any] = {"strategies": {}, "global": {}}

    total = len(records) if records else 1
    global_hallucination = sum(
        1 for r in records if r["judge"]["hallucination"]
    ) / total
    global_pass_rate = sum(
        1 for r in records if r["judge"]["verdict"] == "pass"
    ) / total
    summary["global"] = {
        "total_records": len(records),
        "hallucination_rate": round(global_hallucination, 4),
        "pass_rate": round(global_pass_rate, 4),
    }

    for strategy, strategy_rows in grouped.items():
        count = len(strategy_rows)
        if count == 0:
            continue

        avg = lambda key: sum(r[key] for r in strategy_rows) / count
        avg_judge = lambda key: sum(r["judge"][key] for r in strategy_rows) / count
        hall_rate = sum(1 for r in strategy_rows if r["judge"]["hallucination"]) / count
        pass_rate = sum(1 for r in strategy_rows if r["judge"]["verdict"] == "pass") / count

        summary["strategies"][strategy] = {
            "count": count,
            "avg_composite_score": round(avg("composite_score"), 4),
            "avg_correctness": round(avg_judge("correctness"), 4),
            "avg_groundedness": round(avg_judge("groundedness"), 4),
            "avg_completeness": round(avg_judge("completeness"), 4),
            "avg_citation_quality": round(avg_judge("citation_quality"), 4),
            "avg_must_include_hit_rate": round(avg("must_include_hit_rate"), 4),
            "avg_recall_at_k": round(avg("recall_at_k"), 4),
            "avg_rouge_l_f1": round(avg("rouge_l_f1"), 4),
            "avg_bleu": round(avg("bleu"), 4),
            "avg_latency_ms": round(avg("latency_ms"), 2),
            "hallucination_rate": round(hall_rate, 4),
            "pass_rate": round(pass_rate, 4),
        }

    return summary


def _write_summary_markdown(
    path: Path,
    summary: dict[str, Any],
    run_timestamp: str,
    num_examples: int,
) -> None:
    """Write a clear evaluation report (MD file) with metrics and comparison table."""
    lines: list[str] = []
    lines.append("# ArxivRAG Evaluation Report")
    lines.append("")
    lines.append(f"**Generated:** {run_timestamp}")
    lines.append("")
    lines.append("---")
    lines.append("")

    global_data = summary.get("global", {})
    strategies_data = summary.get("strategies", {})

    lines.append("## Summary")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Examples evaluated | {num_examples} |")
    lines.append(f"| Total records (examples × strategies) | {global_data.get('total_records', 0)} |")
    lines.append(f"| Overall pass rate | {global_data.get('pass_rate', 0.0):.1%} |")
    lines.append(f"| Overall hallucination rate | {global_data.get('hallucination_rate', 0.0):.1%} |")
    lines.append("")

    if strategies_data:
        best_strategy = max(
            strategies_data.items(),
            key=lambda x: (x[1]["avg_composite_score"], -x[1]["hallucination_rate"]),
        )
        lines.append(f"**Best strategy by composite score:** `{best_strategy[0]}` "
                    f"(score: {best_strategy[1]['avg_composite_score']:.3f})")
        lines.append("")

    lines.append("## Strategy Comparison")
    lines.append("")
    lines.append("| Strategy | Composite | Groundedness | Correctness | Completeness | Citation | Pass | Halluc. | Latency (ms) |")
    lines.append("|----------|-----------|--------------|-------------|---------------|----------|------|---------|---------------|")
    for strategy, stats in sorted(strategies_data.items(), key=lambda x: -x[1]["avg_composite_score"]):
        lines.append(
            f"| {strategy} | {stats['avg_composite_score']:.3f} | "
            f"{stats['avg_groundedness']:.2f} | {stats['avg_correctness']:.2f} | "
            f"{stats['avg_completeness']:.2f} | {stats['avg_citation_quality']:.2f} | "
            f"{stats['pass_rate']:.0%} | {stats['hallucination_rate']:.0%} | "
            f"{stats['avg_latency_ms']:.0f} |"
        )
    lines.append("")

    lines.append("## Additional Metrics by Strategy")
    lines.append("")
    lines.append("| Strategy | Must-include hit | Recall@k | ROUGE-L | BLEU |")
    lines.append("|----------|------------------|----------|---------|------|")
    for strategy, stats in sorted(strategies_data.items(), key=lambda x: -x[1]["avg_composite_score"]):
        lines.append(
            f"| {strategy} | {stats['avg_must_include_hit_rate']:.2%} | "
            f"{stats['avg_recall_at_k']:.2%} | {stats['avg_rouge_l_f1']:.3f} | "
            f"{stats['avg_bleu']:.2f} |"
        )
    lines.append("")

    lines.append("## Per-Strategy Details")
    lines.append("")
    for strategy, stats in sorted(strategies_data.items(), key=lambda x: -x[1]["avg_composite_score"]):
        lines.append(f"### {strategy}")
        lines.append("")
        lines.append(f"- **Composite score:** {stats['avg_composite_score']:.4f}")
        lines.append(f"- **LLM judge:** correctness={stats['avg_correctness']:.2f}, groundedness={stats['avg_groundedness']:.2f}, "
                    f"completeness={stats['avg_completeness']:.2f}, citation_quality={stats['avg_citation_quality']:.2f}")
        lines.append(f"- **Pass rate:** {stats['pass_rate']:.1%} | **Hallucination rate:** {stats['hallucination_rate']:.1%}")
        lines.append(f"- **Must-include hit rate:** {stats['avg_must_include_hit_rate']:.2%}")
        lines.append(f"- **Recall@k:** {stats['avg_recall_at_k']:.2%}")
        lines.append(f"- **ROUGE-L F1:** {stats['avg_rouge_l_f1']:.4f} | **BLEU:** {stats['avg_bleu']:.2f}")
        lines.append(f"- **Avg latency:** {stats['avg_latency_ms']:.2f} ms")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def _default_dataset_path() -> Path:
    """Resolve dataset path relative to this module so it works from any cwd."""
    return Path(__file__).resolve().parent / "dataset.template.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LLM-as-judge evaluation for ArxivRAG.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Path to dataset JSONL. Default: src/evaluation/dataset.template.jsonl",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:8000",
        help="FastAPI base URL.",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="deepseek-chat",
        help="Model used for judge scoring.",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=600.0,
        help="Timeout in seconds for each /query request (default 600; full RAG runs can take several minutes).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("src/evaluation/output"),
        help="Directory for detailed results and summary.",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    dataset_path = args.dataset or _default_dataset_path()
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}. "
            "Add evaluation examples or pass --dataset /path/to/your.jsonl"
        )

    dataset_rows = _load_dataset(dataset_path)
    if not dataset_rows:
        raise ValueError("Dataset is empty. Add at least one JSONL row.")

    _check_api_health(args.base_url)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    judge = LLMJudge(model=args.judge_model)

    results: list[dict[str, Any]] = []

    for item in dataset_rows:
        started = time.perf_counter()
        payload = _call_query_api(
            base_url=args.base_url,
            topics=item.topics,
            question=item.question,
            timeout_s=args.timeout_s,
        )
        query_latency_ms = (time.perf_counter() - started) * 1000.0

        strategy_results = payload.get("results", [])
        if not strategy_results:
            continue

        for strategy_result in strategy_results:
            answer = str(strategy_result.get("answer", ""))
            sources = strategy_result.get("sources", []) or []
            debug_context = strategy_result.get("debug_context", []) or []

            judge_scores = judge.judge(
                question=item.question,
                reference_answer=item.reference_answer,
                candidate_answer=answer,
                retrieved_context=debug_context,
                sources=sources,
                must_include=item.must_include,
            )

            simple_metrics = compute_simple_response_metrics(answer, sources)
            text_metrics = compute_text_overlap_metrics(item.reference_answer, answer)
            must_include_hit_rate = compute_must_include_hit_rate(answer, item.must_include)
            recall_at_k = compute_recall_at_k(sources, item.relevant_papers)

            row = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "example_id": item.example_id,
                "topics": item.topics,
                "question": item.question,
                "strategy": strategy_result.get("strategy", "unknown"),
                "strategy_label": strategy_result.get("strategy_label", "unknown"),
                "answer": answer,
                "sources": sources,
                "debug_context": debug_context,
                "latency_ms": round(query_latency_ms, 2),
                "must_include_hit_rate": round(must_include_hit_rate, 4),
                "recall_at_k": round(recall_at_k, 4),
                "rouge_l_f1": round(text_metrics["rouge_l_f1"], 4),
                "bleu": round(text_metrics["bleu"], 4),
                "simple_metrics": simple_metrics,
                "judge": {
                    "correctness": judge_scores.correctness,
                    "groundedness": judge_scores.groundedness,
                    "completeness": judge_scores.completeness,
                    "citation_quality": judge_scores.citation_quality,
                    "hallucination": judge_scores.hallucination,
                    "verdict": judge_scores.verdict,
                    "reasoning": judge_scores.reasoning,
                },
                "composite_score": round(judge_scores.composite_score, 4),
            }
            results.append(row)

    results_path = args.output_dir / "results.jsonl"
    _write_jsonl(results_path, results)

    summary = _aggregate(results)
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    run_timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    report_path = args.output_dir / "report.md"
    _write_summary_markdown(
        report_path,
        summary,
        run_timestamp=run_timestamp,
        num_examples=len(dataset_rows),
    )

    print(f"Finished evaluation. Records: {len(results)}")
    print(f"Detailed results: {results_path}")
    print(f"Summary JSON: {summary_path}")
    print(f"Report (MD): {report_path}")


if __name__ == "__main__":
    main()

