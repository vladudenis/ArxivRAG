"""
Build a frozen evaluation corpus:
- full reset of one snapshot prefix in MinIO
- arXiv search + abstract filter using topics/question
- download PDFs
- write minimal manifest with query -> paper_ids mappings
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import arxiv
from dotenv import load_dotenv

from src.arxiv_retriever import ArxivRetriever
from src.embedder import PaperEmbedder
from src.rag_pipeline import topics_to_search_query
from src.rag_constants import ABSTRACT_FILTER_TOP_K, ARXIV_SEARCH_MAX_RESULTS
from src.evaluation.corpus_storage import clear_snapshot, put_eval_pdf, put_manifest
from src.evaluation.determinism import init_deterministic
from src.evaluation.schemas import EvaluationExample


def _load_dataset(path: Path) -> list[EvaluationExample]:
    rows: list[EvaluationExample] = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            rows.append(
                EvaluationExample(
                    example_id=str(data.get("id", f"ex_{idx}")),
                    topics=str(data.get("topics", "")).strip(),
                    question=str(data.get("question", "")).strip(),
                )
            )
    return rows


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Freeze arXiv papers for retrieval evaluation (MinIO).")
    parser.add_argument("--dataset", type=Path, required=True, help="JSONL dataset path.")
    parser.add_argument("--snapshot-id", type=str, required=True, help="Snapshot folder id (e.g. v1).")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for reproducible embedding order.")
    parser.add_argument(
        "--pause-between-queries-seconds",
        type=float,
        default=8.0,
        help="Cooldown between dataset queries to reduce arXiv throttling.",
    )
    parser.add_argument(
        "--max-query-retries",
        type=int,
        default=3,
        help="How many times to retry a query row on arXiv 429 before failing.",
    )
    args = parser.parse_args()

    init_deterministic(args.seed)
    deleted_count = clear_snapshot(args.snapshot_id)
    print(f"Reset snapshot '{args.snapshot_id}' (deleted {deleted_count} existing object(s)).")

    rows = _load_dataset(args.dataset)
    if not rows:
        raise SystemExit("Dataset is empty.")

    embedder = PaperEmbedder()
    retriever = ArxivRetriever(
        request_delay_seconds=4.0,
        max_rate_limit_retries=2,
        initial_backoff_seconds=6.0,
    )

    seen_paper_ids: set[str] = set()
    query_mappings: list[dict[str, str | list[str]]] = []

    for row_idx, item in enumerate(rows):
        if not item.topics or not item.question:
            raise ValueError(f"Row {item.example_id} needs topics and question.")

        search_q = topics_to_search_query(item.topics)
        papers = []
        last_err: Exception | None = None
        for attempt in range(args.max_query_retries + 1):
            try:
                papers = retriever.search(search_q, max_results=ARXIV_SEARCH_MAX_RESULTS)
                last_err = None
                break
            except arxiv.HTTPError as e:
                last_err = e
                is_rate_limit = "429" in str(e)
                is_last_attempt = attempt >= args.max_query_retries
                if (not is_rate_limit) or is_last_attempt:
                    raise
                cool_down = 60.0 * (2 ** attempt)
                print(
                    f"Row {item.example_id} rate-limited by arXiv. "
                    f"Cooling down for {cool_down:.0f}s before retry "
                    f"({attempt + 1}/{args.max_query_retries})..."
                )
                time.sleep(cool_down)
        if last_err is not None:
            raise last_err

        top_papers = retriever.filter_by_abstract_similarity(
            item.question, papers, embedder, top_k=ABSTRACT_FILTER_TOP_K
        )
        downloaded = retriever.download_top_k(top_papers, k=ABSTRACT_FILTER_TOP_K)

        frozen_ids: list[str] = []
        for pd in downloaded:
            meta = pd["metadata"]
            pid = meta["id"]
            frozen_ids.append(pid)
            if pid not in seen_paper_ids:
                seen_paper_ids.add(pid)
                put_eval_pdf(args.snapshot_id, pid, pd["pdf_bytes"])

        query_mappings.append(
            {
                "id": item.example_id,
                "topics": item.topics,
                "question": item.question,
                "paper_ids": frozen_ids,
            }
        )

        # Give arXiv a cooldown window between rows.
        if row_idx < len(rows) - 1:
            time.sleep(max(0.0, args.pause_between_queries_seconds))

    manifest = {
        "snapshot_id": args.snapshot_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "arxiv_search_max_results": ARXIV_SEARCH_MAX_RESULTS,
        "abstract_filter_top_k": ABSTRACT_FILTER_TOP_K,
        "queries": query_mappings,
    }
    put_manifest(args.snapshot_id, manifest)
    print(f"Frozen {len(seen_paper_ids)} unique papers to MinIO snapshot '{args.snapshot_id}'.")
    print(f"Wrote manifest.json with {len(query_mappings)} query mapping(s).")


if __name__ == "__main__":
    main()
