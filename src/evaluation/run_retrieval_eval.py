"""
Retrieval-only evaluation: frozen corpus (MinIO manifest) → index → metrics per configuration.

No LLM-as-judge. Use after `python -m src.evaluation.freeze_corpus`.
Gold docs come from frozen paper_ids in the snapshot manifest.
Gold passages come from the provided dataset JSONL.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from src.rag_storage import RAGStorage, clear_all_data, process_and_store_paper
from src.chunker import ChunkingStrategy
from src.rag_pipeline import ALL_STRATEGIES
from src.embedder import PaperEmbedder
from src.evaluation.corpus_storage import get_eval_pdf, get_manifest
from src.evaluation.determinism import init_deterministic
from src.evaluation.metrics import compute_retrieval_metrics_for_query
from src.chunk_retrieval import retrieve_chunks_with_metadata
from src.rag_constants import RETRIEVAL_CHUNK_TOP_K


def _parse_strategy(s: str) -> ChunkingStrategy:
    for st in ChunkingStrategy:
        if st.value == s:
            return st
    raise ValueError(f"Unknown chunking strategy: {s}")


def _load_dataset(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            example_id = str(data.get("id", f"ex_{idx}"))
            topics = str(data.get("topics", "")).strip()
            question = str(data.get("question", "")).strip()
            if not topics or not question:
                raise ValueError(f"Dataset row {example_id} must include non-empty topics and question.")
            rows.append(
                {
                    "id": example_id,
                    "topics": topics,
                    "question": question,
                    "gold_passages": [str(x) for x in data.get("gold_passages", [])],
                }
            )
    return rows


def _build_examples_from_dataset_and_manifest(
    dataset_rows: list[dict[str, Any]],
    manifest_queries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    for q in manifest_queries:
        qid = str(q.get("id", "")).strip()
        topics = str(q.get("topics", "")).strip()
        question = str(q.get("question", "")).strip()
        if qid:
            by_id[qid] = q
        if topics and question:
            by_pair[(topics, question)] = q

    examples: list[dict[str, Any]] = []
    for row in dataset_rows:
        q = by_id.get(row["id"]) or by_pair.get((row["topics"], row["question"]))
        if not q:
            raise SystemExit(
                f"No frozen query mapping found for dataset row id='{row['id']}' "
                f"(topics='{row['topics']}', question='{row['question']}'). "
                "Run freeze_corpus first with the same dataset."
            )
        gold_docs = [str(x) for x in q.get("paper_ids", [])]
        if not gold_docs:
            raise SystemExit(
                f"Frozen mapping for row id='{row['id']}' has no paper_ids. "
                "Run freeze_corpus again."
            )
        examples.append(
            {
                "id": row["id"],
                "topics": row["topics"],
                "question": row["question"],
                "gold_docs": gold_docs,
                "gold_passages": row["gold_passages"],
            }
        )
    return examples


def _index_single_strategy(
    storage: RAGStorage,
    embedder: PaperEmbedder,
    papers_data: list[dict[str, Any]],
    strategy: ChunkingStrategy,
    embedding_dim: int,
) -> None:
    clear_all_data(storage, vector_size=embedding_dim)
    for paper_data in papers_data:
        process_and_store_paper(storage, embedder, paper_data, strategy, skip_abstract=True)


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _run_queries_for_config(
    storage: RAGStorage,
    embedder: PaperEmbedder,
    examples: list[dict[str, Any]],
    strategy: ChunkingStrategy,
    top_k: int,
    use_mmr: bool,
    expand_neighbors: bool,
    retrieval_type: str,
    re_ranking: bool,
) -> dict[str, float]:
    recalls: list[float] = []
    hits: list[float] = []
    precisions: list[float] = []
    mrrs: list[float] = []

    for ex in examples:
        retrieved = retrieve_chunks_with_metadata(
            storage,
            embedder,
            ex["question"],
            strategy,
            top_k=top_k,
            use_mmr=use_mmr,
            expand_neighbors=expand_neighbors,
            retrieval_type=retrieval_type,  # type: ignore[arg-type]
            re_ranking=re_ranking,
        )
        m = compute_retrieval_metrics_for_query(
            retrieved,
            ex["gold_docs"],
            ex["gold_passages"],
            embedder,
            k=top_k,
        )
        recalls.append(m["recall_at_k"])
        hits.append(m["hit_at_k"])
        precisions.append(m["precision_at_k"])
        mrrs.append(m["mrr"])

    return {
        "recall_at_k": _mean(recalls),
        "hit_at_k": _mean(hits),
        "precision_at_k": _mean(precisions),
        "mrr": _mean(mrrs),
    }


def run_phase1(
    storage: RAGStorage,
    embedder: PaperEmbedder,
    papers_data: list[dict[str, Any]],
    examples: list[dict[str, Any]],
    embedding_dim: int,
    use_mmr: bool,
    expand_neighbors: bool,
) -> tuple[list[dict[str, Any]], ChunkingStrategy]:
    """
    Sweep chunking strategies; fixed dense retrieval, no rerank, fixed top_k.
    """
    phase1_top_k = RETRIEVAL_CHUNK_TOP_K
    results: list[dict[str, Any]] = []

    for strategy in ALL_STRATEGIES:
        _index_single_strategy(storage, embedder, papers_data, strategy, embedding_dim)
        metrics = _run_queries_for_config(
            storage,
            embedder,
            examples,
            strategy,
            top_k=phase1_top_k,
            use_mmr=use_mmr,
            expand_neighbors=expand_neighbors,
            retrieval_type="dense",
            re_ranking=False,
        )
        row = {
            "config": {
                "chunking_strategy": strategy.value,
                "top_k": phase1_top_k,
                "retrieval_type": "dense",
                "re_ranking": False,
                "use_mmr": use_mmr,
                "expand_neighbors": expand_neighbors,
            },
            "metrics": metrics,
        }
        results.append(row)

    best = max(
        results,
        key=lambda r: (
            r["metrics"]["hit_at_k"],
            r["metrics"]["mrr"],
            r["metrics"]["precision_at_k"],
        ),
    )
    winner = _parse_strategy(best["config"]["chunking_strategy"])
    return results, winner


def run_phase2(
    storage: RAGStorage,
    embedder: PaperEmbedder,
    papers_data: list[dict[str, Any]],
    examples: list[dict[str, Any]],
    strategy: ChunkingStrategy,
    embedding_dim: int,
    use_mmr: bool,
    expand_neighbors: bool,
) -> list[dict[str, Any]]:
    _index_single_strategy(storage, embedder, papers_data, strategy, embedding_dim)

    results: list[dict[str, Any]] = []
    for top_k in (3, 5, 10):
        for retrieval_type in ("dense", "hybrid"):
            for re_ranking in (False, True):
                metrics = _run_queries_for_config(
                    storage,
                    embedder,
                    examples,
                    strategy,
                    top_k=top_k,
                    use_mmr=use_mmr,
                    expand_neighbors=expand_neighbors,
                    retrieval_type=retrieval_type,
                    re_ranking=re_ranking,
                )
                results.append(
                    {
                        "config": {
                            "chunking_strategy": strategy.value,
                            "top_k": top_k,
                            "retrieval_type": retrieval_type,
                            "re_ranking": re_ranking,
                            "use_mmr": use_mmr,
                            "expand_neighbors": expand_neighbors,
                        },
                        "metrics": metrics,
                    }
                )
    return results


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Retrieval evaluation (frozen corpus).")
    parser.add_argument("--dataset", type=Path, required=True, help="JSONL dataset path (topics, question, gold_passages).")
    parser.add_argument("--snapshot-id", type=str, required=True, help="MinIO snapshot id (see freeze_corpus).")
    parser.add_argument("--phase", type=str, choices=("1", "2", "all"), default="all")
    parser.add_argument(
        "--winner-strategy",
        type=str,
        default=None,
        help="For --phase 2 only: ChunkingStrategy value (e.g. STRUCTURE_AWARE_OVERLAP).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-mmr", action="store_true", help="Disable MMR for dense path.")
    parser.add_argument("--expand-neighbors", action="store_true", default=False)
    parser.add_argument("--output-dir", type=Path, default=Path("src/evaluation/output"))
    args = parser.parse_args()

    init_deterministic(args.seed)
    use_mmr = not args.no_mmr

    manifest = get_manifest(args.snapshot_id)
    manifest_queries = manifest.get("queries", [])
    if not manifest_queries:
        raise SystemExit(
            "Manifest has no queries. Run freeze_corpus first to create frozen mappings."
        )

    dataset_rows = _load_dataset(args.dataset)
    if not dataset_rows:
        raise SystemExit("Dataset is empty.")
    examples = _build_examples_from_dataset_and_manifest(dataset_rows, manifest_queries)

    # Union all paper IDs referenced by frozen query mappings.
    paper_ids: list[str] = []
    seen_ids: set[str] = set()
    for q in manifest_queries:
        for pid in q.get("paper_ids", []):
            pid_str = str(pid)
            if pid_str and pid_str not in seen_ids:
                seen_ids.add(pid_str)
                paper_ids.append(pid_str)
    if not paper_ids:
        raise SystemExit("No frozen papers found in manifest. Run freeze_corpus first.")

    papers_data: list[dict[str, Any]] = []
    for pid in paper_ids:
        pdf_bytes = get_eval_pdf(args.snapshot_id, pid)
        papers_data.append(
            {
                "metadata": {
                    "id": pid,
                    "title": "",
                    "abstract": "",
                    "authors": [],
                    "published": "",
                    "url": f"https://arxiv.org/abs/{pid}",
                    "categories": [],
                },
                "pdf_bytes": pdf_bytes,
            }
        )

    embedder = PaperEmbedder()
    emb = embedder.embed_texts(["dimension probe"])
    embedding_dim = int(emb.shape[1])

    storage = RAGStorage()
    storage.init_db()
    storage.init_bucket()
    storage.init_qdrant(vector_size=embedding_dim)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, Any] = {
        "dataset": str(args.dataset),
        "snapshot_id": args.snapshot_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "phase": args.phase,
    }

    winner: ChunkingStrategy | None = None

    if args.phase in ("1", "all"):
        phase1_rows, winner = run_phase1(
            storage,
            embedder,
            papers_data,
            examples,
            embedding_dim,
            use_mmr=use_mmr,
            expand_neighbors=args.expand_neighbors,
        )
        out["phase1"] = phase1_rows
        out["phase1_winner"] = winner.value

    if args.phase in ("2", "all"):
        if args.phase == "2":
            if not args.winner_strategy:
                raise SystemExit("--phase 2 requires --winner-strategy")
            strat = _parse_strategy(args.winner_strategy)
        else:
            if winner is None:
                raise SystemExit("internal: phase all missing winner")
            strat = winner
        phase2_rows = run_phase2(
            storage,
            embedder,
            papers_data,
            examples,
            strat,
            embedding_dim,
            use_mmr=use_mmr,
            expand_neighbors=args.expand_neighbors,
        )
        out["phase2"] = phase2_rows

    out_path = args.output_dir / "retrieval_eval.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
