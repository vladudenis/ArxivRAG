"""
Retrieval-only evaluation: frozen corpus (MinIO manifest) → index → metrics per configuration.

No LLM-as-judge. Use after `python -m src.evaluation.freeze_corpus`.
Gold docs come from frozen paper_ids in the snapshot manifest.
Gold passages come from the provided dataset JSONL.

Metrics: Recall@k, Precision@k, MRR, nDCG@k (graded), MAP. Each configuration also
reports per-query scores and bootstrap 95% confidence intervals. Optional Phase 3
compares embedding models under a fixed relevance judge. Pre-registered comparisons
are tested for significance (paired bootstrap + Wilcoxon).
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

from src.rag_storage import RAGStorage, clear_all_data, process_and_store_paper
from src.chunker import ChunkingStrategy
from src.rag_pipeline import ALL_STRATEGIES
from src.embedder import PaperEmbedder
from src.evaluation.corpus_storage import get_eval_pdf, get_manifest
from src.evaluation.determinism import init_deterministic
from src.evaluation.metrics import METRIC_KEYS, compute_retrieval_metrics_for_query
from src.evaluation.stats import bootstrap_ci, paired_bootstrap_test, wilcoxon_signed_rank
from src.chunk_retrieval import retrieve_chunks_with_metadata
from src.rag_constants import RETRIEVAL_CHUNK_TOP_K

# Retrieval model used for the default run and as the default fixed relevance judge.
BASELINE_EMBEDDING_MODEL = "google/embeddinggemma-300m"

# Config used per embedding model in Phase 3 when no Phase 2 winner is available.
DEFAULT_PHASE3_CONFIG = {"top_k": 10, "retrieval_type": "dense", "re_ranking": False}


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


def _make_row(config: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    """Assemble an output row: config + mean metrics + CIs + per-query scores."""
    return {
        "config": config,
        "metrics": result["metrics"],
        "metrics_ci": result["metrics_ci"],
        "per_query": result["per_query"],
    }


def _run_queries_for_config(
    storage: RAGStorage,
    embedder: PaperEmbedder,
    judge_embedder: PaperEmbedder,
    examples: list[dict[str, Any]],
    strategy: ChunkingStrategy,
    top_k: int,
    use_mmr: bool,
    expand_neighbors: bool,
    retrieval_type: str,
    re_ranking: bool,
    seed: int,
) -> dict[str, Any]:
    """Run all queries for one config; return mean metrics, CIs, and per-query scores."""
    per_query: dict[str, list[float]] = {key: [] for key in METRIC_KEYS}

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
            judge_embedder=judge_embedder,
        )
        for key in METRIC_KEYS:
            per_query[key].append(m[key])

    means = {key: _mean(per_query[key]) for key in METRIC_KEYS}
    cis = {key: bootstrap_ci(per_query[key], seed=seed) for key in METRIC_KEYS}
    return {"metrics": means, "metrics_ci": cis, "per_query": per_query}


def run_phase1(
    storage: RAGStorage,
    embedder: PaperEmbedder,
    judge_embedder: PaperEmbedder,
    papers_data: list[dict[str, Any]],
    examples: list[dict[str, Any]],
    embedding_dim: int,
    use_mmr: bool,
    expand_neighbors: bool,
    seed: int,
) -> tuple[list[dict[str, Any]], ChunkingStrategy]:
    """Sweep chunking strategies; fixed dense retrieval, no rerank, fixed top_k."""
    phase1_top_k = RETRIEVAL_CHUNK_TOP_K
    results: list[dict[str, Any]] = []

    for strategy in ALL_STRATEGIES:
        _index_single_strategy(storage, embedder, papers_data, strategy, embedding_dim)
        result = _run_queries_for_config(
            storage,
            embedder,
            judge_embedder,
            examples,
            strategy,
            top_k=phase1_top_k,
            use_mmr=use_mmr,
            expand_neighbors=expand_neighbors,
            retrieval_type="dense",
            re_ranking=False,
            seed=seed,
        )
        config = {
            "chunking_strategy": strategy.value,
            "top_k": phase1_top_k,
            "retrieval_type": "dense",
            "re_ranking": False,
            "use_mmr": use_mmr,
            "expand_neighbors": expand_neighbors,
        }
        results.append(_make_row(config, result))

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
    judge_embedder: PaperEmbedder,
    papers_data: list[dict[str, Any]],
    examples: list[dict[str, Any]],
    strategy: ChunkingStrategy,
    embedding_dim: int,
    use_mmr: bool,
    expand_neighbors: bool,
    seed: int,
) -> list[dict[str, Any]]:
    _index_single_strategy(storage, embedder, papers_data, strategy, embedding_dim)

    results: list[dict[str, Any]] = []
    for top_k in (3, 5, 10):
        for retrieval_type in ("dense", "hybrid"):
            for re_ranking in (False, True):
                result = _run_queries_for_config(
                    storage,
                    embedder,
                    judge_embedder,
                    examples,
                    strategy,
                    top_k=top_k,
                    use_mmr=use_mmr,
                    expand_neighbors=expand_neighbors,
                    retrieval_type=retrieval_type,
                    re_ranking=re_ranking,
                    seed=seed,
                )
                config = {
                    "chunking_strategy": strategy.value,
                    "top_k": top_k,
                    "retrieval_type": retrieval_type,
                    "re_ranking": re_ranking,
                    "use_mmr": use_mmr,
                    "expand_neighbors": expand_neighbors,
                }
                results.append(_make_row(config, result))
    return results


def _best_config_from_phase2(rows: list[dict[str, Any]]) -> dict[str, Any]:
    best = max(
        rows,
        key=lambda r: (
            r["metrics"]["hit_at_k"],
            r["metrics"]["mrr"],
            r["metrics"]["precision_at_k"],
        ),
    )
    c = best["config"]
    return {
        "top_k": c["top_k"],
        "retrieval_type": c["retrieval_type"],
        "re_ranking": c["re_ranking"],
    }


def run_phase3(
    storage: RAGStorage,
    base_embedder: PaperEmbedder,
    judge_embedder: PaperEmbedder,
    papers_data: list[dict[str, Any]],
    examples: list[dict[str, Any]],
    strategy: ChunkingStrategy,
    best_config: dict[str, Any],
    embedding_models: list[str],
    use_mmr: bool,
    expand_neighbors: bool,
    seed: int,
) -> list[dict[str, Any]]:
    """
    Compare embedding models under the winning chunker + best retrieval config.

    Relevance is scored with the fixed judge_embedder so labels are identical
    across retrieval models. Each model triggers a full re-embed + re-index.
    """
    results: list[dict[str, Any]] = []
    for model_name in embedding_models:
        if model_name == BASELINE_EMBEDDING_MODEL:
            r_embedder = base_embedder
        else:
            try:
                r_embedder = PaperEmbedder(model_name=model_name)
            except Exception as e:  # noqa: BLE001 - skip a bad model, keep the sweep alive
                print(f"Skipping embedding model '{model_name}': {e}")
                continue

        dim = int(r_embedder.embed_texts(["dimension probe"]).shape[1])
        _index_single_strategy(storage, r_embedder, papers_data, strategy, dim)

        result = _run_queries_for_config(
            storage,
            r_embedder,
            judge_embedder,
            examples,
            strategy,
            top_k=best_config["top_k"],
            use_mmr=use_mmr,
            expand_neighbors=expand_neighbors,
            retrieval_type=best_config["retrieval_type"],
            re_ranking=best_config["re_ranking"],
            seed=seed,
        )
        config = {
            "embedding_model": model_name,
            "embedding_dim": dim,
            "chunking_strategy": strategy.value,
            "top_k": best_config["top_k"],
            "retrieval_type": best_config["retrieval_type"],
            "re_ranking": best_config["re_ranking"],
            "use_mmr": use_mmr,
            "expand_neighbors": expand_neighbors,
        }
        results.append(_make_row(config, result))
    return results


def _find_row(rows: list[dict[str, Any]], **conds: Any) -> Optional[dict[str, Any]]:
    for r in rows:
        c = r["config"]
        if all(c.get(key) == val for key, val in conds.items()):
            return r
    return None


def _compare(name: str, row_a: dict[str, Any], row_b: dict[str, Any], metric: str, seed: int) -> dict[str, Any]:
    a = row_a["per_query"][metric]
    b = row_b["per_query"][metric]
    return {
        "name": name,
        "metric": metric,
        "a_config": row_a["config"],
        "b_config": row_b["config"],
        "a_mean": row_a["metrics"][metric],
        "b_mean": row_b["metrics"][metric],
        "paired_bootstrap": paired_bootstrap_test(a, b, seed=seed),
        "wilcoxon": wilcoxon_signed_rank(a, b),
    }


def _significance_tests(out: dict[str, Any], seed: int) -> list[dict[str, Any]]:
    """Pre-registered comparisons: keep the set small to limit multiple testing."""
    tests: list[dict[str, Any]] = []

    def _has_per_query(rows: list[dict[str, Any]]) -> bool:
        return bool(rows) and all(isinstance(r.get("per_query"), dict) for r in rows)

    p2 = out.get("phase2")
    if p2 and _has_per_query(p2):
        ranked = sorted(
            p2,
            key=lambda r: (
                r["metrics"]["hit_at_k"],
                r["metrics"]["mrr"],
                r["metrics"]["precision_at_k"],
            ),
            reverse=True,
        )
        if len(ranked) >= 2:
            tests.append(_compare("phase2_best_vs_second_hit", ranked[0], ranked[1], "hit_at_k", seed))
        dense10 = _find_row(p2, top_k=10, retrieval_type="dense", re_ranking=False)
        hybrid10 = _find_row(p2, top_k=10, retrieval_type="hybrid", re_ranking=False)
        if dense10 and hybrid10:
            tests.append(_compare("phase2_dense_vs_hybrid_topk10_hit", dense10, hybrid10, "hit_at_k", seed))

    p3 = out.get("phase3")
    if p3 and len(p3) >= 2 and _has_per_query(p3):
        baseline = _find_row(p3, embedding_model=BASELINE_EMBEDDING_MODEL)
        # Compare the baseline embedder against each other model (hit@k and MAP).
        if baseline is not None:
            for r in p3:
                if r is baseline:
                    continue
                model = r["config"].get("embedding_model", "model")
                tests.append(_compare(f"phase3_baseline_vs_{model}_hit", baseline, r, "hit_at_k", seed))
                tests.append(_compare(f"phase3_baseline_vs_{model}_map", baseline, r, "average_precision", seed))

    return tests


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Retrieval evaluation (frozen corpus).")
    parser.add_argument("--dataset", type=Path, required=True, help="JSONL dataset path (topics, question, gold_passages).")
    parser.add_argument("--snapshot-id", type=str, required=True, help="MinIO snapshot id (see freeze_corpus).")
    parser.add_argument(
        "--phase",
        type=str,
        choices=("1", "2", "3", "all"),
        default="all",
        help="1=chunker sweep, 2=retrieval sweep, 3=embedding-model comparison only "
        "(reuses existing results and merges in a phase3 block), all=phases 1+2 (+phase3 "
        "if --embedding-models is given).",
    )
    parser.add_argument(
        "--winner-strategy",
        type=str,
        default=None,
        help="Required for --phase 2 and --phase 3: ChunkingStrategy value "
        "(e.g. FIXED_WINDOW_OVERLAP).",
    )
    parser.add_argument(
        "--embedding-models",
        type=str,
        default=None,
        help="Comma-separated embedding model names to compare in Phase 3 (e.g. "
        "'google/embeddinggemma-300m,BAAI/bge-small-en-v1.5'). Requires a winner chunker.",
    )
    parser.add_argument(
        "--judge-embedding-model",
        type=str,
        default=BASELINE_EMBEDDING_MODEL,
        help="Fixed embedder used to score gold-passage relevance (kept constant across "
        "retrieval models so labels do not drift). Defaults to the baseline model.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-mmr", action="store_true", help="Disable MMR for dense path.")
    parser.add_argument("--expand-neighbors", action="store_true", default=False)
    parser.add_argument("--output-dir", type=Path, default=Path("src/evaluation/output"))
    args = parser.parse_args()

    init_deterministic(args.seed)
    use_mmr = not args.no_mmr

    embedding_models: Optional[list[str]] = None
    if args.embedding_models:
        embedding_models = [m.strip() for m in args.embedding_models.split(",") if m.strip()]

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

    # Fixed relevance judge: reuse the retrieval embedder when the judge model matches
    # the baseline, otherwise load a dedicated one.
    if args.judge_embedding_model == BASELINE_EMBEDDING_MODEL:
        judge_embedder = embedder
    else:
        judge_embedder = PaperEmbedder(model_name=args.judge_embedding_model)

    storage = RAGStorage()
    storage.init_db()
    storage.init_bucket()
    storage.init_qdrant(vector_size=embedding_dim)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "retrieval_eval.json"
    out: dict[str, Any] = {
        "dataset": str(args.dataset),
        "snapshot_id": args.snapshot_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "phase": args.phase,
        "seed": args.seed,
        "judge_embedding_model": args.judge_embedding_model,
        "metric_keys": list(METRIC_KEYS),
        "example_ids": [ex["id"] for ex in examples],
    }

    # Phase 3 only: reuse prior Phase 1/2 results (merge) instead of recomputing them.
    if args.phase == "3":
        if not embedding_models:
            raise SystemExit("--phase 3 requires --embedding-models.")
        if not args.winner_strategy:
            raise SystemExit("--phase 3 requires --winner-strategy (the Phase 1 winning chunker).")
        if out_path.exists():
            try:
                existing = json.loads(out_path.read_text(encoding="utf-8"))
                for key in ("phase1", "phase1_winner", "phase2"):
                    if key in existing:
                        out[key] = existing[key]
                out["merged_from_created_utc"] = existing.get("created_utc")
            except Exception as e:  # noqa: BLE001 - fall back to a phase3-only file
                print(f"Could not merge existing results ({e}); writing a phase3-only file.")
        else:
            print(f"No existing {out_path.name} to merge; writing a phase3-only file.")

    winner: ChunkingStrategy | None = None

    if args.phase in ("1", "all"):
        phase1_rows, winner = run_phase1(
            storage,
            embedder,
            judge_embedder,
            papers_data,
            examples,
            embedding_dim,
            use_mmr=use_mmr,
            expand_neighbors=args.expand_neighbors,
            seed=args.seed,
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
            judge_embedder,
            papers_data,
            examples,
            strat,
            embedding_dim,
            use_mmr=use_mmr,
            expand_neighbors=args.expand_neighbors,
            seed=args.seed,
        )
        out["phase2"] = phase2_rows

    if embedding_models:
        if winner is not None:
            strat3 = winner
        elif args.winner_strategy:
            strat3 = _parse_strategy(args.winner_strategy)
        else:
            raise SystemExit(
                "--embedding-models needs a winning chunker: run with --phase 1/all "
                "or pass --winner-strategy."
            )
        best_config = _best_config_from_phase2(out["phase2"]) if out.get("phase2") else dict(DEFAULT_PHASE3_CONFIG)
        phase3_rows = run_phase3(
            storage,
            embedder,
            judge_embedder,
            papers_data,
            examples,
            strat3,
            best_config,
            embedding_models,
            use_mmr=use_mmr,
            expand_neighbors=args.expand_neighbors,
            seed=args.seed,
        )
        out["phase3"] = phase3_rows

    out["significance"] = _significance_tests(out, seed=args.seed)

    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
