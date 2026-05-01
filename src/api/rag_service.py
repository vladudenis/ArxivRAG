"""
RAG service: orchestrates the full pipeline and extracts cited sources.
"""
from __future__ import annotations

from typing import List, Dict, Any, Optional

from src.api.logger import RAGLogger
from src.rag_storage import RAGStorage, process_and_store_paper, clear_all_data
from src.embedder import PaperEmbedder
from src.chunker import ChunkingStrategy
from src.api.llm_client import DeepSeekClient
from src.arxiv_retriever import ArxivRetriever
from src.chunk_retrieval import retrieve_chunks_with_metadata
from src.rag_pipeline import ALL_STRATEGIES, format_strategy_label, topics_to_search_query
from src.rag_constants import (
    ABSTRACT_FILTER_TOP_K,
    ARXIV_SEARCH_MAX_RESULTS,
    RETRIEVAL_CHUNK_TOP_K,
)


def get_sources_from_chunks(
    chunks_with_meta: List[tuple[str, Dict[str, Any]]],
) -> List[Dict[str, str]]:
    """
    Extract unique papers from chunks (by paper_id).
    Returns list of {paper_id, title, section, arxiv_url}.
    """
    if not chunks_with_meta:
        return []

    seen_paper_ids: set[str] = set()
    sources: List[Dict[str, str]] = []

    for _, payload in chunks_with_meta:
        paper_id = payload.get("paper_id", "")
        if not paper_id or paper_id in seen_paper_ids:
            continue
        seen_paper_ids.add(paper_id)

        title = payload.get("title", "")
        section = payload.get("section") or ""

        sources.append(
            {
                "paper_id": paper_id,
                "title": title,
                "section": section,
                "arxiv_url": f"https://arxiv.org/abs/{paper_id}",
            }
        )

    return sources


def _empty_results(message: str) -> Dict[str, Any]:
    """Return results list with one fallback entry when pipeline fails early."""
    return {
        "results": [
            {
                "strategy": ChunkingStrategy.STRUCTURE_AWARE_OVERLAP.value,
                "strategy_label": format_strategy_label(ChunkingStrategy.STRUCTURE_AWARE_OVERLAP),
                "answer": message,
                "sources": [],
            }
        ]
    }


class RAGService:
    """Orchestrates the full RAG pipeline for a single query."""

    def __init__(
        self,
        storage: RAGStorage,
        embedder: PaperEmbedder,
        retriever: ArxivRetriever,
        llm_client: DeepSeekClient,
        embedding_dim: int = 768,
        logger: Optional[RAGLogger] = None,
    ):
        self.storage = storage
        self.embedder = embedder
        self.retriever = retriever
        self.llm_client = llm_client
        self.embedding_dim = embedding_dim
        self.logger = logger or RAGLogger()

    def query(self, query: str, topics: str, include_debug_context: bool = False) -> Dict[str, Any]:
        """
        Run full RAG pipeline for all chunking strategies.
        Returns {results: [{strategy, strategy_label, answer, sources, debug_context?}, ...]}.
        """
        self.logger.log_step(0, "Starting RAG pipeline", query=query[:80], topics=topics[:80])

        # 1. Search arXiv using topics (comma-separated; multi-word terms quoted for phrase search)
        search_terms = topics_to_search_query(topics)
        self.logger.log_step(1, "Search arXiv", max_results=ARXIV_SEARCH_MAX_RESULTS)
        papers = self.retriever.search(search_terms, max_results=ARXIV_SEARCH_MAX_RESULTS)
        if not papers:
            self.logger.log_step(1, "Search arXiv", found=0)
            return _empty_results("No papers found for this query.")
        self.logger.log_step(1, "Search arXiv", found=len(papers))

        # 2. Filter by abstract cosine similarity to query
        self.logger.log_step(2, "Filter by abstract similarity", top_k=ABSTRACT_FILTER_TOP_K)
        top_papers = self.retriever.filter_by_abstract_similarity(
            query, papers, self.embedder, top_k=ABSTRACT_FILTER_TOP_K
        )
        if not top_papers:
            self.logger.log_step(2, "Filter by abstract similarity", kept=0)
            return _empty_results("No relevant papers after filtering.")
        self.logger.log_step(2, "Filter by abstract similarity", kept=len(top_papers))

        # 3. Download filtered papers
        self.logger.log_step(3, "Download top papers", k=ABSTRACT_FILTER_TOP_K)
        paper_data_list = self.retriever.download_top_k(top_papers, k=ABSTRACT_FILTER_TOP_K)
        if not paper_data_list:
            self.logger.log_step(3, "Download top papers", downloaded=0)
            return _empty_results("Failed to download papers.")
        self.logger.log_step(3, "Download top papers", downloaded=len(paper_data_list))

        # 4. Clear and process papers with ALL chunking strategies
        self.logger.log_step(4, "Clear and process papers (all strategies)")
        clear_all_data(self.storage, vector_size=self.embedding_dim)

        for paper_data in paper_data_list:
            for strategy in ALL_STRATEGIES:
                process_and_store_paper(
                    self.storage,
                    self.embedder,
                    paper_data,
                    strategy,
                    skip_abstract=True,
                )
        self.logger.log_step(4, "Clear and process papers", processed=len(paper_data_list) * len(ALL_STRATEGIES))

        # 5 & 6. For each strategy: retrieve chunks and generate answer
        results: List[Dict[str, Any]] = []
        for strategy in ALL_STRATEGIES:
            chunks_with_meta = retrieve_chunks_with_metadata(
                self.storage,
                self.embedder,
                query,
                strategy,
                top_k=RETRIEVAL_CHUNK_TOP_K,
            )

            if not chunks_with_meta:
                results.append({
                    "strategy": strategy.value,
                    "strategy_label": format_strategy_label(strategy),
                    "answer": "No relevant chunks found.",
                    "sources": [],
                })
                continue

            max_chunks = min(10, len(chunks_with_meta))
            chunks_for_llm = chunks_with_meta[:max_chunks]

            try:
                answer = self.llm_client.generate_rag_response(
                    query, chunks_for_llm, max_chunks=max_chunks
                )
            except Exception as e:
                answer = f"Error generating response: {e}"

            sources = get_sources_from_chunks(chunks_for_llm)
            strategy_result: Dict[str, Any] = {
                "strategy": strategy.value,
                "strategy_label": format_strategy_label(strategy),
                "answer": answer,
                "sources": sources,
            }

            if include_debug_context:
                strategy_result["debug_context"] = [
                    {
                        "paper_id": payload.get("paper_id", ""),
                        "title": payload.get("title", ""),
                        "section": payload.get("section") or "",
                        "position": payload.get("position", 0),
                        "chunk_text": chunk_text,
                    }
                    for chunk_text, payload in chunks_for_llm
                ]

            results.append(strategy_result)

        self.logger.log_step(7, "Multi-strategy results", count=len(results))
        return {"results": results}
