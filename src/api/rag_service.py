"""
RAG service: orchestrates the full pipeline and extracts cited sources.
"""
from __future__ import annotations

import re
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.logger import RAGLogger
from interactive_rag import (
    InteractiveRAGStorage,
    process_and_store_paper,
    clear_all_data,
)
from src.embedder import PaperEmbedder
from src.chunker import ChunkingStrategy
from src.llm_client import DeepSeekClient
from src.arxiv_retriever import ArxivRetriever
from src.query_expander import QueryExpander


def retrieve_chunks_with_metadata(
    storage: InteractiveRAGStorage,
    embedder: PaperEmbedder,
    query: str,
    strategy: ChunkingStrategy,
    top_k: int = 8,
) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Retrieve top-k chunks with metadata (chunk_text, payload).
    Returns list of (chunk_text, payload) where payload has paper_id, title, section.
    """
    try:
        vectors, payloads = storage.fetch_embeddings(strategy.value)

        if not vectors or not payloads:
            return []

        query_emb = embedder.embed_texts([query])

        if isinstance(vectors[0], list):
            vectors = np.array(vectors)

        similarities = cosine_similarity(query_emb, vectors)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]

        result: List[Tuple[str, Dict[str, Any]]] = []
        for i in top_indices:
            payload = payloads[i] or {}
            chunk_text = payload.get("chunk_text", "")
            result.append((chunk_text, payload))

        return result

    except Exception as e:
        print(f"Error retrieving chunks: {e}")
        import traceback

        traceback.print_exc()
        return []


def extract_cited_sources(
    response: str,
    chunks_with_meta: List[Tuple[str, Dict[str, Any]]],
) -> List[Dict[str, str]]:
    """
    Parse [N] citations from LLM response and map to chunk metadata.
    Deduplicate by paper_id. Return list of {paper_id, title, section, arxiv_url}.
    """
    if not chunks_with_meta:
        return []

    max_index = len(chunks_with_meta)
    cited_indices: set[int] = set()
    for m in re.finditer(r"\[(\d+)\]", response):
        n = int(m.group(1))
        if 1 <= n <= max_index:
            cited_indices.add(n)

    seen_paper_ids: set[str] = set()
    sources: List[Dict[str, str]] = []

    for idx in sorted(cited_indices):
        _, payload = chunks_with_meta[idx - 1]
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


class RAGService:
    """Orchestrates the full RAG pipeline for a single query."""

    def __init__(
        self,
        storage: InteractiveRAGStorage,
        embedder: PaperEmbedder,
        retriever: ArxivRetriever,
        query_expander: QueryExpander,
        llm_client: DeepSeekClient,
        strategy: ChunkingStrategy = ChunkingStrategy.STRUCTURE_AWARE_OVERLAP,
        embedding_dim: int = 768,
        logger: Optional[RAGLogger] = None,
    ):
        self.storage = storage
        self.embedder = embedder
        self.retriever = retriever
        self.query_expander = query_expander
        self.llm_client = llm_client
        self.strategy = strategy
        self.embedding_dim = embedding_dim
        self.logger = logger or RAGLogger()

    def query(self, query: str) -> Dict[str, Any]:
        """
        Run full RAG pipeline and return {answer, sources}.
        """
        self.logger.log_step(0, "Starting RAG pipeline", query=query[:80])

        # 1. Expand query
        expanded = self.query_expander.expand(query)
        self.logger.log_step(1, "Expand query", query=query[:80], expanded=expanded[:80])

        # 2. Search arXiv (max 25 results)
        self.logger.log_step(2, "Search arXiv", max_results=20)
        papers = self.retriever.search(expanded, max_results=20)
        if not papers:
            self.logger.log_step(2, "Search arXiv", found=0)
            return {"answer": "No papers found for this query.", "sources": []}
        self.logger.log_step(2, "Search arXiv", found=len(papers))

        # 3. Filter by abstract cosine similarity to query (top 5)
        self.logger.log_step(3, "Filter by abstract similarity", top_k=5)
        top_papers = self.retriever.filter_by_abstract_similarity(
            query, papers, self.embedder, top_k=5
        )
        if not top_papers:
            self.logger.log_step(3, "Filter by abstract similarity", kept=0)
            return {"answer": "No relevant papers after filtering.", "sources": []}
        self.logger.log_step(3, "Filter by abstract similarity", kept=len(top_papers))

        # 4. Download top 5
        self.logger.log_step(4, "Download top papers", k=5)
        paper_data_list = self.retriever.download_top_k(top_papers, k=5)
        if not paper_data_list:
            self.logger.log_step(4, "Download top papers", downloaded=0)
            return {"answer": "Failed to download papers.", "sources": []}
        self.logger.log_step(4, "Download top papers", downloaded=len(paper_data_list))

        # 5. Clear and process
        self.logger.log_step(5, "Clear and process papers")
        clear_all_data(self.storage, vector_size=self.embedding_dim)

        for paper_data in paper_data_list:
            process_and_store_paper(
                self.storage,
                self.embedder,
                paper_data,
                self.strategy,
                skip_abstract=True,
            )
        self.logger.log_step(5, "Clear and process papers", processed=len(paper_data_list))

        # 6. Retrieve chunks with metadata
        self.logger.log_step(6, "Retrieve chunks with metadata", top_k=8)
        chunks_with_meta = retrieve_chunks_with_metadata(
            self.storage,
            self.embedder,
            query,
            self.strategy,
            top_k=8,
        )

        if not chunks_with_meta:
            self.logger.log_step(6, "Retrieve chunks with metadata", chunks=0)
            return {"answer": "No relevant chunks found.", "sources": []}
        self.logger.log_step(6, "Retrieve chunks with metadata", chunks=len(chunks_with_meta))

        chunk_texts = [t for t, _ in chunks_with_meta]

        # 7. Generate RAG response (with citation instruction)
        self.logger.log_step(7, "Generate RAG response", max_chunks=min(6, len(chunk_texts)))
        try:
            answer = self.llm_client.generate_rag_response(
                query, chunk_texts, max_chunks=min(6, len(chunk_texts))
            )
        except Exception as e:
            self.logger.log_step(7, "Generate RAG response", error=str(e))
            return {
                "answer": f"Error generating response: {e}",
                "sources": [],
            }

        # 8. Extract cited sources
        sources = extract_cited_sources(answer, chunks_with_meta)
        self.logger.log_step(8, "Extract cited sources", count=len(sources))

        return {"answer": answer, "sources": sources}
