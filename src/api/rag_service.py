"""
RAG service: orchestrates the full pipeline and extracts cited sources.
"""
from __future__ import annotations

from typing import List, Dict, Any, Tuple, Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.logger import RAGLogger
from src.api.rag_storage import RAGStorage, process_and_store_paper, clear_all_data
from src.embedder import PaperEmbedder
from src.chunker import ChunkingStrategy
from src.llm_client import DeepSeekClient
from src.arxiv_retriever import ArxivRetriever


def _mmr_select(
    vectors: np.ndarray,
    query_emb: np.ndarray,
    payloads: List[Dict[str, Any]],
    top_k: int,
    lambda_param: float = 0.7,
) -> List[int]:
    """
    Select top_k indices using Maximal Marginal Relevance.
    MMR = λ * sim(q, d) - (1-λ) * max(sim(d, s)) for s in selected.
    Higher lambda favors relevance; lower favors diversity.
    """
    n = len(vectors)
    if n <= top_k:
        return list(range(n))

    sim_to_query = cosine_similarity(query_emb, vectors)[0]
    sim_matrix = cosine_similarity(vectors)

    selected: List[int] = []
    remaining = set(range(n))

    for _ in range(top_k):
        if not remaining:
            break

        best_score = float("-inf")
        best_idx = -1

        for idx in remaining:
            relevance = sim_to_query[idx]
            if not selected:
                mmr_score = relevance
            else:
                max_sim_to_selected = max(sim_matrix[idx, s] for s in selected)
                mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim_to_selected

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        if best_idx >= 0:
            selected.append(best_idx)
            remaining.discard(best_idx)

    return selected


def _expand_chunk_neighbors(
    selected: List[Tuple[str, Dict[str, Any]]],
    all_payloads: List[Dict[str, Any]],
) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Expand each selected chunk with ±1 neighbors from the same paper.
    Preserves selection order; neighbors are inserted before/after their seed chunk.
    """
    if not selected or not all_payloads:
        return selected

    lookup: Dict[Tuple[str, int], Tuple[str, Dict[str, Any]]] = {}
    for i, p in enumerate(all_payloads):
        if not p:
            continue
        paper_id = p.get("paper_id", "")
        position = p.get("position", 0)
        chunk_text = p.get("chunk_text", "")
        lookup[(paper_id, position)] = (chunk_text, p)

    expanded: List[Tuple[str, Dict[str, Any]]] = []
    seen: set[Tuple[str, int]] = set()

    for chunk_text, payload in selected:
        paper_id = payload.get("paper_id", "")
        pos = payload.get("position", 0)

        for neighbor_pos in (pos - 1, pos, pos + 1):
            key = (paper_id, neighbor_pos)
            if key in seen:
                continue
            if key in lookup:
                seen.add(key)
                expanded.append(lookup[key])

    return expanded


def retrieve_chunks_with_metadata(
    storage: RAGStorage,
    embedder: PaperEmbedder,
    query: str,
    strategy: ChunkingStrategy,
    top_k: int = 8,
    use_mmr: bool = True,
    mmr_lambda: float = 0.7,
    expand_neighbors: bool = True,
) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Retrieve top-k chunks with metadata (chunk_text, payload).
    Returns list of (chunk_text, payload) where payload has paper_id, title, section.

    Args:
        use_mmr: If True, use MMR for diverse retrieval.
        mmr_lambda: MMR balance (0.7 = favor relevance, 0.3 = favor diversity).
        expand_neighbors: If True, include ±1 neighboring chunks from same paper.
    """
    try:
        vectors, payloads = storage.fetch_embeddings(strategy.value)

        if not vectors or not payloads:
            return []

        query_emb = embedder.embed_texts([query])

        if isinstance(vectors[0], list):
            vectors = np.array(vectors)

        if use_mmr:
            top_indices = _mmr_select(
                vectors, query_emb, payloads, top_k=top_k, lambda_param=mmr_lambda
            )
        else:
            similarities = cosine_similarity(query_emb, vectors)[0]
            top_indices = np.argsort(similarities)[::-1][:top_k].tolist()

        result: List[Tuple[str, Dict[str, Any]]] = []
        for i in top_indices:
            payload = payloads[i] or {}
            chunk_text = payload.get("chunk_text", "")
            result.append((chunk_text, payload))

        if expand_neighbors:
            result = _expand_chunk_neighbors(result, payloads)

        return result

    except Exception as e:
        print(f"Error retrieving chunks: {e}")
        import traceback

        traceback.print_exc()
        return []


def get_sources_from_chunks(
    chunks_with_meta: List[Tuple[str, Dict[str, Any]]],
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


class RAGService:
    """Orchestrates the full RAG pipeline for a single query."""

    def __init__(
        self,
        storage: RAGStorage,
        embedder: PaperEmbedder,
        retriever: ArxivRetriever,
        llm_client: DeepSeekClient,
        strategy: ChunkingStrategy = ChunkingStrategy.STRUCTURE_AWARE_OVERLAP,
        embedding_dim: int = 768,
        logger: Optional[RAGLogger] = None,
    ):
        self.storage = storage
        self.embedder = embedder
        self.retriever = retriever
        self.llm_client = llm_client
        self.strategy = strategy
        self.embedding_dim = embedding_dim
        self.logger = logger or RAGLogger()

    def query(self, query: str, topics: str) -> Dict[str, Any]:
        """
        Run full RAG pipeline and return {answer, sources}.
        topics: comma-separated terms for arXiv search.
        query: used for embedding similarity and LLM context.
        """
        self.logger.log_step(0, "Starting RAG pipeline", query=query[:80], topics=topics[:80])

        # 1. Search arXiv using topics (comma-separated; multi-word terms quoted for phrase search)
        terms = [t.strip() for t in topics.split(",") if t.strip()]
        search_terms = " ".join(
            f'"{term}"' if " " in term else term for term in terms
        )
        self.logger.log_step(1, "Search arXiv", max_results=20)
        papers = self.retriever.search(search_terms, max_results=20)
        if not papers:
            self.logger.log_step(1, "Search arXiv", found=0)
            return {"answer": "No papers found for this query.", "sources": []}
        self.logger.log_step(1, "Search arXiv", found=len(papers))

        # 2. Filter by abstract cosine similarity to query (top 5)
        self.logger.log_step(2, "Filter by abstract similarity", top_k=5)
        top_papers = self.retriever.filter_by_abstract_similarity(
            query, papers, self.embedder, top_k=5
        )
        if not top_papers:
            self.logger.log_step(2, "Filter by abstract similarity", kept=0)
            return {"answer": "No relevant papers after filtering.", "sources": []}
        self.logger.log_step(2, "Filter by abstract similarity", kept=len(top_papers))

        # 3. Download top 5
        self.logger.log_step(3, "Download top papers", k=5)
        paper_data_list = self.retriever.download_top_k(top_papers, k=5)
        if not paper_data_list:
            self.logger.log_step(3, "Download top papers", downloaded=0)
            return {"answer": "Failed to download papers.", "sources": []}
        self.logger.log_step(3, "Download top papers", downloaded=len(paper_data_list))

        # 4. Clear and process
        self.logger.log_step(4, "Clear and process papers")
        clear_all_data(self.storage, vector_size=self.embedding_dim)

        for paper_data in paper_data_list:
            process_and_store_paper(
                self.storage,
                self.embedder,
                paper_data,
                self.strategy,
                skip_abstract=True,
            )
        self.logger.log_step(4, "Clear and process papers", processed=len(paper_data_list))

        # 5. Retrieve chunks with metadata (using query for embedding similarity)
        self.logger.log_step(5, "Retrieve chunks with metadata", top_k=8)
        chunks_with_meta = retrieve_chunks_with_metadata(
            self.storage,
            self.embedder,
            query,
            self.strategy,
            top_k=8,
        )

        if not chunks_with_meta:
            self.logger.log_step(5, "Retrieve chunks with metadata", chunks=0)
            return {"answer": "No relevant chunks found.", "sources": []}
        self.logger.log_step(5, "Retrieve chunks with metadata", chunks=len(chunks_with_meta))

        max_chunks = min(10, len(chunks_with_meta))
        chunks_for_llm = chunks_with_meta[:max_chunks]

        # 6. Generate RAG response (context includes paper source per chunk)
        self.logger.log_step(6, "Generate RAG response", max_chunks=max_chunks)
        try:
            answer = self.llm_client.generate_rag_response(
                query, chunks_for_llm, max_chunks=max_chunks
            )
        except Exception as e:
            self.logger.log_step(6, "Generate RAG response", error=str(e))
            return {
                "answer": f"Error generating response: {e}",
                "sources": [],
            }

        # 7. Sources = unique papers from chunks passed to LLM
        sources = get_sources_from_chunks(chunks_for_llm)
        self.logger.log_step(7, "Sources from chunks", count=len(sources))

        return {"answer": answer, "sources": sources}
