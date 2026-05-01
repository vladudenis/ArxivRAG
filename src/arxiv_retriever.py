"""
ArXiv retriever: search, filter by abstract similarity, and download papers.
"""
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import List, Any

import arxiv
import numpy as np
import requests
from sklearn.metrics.pairwise import cosine_similarity

from src.rag_constants import ABSTRACT_FILTER_TOP_K, ARXIV_SEARCH_MAX_RESULTS


@dataclass
class PaperMeta:
    """Metadata for an arXiv paper."""

    id: str
    title: str
    abstract: str
    authors: List[str]
    published: str
    url: str
    categories: List[str]

    def to_dict(self) -> dict:
        """Convert to dict for storage compatibility."""
        return {
            "id": self.id,
            "title": self.title,
            "abstract": self.abstract,
            "authors": self.authors,
            "published": self.published,
            "url": self.url,
            "categories": self.categories,
        }


class ArxivRetriever:
    """Retrieve and filter arXiv papers by keyword search and abstract similarity."""

    def __init__(
        self,
        request_delay_seconds: float = 3.0,
        max_rate_limit_retries: int = 1,
        initial_backoff_seconds: float = 3.0,
    ):
        self.request_delay_seconds = request_delay_seconds
        self.max_rate_limit_retries = max_rate_limit_retries
        self.initial_backoff_seconds = initial_backoff_seconds

    def search(self, query: str, max_results: int = ARXIV_SEARCH_MAX_RESULTS) -> List[PaperMeta]:
        """
        Search arXiv by keyword query.

        Args:
            query: Search keywords (space-separated).
            max_results: Maximum number of results (default from rag_constants.ARXIV_SEARCH_MAX_RESULTS).

        Returns:
            List of PaperMeta for matching papers.
        """
        client = arxiv.Client(
            page_size=max(1, min(max_results, 50)),
            delay_seconds=self.request_delay_seconds,
            num_retries=3,
        )
        search = arxiv.Search(query=query, max_results=max_results)

        for attempt in range(self.max_rate_limit_retries + 1):
            try:
                papers: List[PaperMeta] = []
                for result in client.results(search):
                    paper_id = result.entry_id.split("/")[-1].split("v")[0]
                    papers.append(
                        PaperMeta(
                            id=paper_id,
                            title=result.title,
                            abstract=result.summary,
                            authors=[a.name for a in result.authors],
                            published=result.published.isoformat(),
                            url=result.pdf_url,
                            categories=result.categories,
                        )
                    )
                return papers
            except arxiv.HTTPError as e:
                is_rate_limit = "429" in str(e)
                is_last_attempt = attempt >= self.max_rate_limit_retries
                if (not is_rate_limit) or is_last_attempt:
                    raise
                backoff = self.initial_backoff_seconds * (2 ** attempt)
                print(
                    f"arXiv rate limit hit for query '{query}'. "
                    f"Retrying in {backoff:.1f}s "
                    f"({attempt + 1}/{self.max_rate_limit_retries})..."
                )
                time.sleep(backoff)

        return []

    def filter_by_abstract_similarity(
        self,
        query: str,
        papers: List[PaperMeta],
        embedder: Any,
        top_k: int = ABSTRACT_FILTER_TOP_K,
    ) -> List[PaperMeta]:
        """
        Filter papers by cosine similarity of abstract to query.

        Args:
            query: User query string.
            papers: List of papers to filter.
            embedder: PaperEmbedder (or similar) with embed_texts(texts) -> np.ndarray.
            top_k: Number of top papers to return (default from rag_constants.ABSTRACT_FILTER_TOP_K).

        Returns:
            Top-k papers by abstract similarity to query.
        """
        if not papers:
            return []

        abstracts = [p.abstract for p in papers]
        query_emb = embedder.embed_texts([query])
        abstract_embs = embedder.embed_texts(abstracts)

        if isinstance(abstract_embs[0], list):
            abstract_embs = np.array(abstract_embs)
        sims = cosine_similarity(query_emb, abstract_embs)[0]

        top_indices = np.argsort(sims)[::-1][:top_k]
        return [papers[i] for i in top_indices]

    def download_top_k(
        self,
        papers: List[PaperMeta],
        k: int = ABSTRACT_FILTER_TOP_K,
    ) -> List[dict]:
        """
        Download PDFs for top k papers.

        Args:
            papers: List of papers (already filtered).
            k: Number of papers to download (default from rag_constants.ABSTRACT_FILTER_TOP_K).

        Returns:
            List of {"metadata": paper_info, "pdf_bytes": content}.
        """
        results: List[dict] = []
        to_download = papers[:k]

        for paper in to_download:
            try:
                response = requests.get(
                    paper.url,
                    headers={"User-Agent": "ArxivRAG-Interactive/1.0"},
                )
                if response.status_code == 200:
                    results.append(
                        {
                            "metadata": paper.to_dict(),
                            "pdf_bytes": response.content,
                        }
                    )
                else:
                    print(f"  Error: Failed to download {paper.id} (Status {response.status_code})")
            except Exception as e:
                print(f"  Error downloading paper {paper.id}: {e}")

        return results
