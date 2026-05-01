"""
Central RAG / arXiv search constants — single source of truth for the project.
"""
from __future__ import annotations

# arXiv keyword search: max papers returned before abstract filtering
ARXIV_SEARCH_MAX_RESULTS: int = 20

# After embedding similarity against abstracts, how many papers to keep and download
ABSTRACT_FILTER_TOP_K: int = 6

# Chunk-level retrieval: how many chunks to pass toward generation / eval (after optional rerank)
RETRIEVAL_CHUNK_TOP_K: int = 8

# Passage-level relevance (gold_passages vs chunk text) via cosine similarity
PASSAGE_RELEVANCE_THRESHOLD: float = 0.8

# Hybrid retrieval: weight on dense cosine vs BM25 (0..1)
HYBRID_DENSE_WEIGHT: float = 0.5

# Candidates to score with cross-encoder before truncating to top_k
RERANK_POOL_MULTIPLIER: int = 4
RERANK_POOL_MIN: int = 16
RERANK_POOL_MAX: int = 64

# Default cross-encoder (lightweight; override via env or CLI if needed)
DEFAULT_CROSS_ENCODER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# MinIO bucket for frozen evaluation corpora (PDFs + manifest JSON)
EVAL_FROZEN_CORPUS_BUCKET: str = "eval-frozen-corpus"