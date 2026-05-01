"""
Shared RAG pipeline helpers used by API and evaluation.
"""
from __future__ import annotations

from src.chunker import ChunkingStrategy


ALL_STRATEGIES = [
    ChunkingStrategy.STRUCTURE_AWARE_OVERLAP,
    ChunkingStrategy.SEMANTIC_PARAGRAPH_GROUPING,
    ChunkingStrategy.FIXED_WINDOW_OVERLAP,
    ChunkingStrategy.SECTION_LEVEL_CHUNKING,
]


def format_strategy_label(strategy: ChunkingStrategy) -> str:
    """Human-readable label for a chunking strategy."""
    labels = {
        ChunkingStrategy.STRUCTURE_AWARE_OVERLAP: "Structure-Aware Overlap",
        ChunkingStrategy.SEMANTIC_PARAGRAPH_GROUPING: "Semantic Paragraph Grouping",
        ChunkingStrategy.FIXED_WINDOW_OVERLAP: "Fixed Window Overlap",
        ChunkingStrategy.SECTION_LEVEL_CHUNKING: "Section-Level Chunking",
    }
    return labels.get(strategy, strategy.value)


def topics_to_search_query(topics: str) -> str:
    """Build arXiv search query from comma-separated topics."""
    terms = [t.strip() for t in topics.split(",") if t.strip()]
    return " ".join(f'"{term}"' if " " in term else term for term in terms)
