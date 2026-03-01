"""
Chunker: delegates to a selected chunking strategy.
"""
from __future__ import annotations

from typing import List, Optional, Callable, Any

from src.chunk import Chunk
from src.document_processor import DocumentProcessor
from src.chunking_strategies import (
    ChunkingStrategy,
    StructureAwareOverlapStrategy,
    SemanticParagraphGroupingStrategy,
    FixedWindowOverlapStrategy,
    SectionLevelChunkingStrategy,
)

__all__ = ["Chunk", "Chunker", "ChunkingStrategy", "extract_text_from_pdf", "chunk_document"]


_STRATEGY_MAP = {
    ChunkingStrategy.STRUCTURE_AWARE_OVERLAP: StructureAwareOverlapStrategy(),
    ChunkingStrategy.SEMANTIC_PARAGRAPH_GROUPING: SemanticParagraphGroupingStrategy(),
    ChunkingStrategy.FIXED_WINDOW_OVERLAP: FixedWindowOverlapStrategy(),
    ChunkingStrategy.SECTION_LEVEL_CHUNKING: SectionLevelChunkingStrategy(),
}


class Chunker:
    """
    Chunker that delegates to a selected strategy.
    Uses DocumentProcessor for shared document parsing/processing.
    """

    def __init__(self, strategy: ChunkingStrategy, processor: Optional[DocumentProcessor] = None):
        """
        Args:
            strategy: Chunking strategy to use
            processor: Document processor (created if not provided)
        """
        self.strategy = strategy
        self.processor = processor or DocumentProcessor()
        self._strategy_impl = _STRATEGY_MAP[strategy]

    def chunk(
        self,
        text: str,
        paper_id: str,
        title: str,
        embed_fn: Optional[Callable[[List[str]], Any]] = None,
        **kwargs: Any,
    ) -> List[Chunk]:
        """Chunk document text. Returns list of Chunks."""
        if not text or not text.strip():
            return []
        return self._strategy_impl.chunk(
            text, paper_id, title, self.processor, embed_fn, **kwargs
        )

    def extract_text_from_pdf(self, pdf_bytes: bytes) -> str:
        """Extract text from PDF bytes (delegates to processor)."""
        return self.processor.extract_text_from_pdf(pdf_bytes)


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract text from PDF bytes. Standalone for backward compatibility."""
    return DocumentProcessor().extract_text_from_pdf(pdf_bytes)


def chunk_document(
    text: str,
    strategy: ChunkingStrategy,
    paper_id: str,
    title: str,
    embed_fn: Optional[Callable[[List[str]], Any]] = None,
    **kwargs: Any,
) -> List[Chunk]:
    """
    Factory: chunk document with selected strategy.
    Backward-compatible wrapper around Chunker.
    """
    chunker = Chunker(strategy)
    return chunker.chunk(text, paper_id, title, embed_fn, **kwargs)
