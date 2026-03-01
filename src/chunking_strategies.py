"""
RAG Chunking Strategies for ArXiv Papers.
Strategy implementations only; Chunk and Chunker live in chunker.py.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Optional, Dict, Any, Callable

from src.chunk import Chunk
from src.document_processor import DocumentProcessor


class ChunkingStrategy(str, Enum):
    """Available chunking strategies."""

    STRUCTURE_AWARE_OVERLAP = "STRUCTURE_AWARE_OVERLAP"
    SEMANTIC_PARAGRAPH_GROUPING = "SEMANTIC_PARAGRAPH_GROUPING"
    FIXED_WINDOW_OVERLAP = "FIXED_WINDOW_OVERLAP"
    SECTION_LEVEL_CHUNKING = "SECTION_LEVEL_CHUNKING"


def _make_chunk(
    paper_id: str,
    title: str,
    strategy_name: str,
    text: str,
    position: int,
    section: Optional[str] = None,
    subsection: Optional[str] = None,
) -> Chunk:
    """Helper to create a Chunk with consistent metadata."""
    chunk_id = f"{paper_id}_{strategy_name}_{position}"
    return Chunk(
        id=chunk_id,
        text=text,
        metadata={
            "paper_id": paper_id,
            "title": title,
            "section": section,
            "subsection": subsection,
            "position": position,
            "strategy": strategy_name,
        },
    )


class BaseChunkingStrategy(ABC):
    """Base class for chunking strategies."""

    @abstractmethod
    def chunk(
        self,
        text: str,
        paper_id: str,
        title: str,
        processor: DocumentProcessor,
        embed_fn: Optional[Callable[[List[str]], Any]] = None,
        **kwargs: Any,
    ) -> List[Chunk]:
        """Chunk document text. Returns list of Chunks."""
        pass


class StructureAwareOverlapStrategy(BaseChunkingStrategy):
    """
    Preserve academic structure. Target 500-800 tokens, max 900.
    10-15% overlap. Abstract and Conclusion as standalone chunks.
    References excluded.
    """

    def chunk(
        self,
        text: str,
        paper_id: str,
        title: str,
        processor: DocumentProcessor,
        embed_fn: Optional[Callable[[List[str]], Any]] = None,
        **kwargs: Any,
    ) -> List[Chunk]:
        strategy_name = ChunkingStrategy.STRUCTURE_AWARE_OVERLAP.value
        blocks = processor.parse_hierarchy(text)
        chunks: List[Chunk] = []
        target_tokens = 650
        max_tokens = 900
        overlap_ratio = 0.12
        position = 0

        skip_abstract = kwargs.get("skip_abstract", False)

        for block in blocks:
            if not block.text.strip():
                continue
            if skip_abstract and block.block_type == "abstract":
                continue

            section = block.section
            subsection = block.subsection

            if block.block_type in ("abstract", "conclusion"):
                chunks.append(
                    _make_chunk(
                        paper_id, title, strategy_name,
                        block.text.strip(), position,
                        section=section, subsection=subsection,
                    )
                )
                position += 1
                continue

            paras = processor.split_into_paragraphs(block.text)
            if not paras:
                continue

            current_chunk_paras: List[str] = []
            current_tokens = 0
            overlap_tokens = int(target_tokens * overlap_ratio)

            for para in paras:
                para_tokens = processor.count_tokens(para)
                if para_tokens > max_tokens:
                    if current_chunk_paras:
                        chunk_text = "\n\n".join(current_chunk_paras)
                        chunks.append(
                            _make_chunk(
                                paper_id, title, strategy_name, chunk_text, position,
                                section=section, subsection=subsection,
                            )
                        )
                        position += 1
                        current_chunk_paras = []
                        current_tokens = 0
                    chunks.append(
                        _make_chunk(
                            paper_id, title, strategy_name, para, position,
                            section=section, subsection=subsection,
                        )
                    )
                    position += 1
                elif current_tokens + para_tokens > max_tokens and current_chunk_paras:
                    chunk_text = "\n\n".join(current_chunk_paras)
                    chunks.append(
                        _make_chunk(
                            paper_id, title, strategy_name, chunk_text, position,
                            section=section, subsection=subsection,
                        )
                    )
                    position += 1
                    overlap_paras = []
                    overlap_t = 0
                    for p in reversed(current_chunk_paras):
                        if overlap_t + processor.count_tokens(p) <= overlap_tokens:
                            overlap_paras.insert(0, p)
                            overlap_t += processor.count_tokens(p)
                        else:
                            break
                    current_chunk_paras = overlap_paras + [para]
                    current_tokens = sum(processor.count_tokens(p) for p in current_chunk_paras)
                else:
                    current_chunk_paras.append(para)
                    current_tokens += para_tokens

            if current_chunk_paras:
                chunk_text = "\n\n".join(current_chunk_paras)
                chunks.append(
                    _make_chunk(
                        paper_id, title, strategy_name, chunk_text, position,
                        section=section, subsection=subsection,
                    )
                )
                position += 1

        return chunks


class SemanticParagraphGroupingStrategy(BaseChunkingStrategy):
    """
    Group paragraphs by embedding similarity.
    Min 300, max 900 tokens. Requires embed_fn.
    Falls back to structure-aware if no embed_fn.
    """

    def __init__(self, similarity_threshold: float = 0.75):
        self.similarity_threshold = similarity_threshold

    def chunk(
        self,
        text: str,
        paper_id: str,
        title: str,
        processor: DocumentProcessor,
        embed_fn: Optional[Callable[[List[str]], Any]] = None,
        **kwargs: Any,
    ) -> List[Chunk]:
        if embed_fn is None:
            fallback = StructureAwareOverlapStrategy()
            return fallback.chunk(text, paper_id, title, processor, embed_fn, **kwargs)

        strategy_name = ChunkingStrategy.SEMANTIC_PARAGRAPH_GROUPING.value
        text_to_chunk = processor.get_text_without_abstract(text) if kwargs.get("skip_abstract") else text
        paras = processor.split_into_paragraphs(text_to_chunk)
        if not paras:
            return []

        ref_keywords = ["references", "bibliography"]
        clean_paras = []
        for p in paras:
            if any(kw in p.lower()[:100] for kw in ref_keywords) and len(paras) > 5:
                break
            clean_paras.append(p)
        if not clean_paras:
            clean_paras = paras
        if not clean_paras:
            return []

        from sklearn.metrics.pairwise import cosine_similarity

        embeddings = embed_fn(clean_paras)
        sim_matrix = cosine_similarity(embeddings)
        max_tokens = 900

        chunks: List[Chunk] = []
        current_paras: List[str] = []
        current_tokens = 0
        position = 0

        for i, para in enumerate(clean_paras):
            para_tokens = processor.count_tokens(para)
            prev_sim = 1.0 if i == 0 else float(sim_matrix[i, i - 1])

            if prev_sim < self.similarity_threshold and current_paras:
                chunk_text = "\n\n".join(current_paras)
                chunks.append(
                    _make_chunk(paper_id, title, strategy_name, chunk_text, position)
                )
                position += 1
                current_paras = []
                current_tokens = 0

            if para_tokens > max_tokens:
                if current_paras:
                    chunk_text = "\n\n".join(current_paras)
                    chunks.append(
                        _make_chunk(paper_id, title, strategy_name, chunk_text, position)
                    )
                    position += 1
                    current_paras = []
                    current_tokens = 0
                chunks.append(
                    _make_chunk(paper_id, title, strategy_name, para, position)
                )
                position += 1
            elif current_tokens + para_tokens > max_tokens and current_paras:
                chunk_text = "\n\n".join(current_paras)
                chunks.append(
                    _make_chunk(paper_id, title, strategy_name, chunk_text, position)
                )
                position += 1
                current_paras = [para]
                current_tokens = para_tokens
            else:
                current_paras.append(para)
                current_tokens += para_tokens

        if current_paras:
            chunk_text = "\n\n".join(current_paras)
            chunks.append(
                _make_chunk(paper_id, title, strategy_name, chunk_text, position)
            )

        return chunks


class FixedWindowOverlapStrategy(BaseChunkingStrategy):
    """Sliding window: 700 tokens, 150 overlap. References removed."""

    def __init__(self, window_size: int = 700, overlap: int = 150):
        self.window_size = window_size
        self.overlap = overlap

    def chunk(
        self,
        text: str,
        paper_id: str,
        title: str,
        processor: DocumentProcessor,
        embed_fn: Optional[Callable[[List[str]], Any]] = None,
        **kwargs: Any,
    ) -> List[Chunk]:
        strategy_name = ChunkingStrategy.FIXED_WINDOW_OVERLAP.value
        text = processor.remove_references(text)
        if kwargs.get("skip_abstract"):
            text = processor.get_text_without_abstract(text)

        tokens, tok = processor.tokenize(text)
        window_size = self.window_size
        overlap = self.overlap
        if tok is None:
            window_size = window_size * 4
            overlap = overlap * 4

        step = window_size - overlap
        chunks: List[Chunk] = []
        position = 0
        start = 0

        while start < len(tokens):
            end = min(start + window_size, len(tokens))
            chunk_tokens = tokens[start:end]
            if tok:
                chunk_text = tok.decode(chunk_tokens).strip()
            else:
                chunk_text = "".join(chunk_tokens).strip()

            if chunk_text:
                chunks.append(
                    _make_chunk(paper_id, title, strategy_name, chunk_text, position)
                )
                position += 1

            start += step
            if step <= 0:
                break

        return chunks


class SectionLevelChunkingStrategy(BaseChunkingStrategy):
    """One chunk per subsection. If >1500 tokens, split at midpoint."""

    def __init__(self, max_tokens: int = 1500):
        self.max_tokens = max_tokens

    def chunk(
        self,
        text: str,
        paper_id: str,
        title: str,
        processor: DocumentProcessor,
        embed_fn: Optional[Callable[[List[str]], Any]] = None,
        **kwargs: Any,
    ) -> List[Chunk]:
        strategy_name = ChunkingStrategy.SECTION_LEVEL_CHUNKING.value
        blocks = processor.parse_hierarchy(text)
        chunks: List[Chunk] = []
        position = 0
        skip_abstract = kwargs.get("skip_abstract", False)

        for block in blocks:
            if not block.text.strip():
                continue
            if skip_abstract and block.block_type == "abstract":
                continue

            section = block.section
            subsection = block.subsection
            tokens = processor.count_tokens(block.text)

            if tokens <= self.max_tokens:
                chunks.append(
                    _make_chunk(
                        paper_id, title, strategy_name,
                        block.text.strip(), position,
                        section=section, subsection=subsection,
                    )
                )
                position += 1
            else:
                paras = processor.split_into_paragraphs(block.text)
                mid = len(paras) // 2
                first_half = "\n\n".join(paras[:mid])
                second_half = "\n\n".join(paras[mid:])
                for half in (first_half, second_half):
                    if half.strip():
                        chunks.append(
                            _make_chunk(
                                paper_id, title, strategy_name, half.strip(), position,
                                section=section, subsection=subsection,
                            )
                        )
                        position += 1

        return chunks
