"""
RAG storage: bucket/collections for web RAG pipeline.
"""
from __future__ import annotations

from typing import Dict

from src.storage_manager import StorageManager
from src.embedder import PaperEmbedder
from src.chunker import Chunker, ChunkingStrategy
from src.chunk import Chunk

RAG_BUCKET = "interactive-rag-pdfs"
RAG_COLLECTION = "interactive_rag_chunks"
RAG_PAPERS_COLLECTION = "interactive_rag_papers"


class RAGStorage(StorageManager):
    """Storage manager for web RAG pipeline."""

    def __init__(self):
        super().__init__()
        self.bucket_name = RAG_BUCKET
        self.qdrant_collection = RAG_COLLECTION
        self.papers_collection = RAG_PAPERS_COLLECTION


def process_and_store_paper(
    storage: RAGStorage,
    embedder: PaperEmbedder,
    paper_data: Dict,
    strategy: ChunkingStrategy,
    skip_abstract: bool = True,
) -> bool:
    """
    Process a paper: chunk (excluding abstract), embed body chunks and abstract separately, store.
    When skip_abstract=True, abstract is embedded separately with section=Abstract.
    """
    try:
        metadata = paper_data["metadata"]
        pdf_bytes = paper_data["pdf_bytes"]
        paper_id = metadata["id"]
        title = metadata.get("title", "")
        abstract = metadata.get("abstract", "")

        storage.save_paper_metadata(metadata)
        storage.save_paper_pdf(paper_id, pdf_bytes)

        chunker = Chunker(strategy)
        text = chunker.extract_text_from_pdf(pdf_bytes)
        if not text:
            return False

        embed_fn = None
        if strategy == ChunkingStrategy.SEMANTIC_PARAGRAPH_GROUPING:
            embed_fn = lambda texts: embedder.embed_texts(texts)

        chunks = chunker.chunk(text, paper_id, title, embed_fn=embed_fn, skip_abstract=skip_abstract)

        if not chunks:
            return False

        valid_chunks = [c for c in chunks if len(c.text) >= 50]
        if not valid_chunks:
            return False

        strategy_name = strategy.value

        chunk_texts = [c.text for c in valid_chunks]
        embeddings = embedder.embed_texts(chunk_texts)
        payloads = [c.to_dict() for c in valid_chunks]
        vectors = list(embeddings)
        storage.save_embeddings(vectors, payloads)

        if abstract and skip_abstract:
            abstract_chunk = Chunk(
                id=f"{paper_id}_{strategy_name}_abstract",
                text=abstract,
                metadata={
                    "paper_id": paper_id,
                    "title": title,
                    "section": "Abstract",
                    "subsection": None,
                    "position": -1,
                    "strategy": strategy_name,
                },
            )
            abstract_emb = embedder.embed_texts([abstract])
            storage.save_embeddings(
                list(abstract_emb),
                [abstract_chunk.to_dict()],
            )

        return True

    except Exception as e:
        print(f"  Error processing paper: {e}")
        import traceback
        traceback.print_exc()
        return False


def clear_all_data(storage: RAGStorage, vector_size: int = 768):
    """Clear all data from storage."""
    try:
        storage.reset_db()
        storage.reset_bucket()

        try:
            storage.qdrant.delete_collection(storage.qdrant_collection)
        except Exception:
            pass
        storage.init_qdrant(vector_size=vector_size)
    except Exception as e:
        print(f"Error clearing data: {e}")
