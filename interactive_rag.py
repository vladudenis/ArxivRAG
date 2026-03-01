"""
Interactive RAG System - Query-First Workflow
Enter your query directly; the system expands it, searches arXiv, filters by abstract similarity,
downloads top papers, and answers using RAG.

Usage:
    1. Start vLLM server for query expansion (e.g., TinyLlama on localhost:8001)
    2. Run: python interactive_rag.py
    3. Phase 0: Select chunking strategy
    4. Phase 1: Enter your query (or 'done' to end)
    5. Per query: expand -> search -> filter -> download -> embed -> retrieve -> answer
    6. All data is cleared when session ends

Configuration:
    - QUERY_EXPANSION_BASE_URL, QUERY_EXPANSION_MODEL for vLLM (fallback to original query if unavailable)
    - LLM_BASE_URL, LLM_API_KEY for DeepSeek (answer generation)
    - Uses MinIO for PDFs and Qdrant for embeddings and paper metadata
"""
import re
import numpy as np
from datetime import datetime
from typing import List, Dict
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

from src.storage_manager import StorageManager
from src.embedder import PaperEmbedder
from src.chunker import Chunker, ChunkingStrategy
from src.chunk import Chunk
from src.llm_client import DeepSeekClient
from src.arxiv_retriever import ArxivRetriever
from src.query_expander import QueryExpander

# Load environment variables
load_dotenv()

# Configuration for interactive RAG
INTERACTIVE_BUCKET = "interactive-rag-pdfs"
INTERACTIVE_COLLECTION = "interactive_rag_chunks"
INTERACTIVE_PAPERS_COLLECTION = "interactive_rag_papers"


class InteractiveRAGStorage(StorageManager):
    """Storage manager with separate bucket/collections for interactive RAG."""

    def __init__(self):
        super().__init__()
        self.bucket_name = INTERACTIVE_BUCKET
        self.qdrant_collection = INTERACTIVE_COLLECTION
        self.papers_collection = INTERACTIVE_PAPERS_COLLECTION


def process_and_store_paper(
    storage: InteractiveRAGStorage,
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

        print(f"  Processing paper: {title[:60]}...")

        storage.save_paper_metadata(metadata)
        storage.save_paper_pdf(paper_id, pdf_bytes)

        chunker = Chunker(strategy)
        text = chunker.extract_text_from_pdf(pdf_bytes)
        if not text:
            print(f"  Warning: Could not extract text from PDF")
            return False

        embed_fn = None
        if strategy == ChunkingStrategy.SEMANTIC_PARAGRAPH_GROUPING:
            embed_fn = lambda texts: embedder.embed_texts(texts)

        chunks = chunker.chunk(text, paper_id, title, embed_fn=embed_fn, skip_abstract=skip_abstract)

        if not chunks:
            print(f"  Warning: No chunks generated")
            return False

        print(f"  Generated {len(chunks)} body chunks")

        valid_chunks = [c for c in chunks if len(c.text) >= 50]
        if not valid_chunks:
            print(f"  Warning: No valid chunks after filtering")
            return False

        strategy_name = strategy.value

        # Embed and store body chunks
        chunk_texts = [c.text for c in valid_chunks]
        print(f"  Embedding {len(chunk_texts)} body chunks...")
        embeddings = embedder.embed_texts(chunk_texts)
        payloads = [c.to_dict() for c in valid_chunks]
        vectors = list(embeddings)
        storage.save_embeddings(vectors, payloads)

        # Embed and store abstract separately with section=Abstract
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
            print(f"  Embedding abstract...")
            abstract_emb = embedder.embed_texts([abstract])
            storage.save_embeddings(
                list(abstract_emb),
                [abstract_chunk.to_dict()],
            )

        print(f"  ✓ Successfully processed and stored paper")
        return True

    except Exception as e:
        print(f"  Error processing paper: {e}")
        import traceback

        traceback.print_exc()
        return False


def retrieve_chunks(
    storage: InteractiveRAGStorage,
    embedder: PaperEmbedder,
    query: str,
    strategy: ChunkingStrategy,
    top_k: int = 5,
) -> List[str]:
    """Retrieve top-k chunks for a query."""
    try:
        vectors, payloads = storage.fetch_embeddings(strategy.value)

        if not vectors or not payloads:
            return []

        query_emb = embedder.embed_texts([query])

        if isinstance(vectors[0], list):
            vectors = np.array(vectors)

        similarities = cosine_similarity(query_emb, vectors)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]

        return [payloads[i].get("chunk_text", "") for i in top_indices]

    except Exception as e:
        print(f"Error retrieving chunks: {e}")
        import traceback

        traceback.print_exc()
        return []


def save_session_log(paper_ids: List[str], query_answer_pairs: List[Dict[str, str]]):
    """Save session queries and answers to a markdown file."""
    if not query_answer_pairs:
        print("No queries to log.")
        return

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    if len(paper_ids) == 1:
        paper_prefix = paper_ids[0]
    elif len(paper_ids) <= 3:
        paper_prefix = "_".join(paper_ids)
    else:
        paper_prefix = f"{paper_ids[0]}_..._{paper_ids[-1]}"

    paper_prefix = re.sub(r'[<>:"/\\|?*]', "_", paper_prefix)
    filename = f"{paper_prefix}_{timestamp}.md"

    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write("# RAG Session Log\n\n")
            f.write(f"**Session Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("## Papers in Session\n\n")
            for paper_id in paper_ids:
                f.write(f"- `{paper_id}`\n")
            f.write("\n## Query-Answer Pairs\n\n")
            for idx, pair in enumerate(query_answer_pairs, 1):
                f.write(f"### Query {idx}\n\n")
                f.write(f"**Query:** {pair['query']}\n\n")
                f.write(f"**Answer:**\n\n{pair['answer']}\n\n---\n\n")
        print(f"✓ Session log saved to: {filename}")
    except Exception as e:
        print(f"Error saving session log: {e}")
        import traceback

        traceback.print_exc()


def clear_all_data(storage: InteractiveRAGStorage, vector_size: int = 768):
    """Clear all data from storage."""
    print("\nClearing all data...")
    try:
        storage.reset_db()
        storage.reset_bucket()

        try:
            storage.qdrant.delete_collection(storage.qdrant_collection)
            print(f"Collection {storage.qdrant_collection} deleted.")
        except Exception:
            pass
        storage.init_qdrant(vector_size=vector_size)

        print("✓ All data cleared")
    except Exception as e:
        print(f"Error clearing data: {e}")


def select_strategy() -> ChunkingStrategy:
    """Let user select chunking strategy before session."""
    print("\nChunking strategies:")
    for i, s in enumerate(ChunkingStrategy, 1):
        print(f"  {i}. {s.value}")
    print("  (1 = default, recommended)")

    while True:
        try:
            choice = input("\nSelect strategy [1-4] (default 1): ").strip() or "1"
            idx = int(choice)
            if 1 <= idx <= 4:
                return list(ChunkingStrategy)[idx - 1]
        except ValueError:
            pass
        print("  Invalid. Enter 1, 2, 3, or 4.")


def main():
    """Main interactive RAG loop - query-first workflow."""
    print("=" * 80)
    print("Interactive RAG System (Query-First)")
    print("=" * 80)

    strategy = select_strategy()
    print(f"Using strategy: {strategy.value}")
    print(f"LLM: DeepSeek API (8k context window)")
    print("=" * 80)

    print("\nInitializing components...")
    storage = InteractiveRAGStorage()
    storage.init_db()
    storage.init_bucket()

    embedder = PaperEmbedder(model_name="google/embeddinggemma-300m")
    test_embedding = embedder.embed_texts(["test"])
    embedding_dim = test_embedding.shape[1]
    print(f"Embedding dimension: {embedding_dim}")
    storage.init_qdrant(vector_size=embedding_dim)

    retriever = ArxivRetriever()
    query_expander = QueryExpander()

    try:
        llm_client = DeepSeekClient()
        print(f"DeepSeek client initialized (8k context)")

        if not llm_client.check_health():
            print("WARNING: DeepSeek API health check failed. Queries may fail.")
    except Exception as e:
        print(f"WARNING: Could not initialize DeepSeek client: {e}")
        print("Make sure LLM_BASE_URL and LLM_API_KEY are set correctly in .env file.")
        llm_client = None

    # Query-first workflow
    print("\n" + "=" * 80)
    print("Enter your query (or 'done' to end)")
    print("=" * 80)
    print("Each query will: expand -> search arXiv -> filter by abstract -> download top 5 -> answer")
    print("Type 'done' to end the session and clear all data.\n")

    if not llm_client:
        print("ERROR: LLM client not available. Cannot answer queries.")
        clear_all_data(storage, vector_size=embedding_dim)
        return

    query_answer_pairs: List[Dict[str, str]] = []
    all_paper_ids: List[str] = []

    while True:
        query = input("\nQuery (or 'done'): ").strip()

        if query.lower() == "done":
            break
        if not query:
            continue

        print("\n1. Expanding query...")
        expanded = query_expander.expand(query)
        print(f"   Expanded: {expanded}")

        print("\n2. Searching arXiv (max 25 results)...")
        papers = retriever.search(expanded, max_results=25)
        if not papers:
            print("   No papers found. Try a different query.")
            query_answer_pairs.append({"query": query, "answer": "No papers found for this query."})
            continue

        print(f"   Found {len(papers)} papers")

        print("\n3. Filtering by abstract similarity (top 5)...")
        top_papers = retriever.filter_by_abstract_similarity(query, papers, embedder, top_k=5)
        if not top_papers:
            query_answer_pairs.append({"query": query, "answer": "No relevant papers after filtering."})
            continue

        print("\n4. Downloading top 5 papers...")
        paper_data_list = retriever.download_top_k(top_papers, k=5)
        if not paper_data_list:
            print("   Failed to download any papers.")
            query_answer_pairs.append({"query": query, "answer": "Failed to download papers."})
            continue

        print(f"   Downloaded {len(paper_data_list)} papers")

        # Clear previous session data before processing new papers
        clear_all_data(storage, vector_size=embedding_dim)

        print("\n5. Processing and embedding papers...")
        paper_ids = []
        for paper_data in paper_data_list:
            if process_and_store_paper(storage, embedder, paper_data, strategy, skip_abstract=True):
                paper_ids.append(paper_data["metadata"]["id"])

        all_paper_ids.extend(paper_ids)

        if not paper_ids:
            print("   No papers could be processed.")
            query_answer_pairs.append({"query": query, "answer": "Failed to process papers."})
            continue

        print("\n6. Retrieving relevant chunks...")
        chunks = retrieve_chunks(storage, embedder, query, strategy, top_k=8)

        if not chunks:
            print("No relevant chunks found.")
            query_answer_pairs.append({"query": query, "answer": "No relevant chunks found."})
            continue

        print(f"Found {len(chunks)} relevant chunk(s)")
        print("\n7. Generating response...")

        try:
            response = llm_client.generate_rag_response(query, chunks, max_chunks=min(6, len(chunks)))
            print("\n" + "-" * 80)
            print("Answer:")
            print("-" * 80)
            print(response)
            print("-" * 80)

            query_answer_pairs.append({"query": query, "answer": response})
        except Exception as e:
            error_msg = f"Error generating response: {e}"
            print(f"\n{error_msg}")
            import traceback

            traceback.print_exc()
            query_answer_pairs.append({"query": query, "answer": f"[ERROR] {error_msg}"})

    # Session end: save log and cleanup
    print("\n" + "=" * 80)
    print("Ending session...")
    print("=" * 80)

    if query_answer_pairs:
        print("\nSaving session log...")
        save_session_log(all_paper_ids, query_answer_pairs)

    clear_all_data(storage, vector_size=embedding_dim)
    print("\nSession ended. All data cleared.")


if __name__ == "__main__":
    main()
