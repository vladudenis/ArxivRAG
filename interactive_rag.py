"""
Interactive RAG System
Allows users to add papers via URL and query them interactively.
Uses section-based chunking and a 14B parameter LLM.

Usage:
    1. Start vLLM server with a 14B model (e.g., Qwen/Qwen2-14B-Instruct)
    2. Run: python interactive_rag.py
    3. Phase 1: Paste arXiv URLs/IDs, type 'done' when finished
    4. Phase 2: Ask questions, type 'done' to end session
    5. All data is automatically cleared when session ends

Configuration:
    - Set VLLM_MODEL in .env to override default model (requires 4-8k context window)
    - Uses separate MinIO bucket and DynamoDB table (doesn't interfere with evaluation pipeline)
"""
import os
import re
import arxiv
import requests
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

from src.storage_manager import StorageManager
from src.embedder import PaperEmbedder
from src.chunking import Chunker
from src.llm_client import VLLMClient

# Load environment variables
load_dotenv()

# Configuration for interactive RAG
INTERACTIVE_BUCKET = "interactive-rag-pdfs"
INTERACTIVE_TABLE = "InteractiveRAGPapers"
INTERACTIVE_COLLECTION = "interactive_rag_chunks"
# Interactive RAG uses DeepSeek API (configured via LLM_BASE_URL and LLM_API_KEY in .env)
# DeepSeek has 8k context window


class InteractiveRAGStorage(StorageManager):
    """Storage manager with separate bucket/table for interactive RAG."""
    
    def __init__(self):
        # Call parent init first
        super().__init__()
        # Override with interactive-specific names
        self.bucket_name = INTERACTIVE_BUCKET
        self.table_name = INTERACTIVE_TABLE
        self.qdrant_collection = INTERACTIVE_COLLECTION


def extract_arxiv_id(url_or_id: str) -> Optional[str]:
    """
    Extract arXiv ID from URL or return ID if already extracted.
    
    Examples:
        "https://arxiv.org/abs/2301.12345" -> "2301.12345"
        "https://arxiv.org/pdf/2301.12345.pdf" -> "2301.12345"
        "2301.12345" -> "2301.12345"
    """
    # Remove whitespace
    url_or_id = url_or_id.strip()
    
    # If it's already just an ID (format: YYMM.NNNNN)
    if re.match(r'^\d{4}\.\d{5}(v\d+)?$', url_or_id):
        return url_or_id.split('v')[0]  # Remove version suffix if present
    
    # Extract from URL
    patterns = [
        r'arxiv\.org/abs/(\d{4}\.\d{5})',
        r'arxiv\.org/pdf/(\d{4}\.\d{5})',
        r'arxiv\.org/e-print/(\d{4}\.\d{5})',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url_or_id)
        if match:
            return match.group(1)
    
    return None


def download_paper_from_arxiv(arxiv_id: str) -> Optional[Dict]:
    """
    Download a single paper from arXiv by ID.
    
    Args:
        arxiv_id: arXiv paper ID (e.g., "2301.12345")
        
    Returns:
        Paper metadata dict or None if failed
    """
    try:
        client = arxiv.Client()
        search = arxiv.Search(id_list=[arxiv_id])
        
        result = next(client.results(search), None)
        if not result:
            print(f"  Error: Paper {arxiv_id} not found on arXiv")
            return None
        
        paper_id = result.entry_id.split('/')[-1]
        
        paper_info = {
            "id": paper_id,
            "title": result.title,
            "abstract": result.summary,
            "authors": [author.name for author in result.authors],
            "published": result.published.isoformat(),
            "url": result.pdf_url,
            "categories": result.categories
        }
        
        # Download PDF
        print(f"  Downloading PDF...")
        response = requests.get(result.pdf_url, headers={'User-Agent': 'ArxivRAG-Interactive/1.0'})
        if response.status_code == 200:
            pdf_bytes = response.content
            return {
                'metadata': paper_info,
                'pdf_bytes': pdf_bytes
            }
        else:
            print(f"  Error: Failed to download PDF (Status {response.status_code})")
            return None
            
    except Exception as e:
        print(f"  Error downloading paper {arxiv_id}: {e}")
        return None


def process_and_store_paper(
    storage: InteractiveRAGStorage,
    embedder: PaperEmbedder,
    paper_data: Dict
) -> bool:
    """
    Process a paper: chunk, embed, and store.
    
    Args:
        storage: Storage manager instance
        embedder: Embedder instance
        paper_data: Dict with 'metadata' and 'pdf_bytes'
        
    Returns:
        True if successful, False otherwise
    """
    try:
        metadata = paper_data['metadata']
        pdf_bytes = paper_data['pdf_bytes']
        paper_id = metadata['id']
        
        print(f"  Processing paper: {metadata['title'][:60]}...")
        
        # Save metadata and PDF
        storage.save_paper_metadata(metadata)
        storage.save_paper_pdf(paper_id, pdf_bytes)
        
        # Chunk using section-based strategy
        chunker = Chunker(strategy="section", chunk_size=0, chunk_overlap=0)
        text = chunker.extract_text_from_pdf(pdf_bytes)
        
        if not text:
            print(f"  Warning: Could not extract text from PDF")
            return False
        
        chunks = chunker.chunk_paper(paper_id, text)
        
        if not chunks:
            print(f"  Warning: No chunks generated")
            return False
        
        print(f"  Generated {len(chunks)} chunks")
        
        # Filter out very short chunks
        valid_chunks = [c for c in chunks if len(c.chunk_text) >= 50]
        if not valid_chunks:
            print(f"  Warning: No valid chunks after filtering")
            return False
        
        # Embed chunks
        chunk_texts = [c.chunk_text for c in valid_chunks]
        print(f"  Embedding {len(chunk_texts)} chunks...")
        embeddings = embedder.embed_texts(chunk_texts)
        
        # Prepare payloads for Qdrant
        vectors = []
        payloads = []
        
        for chunk, embedding in zip(valid_chunks, embeddings):
            vectors.append(embedding)
            payloads.append({
                'paper_id': chunk.paper_id,
                'section_id': chunk.section_id,
                'chunk_id': chunk.chunk_id,
                'chunk_text': chunk.chunk_text,
                'strategy': 'section'  # Fixed strategy for interactive RAG
            })
        
        # Store embeddings
        print(f"  Storing embeddings...")
        storage.save_embeddings(vectors, payloads)
        
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
    top_k: int = 5
) -> List[str]:
    """
    Retrieve top-k chunks for a query.
    
    Args:
        storage: Storage manager instance
        embedder: Embedder instance
        query: User query text
        top_k: Number of chunks to retrieve
        
    Returns:
        List of chunk texts
    """
    try:
        # Get all embeddings for section strategy
        vectors, payloads = storage.fetch_embeddings('section')
        
        if not vectors or not payloads:
            return []
        
        # Embed query
        query_emb = embedder.embed_texts([query])
        
        # Convert vectors to numpy array if needed
        if isinstance(vectors[0], list):
            vectors = np.array(vectors)
        
        # Calculate similarities
        similarities = cosine_similarity(query_emb, vectors)[0]
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Return chunk texts
        retrieved_chunks = [payloads[i]['chunk_text'] for i in top_indices]
        
        return retrieved_chunks
        
    except Exception as e:
        print(f"Error retrieving chunks: {e}")
        import traceback
        traceback.print_exc()
        return []


def save_session_log(paper_ids: List[str], query_answer_pairs: List[Dict[str, str]]):
    """
    Save session queries and answers to a markdown file.
    
    Args:
        paper_ids: List of paper IDs that were added
        query_answer_pairs: List of dicts with 'query' and 'answer' keys
    """
    if not query_answer_pairs:
        print("No queries to log.")
        return
    
    # Generate filename: paper_ids + timestamp
    # Format: paper_id1_paper_id2_YYYY-MM-DD_HH-MM.md
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    
    # Create paper ID prefix (use first paper ID if multiple, or combine them)
    if len(paper_ids) == 1:
        paper_prefix = paper_ids[0]
    elif len(paper_ids) <= 3:
        # Use all paper IDs if 3 or fewer
        paper_prefix = "_".join(paper_ids)
    else:
        # Use first and last if many papers
        paper_prefix = f"{paper_ids[0]}_..._{paper_ids[-1]}"
    
    # Sanitize filename (remove invalid characters)
    paper_prefix = re.sub(r'[<>:"/\\|?*]', '_', paper_prefix)
    filename = f"{paper_prefix}_{timestamp}.md"
    
    # Write markdown file
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("# RAG Session Log\n\n")
            f.write(f"**Session Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # List papers
            f.write("## Papers in Session\n\n")
            for paper_id in paper_ids:
                f.write(f"- `{paper_id}`\n")
            f.write("\n")
            
            # Write queries and answers
            f.write("## Query-Answer Pairs\n\n")
            for idx, pair in enumerate(query_answer_pairs, 1):
                f.write(f"### Query {idx}\n\n")
                f.write(f"**Query:** {pair['query']}\n\n")
                f.write(f"**Answer:**\n\n")
                f.write(f"{pair['answer']}\n\n")
                f.write("---\n\n")
        
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
        
        # Reset Qdrant collection and recreate with correct vector size
        try:
            storage.qdrant.delete_collection(storage.qdrant_collection)
            print(f"Collection {storage.qdrant_collection} deleted.")
        except Exception:
            pass
        storage.init_qdrant(vector_size=vector_size)
        
        print("✓ All data cleared")
    except Exception as e:
        print(f"Error clearing data: {e}")


def main():
    """Main interactive RAG loop."""
    print("="*80)
    print("Interactive RAG System")
    print("="*80)
    print(f"Using section-based chunking")
    print(f"LLM: DeepSeek API (8k context window)")
    print("="*80)
    
    # Initialize components
    print("\nInitializing components...")
    storage = InteractiveRAGStorage()
    storage.init_db()
    storage.init_bucket()
    
    embedder = PaperEmbedder(model_name="google/embeddinggemma-300m")
    
    # Detect embedding dimension
    test_embedding = embedder.embed_texts(["test"])
    embedding_dim = test_embedding.shape[1]
    print(f"Embedding dimension: {embedding_dim}")
    storage.init_qdrant(vector_size=embedding_dim)
    
    # Initialize LLM client - Use DeepSeek API for interactive RAG
    try:
        # Use DeepSeek API (LLM_BASE_URL and LLM_API_KEY from env)
        llm_base_url = os.getenv("LLM_BASE_URL")
        llm_api_key = os.getenv("LLM_API_KEY")
        
        if not llm_base_url or not llm_api_key:
            raise ValueError("LLM_BASE_URL and LLM_API_KEY must be set in .env file for interactive RAG")
        
        llm_client = VLLMClient(
            base_url=llm_base_url,
            api_key=llm_api_key,
            model="deepseek-chat",  # DeepSeek model name
            temperature=0.7,
            max_tokens=2048,
            context_limit=8192  # DeepSeek has 8k context window
        )
        print(f"DeepSeek LLM client initialized: {llm_base_url} (8k context)")
        
        # Check health
        if not llm_client.check_health():
            print("WARNING: DeepSeek API health check failed.")
            print("You can still add papers, but queries will fail.")
    except Exception as e:
        print(f"WARNING: Could not initialize DeepSeek LLM client: {e}")
        print("Make sure LLM_BASE_URL and LLM_API_KEY are set correctly in .env file.")
        print("You can still add papers, but queries will fail.")
        llm_client = None
    
    # Phase 1: Paper ingestion
    print("\n" + "="*80)
    print("PHASE 1: Add Papers")
    print("="*80)
    print("Paste arXiv URLs or IDs (one per line).")
    print("Type 'done' when finished adding papers.\n")
    
    papers_added = 0
    paper_ids = []  # Track paper IDs for logging
    
    while True:
        user_input = input("arXiv URL/ID (or 'done'): ").strip()
        
        if user_input.lower() == 'done':
            break
        
        if not user_input:
            continue
        
        # Extract arXiv ID
        arxiv_id = extract_arxiv_id(user_input)
        if not arxiv_id:
            print(f"  Error: Could not extract arXiv ID from: {user_input}")
            print("  Please provide a valid arXiv URL or ID (e.g., 2301.12345)")
            continue
        
        print(f"\nProcessing arXiv ID: {arxiv_id}")
        
        # Download paper
        paper_data = download_paper_from_arxiv(arxiv_id)
        if not paper_data:
            print(f"  Failed to download paper {arxiv_id}")
            continue
        
        # Process and store
        if process_and_store_paper(storage, embedder, paper_data):
            papers_added += 1
            paper_ids.append(paper_data['metadata']['id'])
            print(f"  Total papers added: {papers_added}\n")
        else:
            print(f"  Failed to process paper {arxiv_id}\n")
    
    if papers_added == 0:
        print("\nNo papers were added. Exiting.")
        return
    
    print(f"\n✓ Added {papers_added} paper(s)")
    
    # Phase 2: Query interface
    print("\n" + "="*80)
    print("PHASE 2: Query Papers")
    print("="*80)
    print("Ask questions about the papers you added.")
    print("Type 'done' to end the session and clear all data.\n")
    
    if not llm_client:
        print("ERROR: LLM client not available. Cannot answer queries.")
        print("Ending session...")
        clear_all_data(storage, vector_size=embedding_dim)
        return
    
    # Track queries and answers for logging
    query_answer_pairs = []
    
    while True:
        query = input("\nQuery (or 'done'): ").strip()
        
        if query.lower() == 'done':
            break
        
        if not query:
            continue
        
        print("\nRetrieving relevant chunks...")
        # Retrieve more chunks - with 8k context window we can use more chunks with full detail
        chunks = retrieve_chunks(storage, embedder, query, top_k=8)
        
        if not chunks:
            print("No relevant chunks found.")
            # Still log the query even if no chunks found
            query_answer_pairs.append({
                'query': query,
                'answer': 'No relevant chunks found.'
            })
            continue
        
        print(f"Found {len(chunks)} relevant chunk(s)")
        print("\nGenerating response...")
        
        try:
            # Use more chunks with larger context window - can include 6-8 chunks with full detail
            response = llm_client.generate_rag_response(query, chunks, max_chunks=min(6, len(chunks)))
            print("\n" + "-"*80)
            print("Answer:")
            print("-"*80)
            print(response)
            print("-"*80)
            
            # Log query and answer
            query_answer_pairs.append({
                'query': query,
                'answer': response
            })
        except Exception as e:
            error_msg = f"Error generating response: {e}"
            print(f"\n{error_msg}")
            import traceback
            traceback.print_exc()
            
            # Log query with error
            query_answer_pairs.append({
                'query': query,
                'answer': f"[ERROR] {error_msg}"
            })
    
    # Phase 3: Save log and cleanup
    print("\n" + "="*80)
    print("Ending session...")
    print("="*80)
    
    # Save session log before clearing data
    if query_answer_pairs:
        print("\nSaving session log...")
        save_session_log(paper_ids, query_answer_pairs)
    
    clear_all_data(storage, vector_size=embedding_dim)
    print("\nSession ended. All data cleared.")


if __name__ == "__main__":
    main()
