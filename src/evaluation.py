import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from src.chunking import Chunker
from src.metrics import MetricsCalculator, aggregate_metrics
from src.llm_client import VLLMClient
from typing import List, Dict, Optional

class Evaluator:
    def __init__(self, embedder, storage, llm_client: Optional[VLLMClient] = None):
        """
        Initialize evaluator.
        
        Args:
            embedder: PaperEmbedder instance
            storage: StorageManager instance
            llm_client: VLLMClient instance (optional, for RAG evaluation)
        """
        self.embedder = embedder
        self.storage = storage
        self.llm_client = llm_client
        self.metrics_calculator = MetricsCalculator()

    def run_experiment(self, papers, strategy="recursive", chunk_size=500, chunk_overlap=50):
        """
        Runs the basic retrieval evaluation experiment (backward compatible).
        
        Args:
            papers (list): List of paper metadata dicts.
            strategy (str): Chunking strategy name.
            chunk_size (int): Size of chunks.
            chunk_overlap (int): Overlap size.
            
        Returns:
            dict: Metrics {hit_rate, mrr}
        """
        chunker = Chunker(strategy=strategy, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        
        all_chunks = []
        doc_indices = [] # Map chunk index to paper ID
        
        print(f"Preparing data for strategy: {strategy}...")
        
        # 1. Chunking
        for paper in papers:
            pdf_bytes = self.storage.get_paper_pdf(paper['id'])
            if not pdf_bytes:
                continue
                
            text = chunker.extract_text_from_pdf(pdf_bytes)
            chunks = chunker.chunk_text(text)
            
            for chunk in chunks:
                if len(chunk) < 50: # Skip very small chunks
                    continue
                all_chunks.append(chunk)
                doc_indices.append(paper['id'])
                
        if not all_chunks:
            return {"hit_rate": 0, "mrr": 0, "num_chunks": 0}

        # 2. Embedding Chunks
        print(f"Embedding {len(all_chunks)} chunks (this might take a while)...")
        chunk_embeddings = self.embedder.embed_texts(all_chunks)
        
        # 3. Retrieval & Evaluation
        print("Evaluating retrieval performance...")
        hits = 0
        reciprocal_ranks = 0
        
        for paper in papers:
            abstract = paper['abstract']
            target_id = paper['id']
            
            # Embed query (abstract)
            query_emb = self.embedder.embed_texts([abstract])
            
            # Similarity
            scores = cosine_similarity(query_emb, chunk_embeddings)[0]
            
            # Top K
            k = 5
            top_k_indices = np.argsort(scores)[::-1][:k]
            
            # Check for hit
            found = False
            for rank, idx in enumerate(top_k_indices):
                retrieved_id = doc_indices[idx]
                if retrieved_id == target_id:
                    hits += 1
                    reciprocal_ranks += 1 / (rank + 1)
                    found = True
                    break
                    
        num_queries = len(papers)
        metrics = {
            "hit_rate": hits / num_queries if num_queries > 0 else 0,
            "mrr": reciprocal_ranks / num_queries if num_queries > 0 else 0,
            "num_chunks": len(all_chunks)
        }
        
        return metrics

    def run_rag_experiment(
        self,
        papers: List[Dict],
        test_queries: List[Dict],
        strategy: str = "recursive",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        top_k: int = 5
    ) -> Dict:
        """
        Run full RAG evaluation experiment with generation and comprehensive metrics.
        
        Args:
            papers: List of paper metadata dicts
            test_queries: List of test query dicts with 'query', 'reference', 'paper_id'
            strategy: Chunking strategy
            chunk_size: Chunk size
            chunk_overlap: Overlap size
            top_k: Number of chunks to retrieve
            
        Returns:
            Dictionary with aggregated metrics and per-query results
        """
        if self.llm_client is None:
            raise ValueError("LLM client required for RAG experiment. Initialize Evaluator with VLLMClient.")
        
        print(f"\n=== Running RAG Experiment: {strategy} ===")
        
        # 1. Chunk all papers
        all_chunks, doc_indices, chunk_embeddings = self._prepare_chunks(
            papers, strategy, chunk_size, chunk_overlap
        )
        
        if not all_chunks:
            return {"error": "No chunks generated", "num_chunks": 0}
        
        # 2. Evaluate each query
        print(f"Evaluating {len(test_queries)} queries...")
        all_query_metrics = []
        
        for i, query_dict in enumerate(test_queries):
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(test_queries)} queries...")
            
            query_metrics = self._evaluate_single_query(
                query_dict,
                all_chunks,
                doc_indices,
                chunk_embeddings,
                top_k
            )
            
            all_query_metrics.append(query_metrics)
        
        # 3. Aggregate metrics
        aggregated = aggregate_metrics(all_query_metrics)
        
        # 4. Add metadata
        result = {
            "strategy": strategy,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "num_chunks": len(all_chunks),
            "num_queries": len(test_queries),
            "metrics": aggregated,
            "per_query_metrics": all_query_metrics
        }
        
        return result
    
    def _prepare_chunks(
        self,
        papers: List[Dict],
        strategy: str,
        chunk_size: int,
        chunk_overlap: int
    ):
        """Chunk and embed all papers."""
        chunker = Chunker(strategy=strategy, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        
        all_chunks = []
        doc_indices = []
        
        print(f"Chunking papers with strategy: {strategy}...")
        
        for paper in papers:
            pdf_bytes = self.storage.get_paper_pdf(paper['id'])
            if not pdf_bytes:
                continue
            
            text = chunker.extract_text_from_pdf(pdf_bytes)
            chunks = chunker.chunk_text(text)
            
            for chunk in chunks:
                if len(chunk) < 50:  # Skip very small chunks
                    continue
                all_chunks.append(chunk)
                doc_indices.append(paper['id'])
        
        print(f"Generated {len(all_chunks)} chunks")
        print(f"Embedding chunks...")
        chunk_embeddings = self.embedder.embed_texts(all_chunks)
        
        return all_chunks, doc_indices, chunk_embeddings
    
    def _retrieve_chunks(
        self,
        query: str,
        all_chunks: List[str],
        doc_indices: List[str],
        chunk_embeddings: np.ndarray,
        k: int
    ):
        """Retrieve top-k chunks for a query."""
        # Embed query
        query_emb = self.embedder.embed_texts([query])
        
        # Calculate similarities
        scores = cosine_similarity(query_emb, chunk_embeddings)[0]
        
        # Get top-k indices
        top_k_indices = np.argsort(scores)[::-1][:k]
        
        # Get chunks and their source papers
        retrieved_chunks = [all_chunks[idx] for idx in top_k_indices]
        retrieved_ids = [doc_indices[idx] for idx in top_k_indices]
        
        return retrieved_chunks, retrieved_ids
    
    def _evaluate_single_query(
        self,
        query_dict: Dict,
        all_chunks: List[str],
        doc_indices: List[str],
        chunk_embeddings: np.ndarray,
        k: int
    ) -> Dict[str, float]:
        """Evaluate a single query with retrieval + generation + metrics."""
        query = query_dict['query']
        reference = query_dict['reference']
        relevant_paper_id = query_dict['paper_id']
        
        # 1. Retrieve chunks
        retrieved_chunks, retrieved_ids = self._retrieve_chunks(
            query, all_chunks, doc_indices, chunk_embeddings, k
        )
        
        # 2. Generate response
        try:
            generated_response = self.llm_client.generate_rag_response(
                query, retrieved_chunks, max_chunks=k
            )
        except Exception as e:
            print(f"Error generating response: {e}")
            generated_response = ""
        
        # 3. Calculate metrics
        metrics = self.metrics_calculator.calculate_all_metrics(
            prediction=generated_response,
            reference=reference,
            retrieved_ids=retrieved_ids,
            relevant_ids=[relevant_paper_id],
            k=k
        )
        
        return metrics

