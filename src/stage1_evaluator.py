"""
Stage 1: Document-Level Retrieval Evaluation
Purpose: Determine whether queries retrieve chunks from the correct paper(s).
Only paper identity matters at this stage.
"""
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple
from src.chunking import Chunker, ChunkMetadata
from src.embedder import PaperEmbedder
from src.storage_manager import StorageManager

class Stage1Evaluator:
    """Evaluates document-level retrieval performance."""
    
    def __init__(self, embedder: PaperEmbedder, storage: StorageManager):
        """
        Initialize Stage 1 evaluator.
        
        Args:
            embedder: PaperEmbedder instance
            storage: StorageManager instance
        """
        self.embedder = embedder
        self.storage = storage
    
    def evaluate_strategy(
        self,
        papers: List[Dict],
        queries: List[Dict],
        strategy: str,
        chunk_size: int,
        chunk_overlap: int,
        top_k: int = 5
    ) -> Dict:
        """
        Evaluate a chunking strategy for document-level retrieval.
        
        Args:
            papers: List of paper metadata dicts
            queries: List of query dicts with 'query' and 'paper_id'
            strategy: Chunking strategy name
            chunk_size: Chunk size parameter
            chunk_overlap: Chunk overlap parameter
            top_k: Number of top chunks to retrieve
            
        Returns:
            Dictionary with metrics:
            - paper_recall_at_k: Percentage of queries with correct paper in top-k
            - paper_mrr: Mean Reciprocal Rank for paper retrieval
            - queries_with_correct_paper: Percentage of queries with ≥1 correct paper
            - num_chunks: Total number of chunks generated
        """
        print(f"  Stage 1: Evaluating strategy '{strategy}'...")
        
        # 1. Chunk papers
        chunker = Chunker(strategy=strategy, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        all_chunks = []
        chunk_metadata = []
        
        for paper in papers:
            pdf_bytes = self.storage.get_paper_pdf(paper['id'])
            if not pdf_bytes:
                continue
            
            text = chunker.extract_text_from_pdf(pdf_bytes)
            paper_chunks = chunker.chunk_paper(paper['id'], text)
            
            for chunk_meta in paper_chunks:
                if len(chunk_meta.chunk_text) >= 50:  # Skip very small chunks
                    all_chunks.append(chunk_meta.chunk_text)
                    chunk_metadata.append(chunk_meta)
        
        if not all_chunks:
            return {
                'paper_recall_at_k': 0.0,
                'paper_mrr': 0.0,
                'queries_with_correct_paper': 0.0,
                'num_chunks': 0
            }
        
        print(f"    Generated {len(all_chunks)} chunks")
        
        # 2. Embed chunks
        print(f"    Embedding {len(all_chunks)} chunks...")
        chunk_embeddings = self.embedder.embed_texts(all_chunks)
        
        # 3. Evaluate each query
        correct_papers_at_k = 0
        reciprocal_ranks = []
        queries_with_correct = 0
        
        for query_dict in queries:
            query_text = query_dict['query']
            correct_paper_id = query_dict['paper_id']
            
            # Embed query
            query_emb = self.embedder.embed_texts([query_text])
            
            # Calculate similarities
            scores = cosine_similarity(query_emb, chunk_embeddings)[0]
            
            # Get top-k indices
            top_k_indices = np.argsort(scores)[::-1][:top_k]
            
            # Check if correct paper is in top-k
            found_correct_paper = False
            rank = None
            
            for idx, chunk_idx in enumerate(top_k_indices):
                retrieved_paper_id = chunk_metadata[chunk_idx].paper_id
                if retrieved_paper_id == correct_paper_id:
                    found_correct_paper = True
                    rank = idx + 1
                    break
            
            if found_correct_paper:
                correct_papers_at_k += 1
                queries_with_correct += 1
                reciprocal_ranks.append(1.0 / rank)
            else:
                reciprocal_ranks.append(0.0)
        
        num_queries = len(queries)
        metrics = {
            'paper_recall_at_k': correct_papers_at_k / num_queries if num_queries > 0 else 0.0,
            'paper_mrr': np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0,
            'queries_with_correct_paper': queries_with_correct / num_queries if num_queries > 0 else 0.0,
            'num_chunks': len(all_chunks)
        }
        
        return metrics
    
    def run_all_strategies(
        self,
        papers: List[Dict],
        queries: List[Dict],
        chunking_strategies: List[Dict],
        top_k: int = 5
    ) -> List[Dict]:
        """
        Run evaluation for all chunking strategies.
        
        Args:
            papers: List of paper metadata dicts
            queries: List of query dicts
            chunking_strategies: List of strategy dicts with 'name', 'strategy', 'chunk_size', 'chunk_overlap'
            top_k: Number of top chunks to retrieve
            
        Returns:
            List of result dicts, each containing strategy info and metrics
        """
        results = []
        
        for i, strat in enumerate(chunking_strategies):
            print(f"\n[{i+1}/{len(chunking_strategies)}] Strategy: {strat['name']}")
            
            metrics = self.evaluate_strategy(
                papers=papers,
                queries=queries,
                strategy=strat['strategy'],
                chunk_size=strat['chunk_size'],
                chunk_overlap=strat['chunk_overlap'],
                top_k=top_k
            )
            
            result = {
                'strategy_name': strat['name'],
                'strategy': strat['strategy'],
                'chunk_size': strat['chunk_size'],
                'chunk_overlap': strat['chunk_overlap'],
                'metrics': metrics
            }
            
            results.append(result)
            print(f"  Paper Recall@5: {metrics['paper_recall_at_k']:.4f}, Paper MRR: {metrics['paper_mrr']:.4f}")
        
        return results
    
    def save_results(self, results: List[Dict], output_json: str = 'stage1_results.json', output_md: str = 'stage1_results.md'):
        """
        Save Stage 1 results to JSON and generate Markdown report.
        
        Args:
            results: List of result dicts from run_all_strategies()
            output_json: Path to JSON output file
            output_md: Path to Markdown output file
        """
        # Save JSON
        output = {'results': results}
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, default=str)
        
        # Generate Markdown report
        self._generate_report(results, output_md)
        
        print(f"Stage 1 results saved to {output_json} and {output_md}")
    
    def _generate_report(self, results: List[Dict], output_path: str):
        """Generate Stage 1 Markdown report."""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# Stage 1: Document-Level Retrieval Evaluation Results\n\n")
            f.write("This stage evaluates whether queries retrieve chunks from the correct paper(s).\n\n")
            
            f.write("| Strategy | Paper Recall@5 | Paper MRR | Queries with Correct Paper |\n")
            f.write("|----------|----------------|-----------|----------------------------|\n")
            
            for result in results:
                metrics = result['metrics']
                f.write(f"| {result['strategy_name']} | "
                       f"{metrics['paper_recall_at_k']:.4f} | "
                       f"{metrics['paper_mrr']:.4f} | "
                       f"{metrics['queries_with_correct_paper']:.4f} |\n")

