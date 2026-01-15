"""
Stage 3: Inference-Based Qualitative Validation
Purpose: After selecting best chunking strategy, validate with inference model.
This stage is ONLY for qualitative validation and must NOT be used for ranking strategies.
"""
import json
import random
from typing import List, Dict, Optional
from src.chunking import Chunker
from src.embedder import PaperEmbedder
from src.storage_manager import StorageManager
from src.llm_client import VLLMClient

class Stage3Evaluator:
    """Evaluates RAG outputs qualitatively using inference model."""
    
    def __init__(
        self,
        embedder: PaperEmbedder,
        storage: StorageManager,
        llm_client: VLLMClient
    ):
        """
        Initialize Stage 3 evaluator.
        
        Args:
            embedder: PaperEmbedder instance
            storage: StorageManager instance
            llm_client: VLLMClient instance for inference
        """
        self.embedder = embedder
        self.storage = storage
        self.llm_client = llm_client
    
    def validate_strategy(
        self,
        papers: List[Dict],
        queries: List[Dict],
        strategy: str,
        chunk_size: int,
        chunk_overlap: int,
        top_k: int = 5,
        max_papers: int = 10
    ) -> Dict:
        """
        Validate a chunking strategy using inference model.
        Only run this AFTER selecting the best strategy from Stages 1 & 2.
        
        Args:
            papers: List of paper metadata dicts
            queries: List of query dicts
            strategy: Chunking strategy name
            chunk_size: Chunk size parameter
            chunk_overlap: Chunk overlap parameter
            top_k: Number of top chunks to retrieve
            max_papers: Maximum number of papers to validate (for efficiency)
            
        Returns:
            Dictionary with qualitative results:
            - results: List of dicts, each containing:
                - query: Query dict with 'query', 'paper_id', 'title', 'claim_type', 'gold_sections'
                - retrieved_chunks: List of retrieved chunk texts
                - response: Generated response text
                - error: Error message (if any)
            - strategy: Chunking strategy name
            - total_queries: Total number of queries evaluated
            - successful_queries: Number of queries without errors
            - errors: List of all error messages (if any)
        """
        print(f"  Stage 3: Qualitative validation for strategy '{strategy}'...")
        print(f"    Note: This is for qualitative validation only, not for ranking strategies.")
        
        # Check vLLM health before proceeding
        print("    Checking vLLM server availability...")
        if not self.llm_client.check_health():
            error_msg = "vLLM server is not available. Skipping Stage 3 validation."
            print(f"    ERROR: {error_msg}")
            return {
                'results': [],
                'errors': [error_msg],
                'strategy': strategy,
                'total_queries': 0,
                'successful_queries': 0
            }
        print("    vLLM server is available.")
        
        # Limit to subset of papers for qualitative validation
        papers_subset = papers[:max_papers]
        queries_subset = [q for q in queries if q['paper_id'] in [p['id'] for p in papers_subset]]
        
        print(f"    Validating on {len(papers_subset)} papers with {len(queries_subset)} queries")
        
        # Chunk papers
        chunker = Chunker(strategy=strategy, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        all_chunks = []
        chunk_metadata = []
        
        for paper in papers_subset:
            pdf_bytes = self.storage.get_paper_pdf(paper['id'])
            if not pdf_bytes:
                print(f"    Warning: Could not load PDF for paper {paper['id']}")
                continue
            
            text = chunker.extract_text_from_pdf(pdf_bytes)
            paper_chunks = chunker.chunk_paper(paper['id'], text)
            
            for chunk_meta in paper_chunks:
                if len(chunk_meta.chunk_text) >= 50:
                    all_chunks.append(chunk_meta.chunk_text)
                    chunk_metadata.append(chunk_meta)
        
        if not all_chunks:
            error_msg = "No chunks generated. Cannot proceed with validation."
            print(f"    ERROR: {error_msg}")
            return {
                'results': [],
                'errors': [error_msg],
                'strategy': strategy,
                'total_queries': 0,
                'successful_queries': 0
            }
        
        # Embed chunks
        print(f"    Embedding {len(all_chunks)} chunks...")
        chunk_embeddings = self.embedder.embed_texts(all_chunks)
        
        # Generate responses for each query
        results = []
        errors = []
        
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity
        
        for idx, query_dict in enumerate(queries_subset):
            query_text = query_dict['query']
            
            # Retrieve chunks
            query_emb = self.embedder.embed_texts([query_text])
            scores = cosine_similarity(query_emb, chunk_embeddings)[0]
            top_k_indices = np.argsort(scores)[::-1][:top_k]
            
            retrieved_chunks = [all_chunks[i] for i in top_k_indices]
            
            # Generate response
            error = None
            try:
                response = self.llm_client.generate_rag_response(
                    query_text,
                    retrieved_chunks,
                    max_chunks=top_k
                )
                if not response:
                    error = f"Empty response for query {idx+1}/{len(queries_subset)}"
                    print(f"    WARNING: {error}")
                    response = f"[ERROR: Empty response]"
            except Exception as e:
                import traceback
                error = f"Error generating response for query {idx+1}/{len(queries_subset)}: {e}"
                print(f"    ERROR: {error}")
                print(f"    Traceback: {traceback.format_exc()}")
                response = f"[ERROR: {str(e)}]"
            
            # Create paired result
            result_item = {
                'query': query_dict,
                'retrieved_chunks': retrieved_chunks,
                'response': response
            }
            
            if error:
                result_item['error'] = error
                errors.append(error)
            
            results.append(result_item)
            
            # Progress indicator
            if (idx + 1) % 10 == 0:
                print(f"    Processed {idx+1}/{len(queries_subset)} queries...")
        
        output = {
            'results': results,
            'strategy': strategy,
            'total_queries': len(results),
            'successful_queries': len([r for r in results if 'error' not in r])
        }
        
        if errors:
            output['errors'] = errors
            print(f"    WARNING: {len(errors)} errors occurred during validation")
        else:
            print(f"    Successfully generated {len(results)} responses")
        
        return output
    
    def save_results(
        self,
        validation_results: Dict,
        best_strategy: Dict,
        output_json: str = 'stage3_results.json',
        output_md: str = 'stage3_results.md'
    ):
        """
        Save Stage 3 results to JSON and generate Markdown report.
        
        Args:
            validation_results: Results dict from validate_strategy()
            best_strategy: Best strategy dict with stage1 and stage2 metrics
            output_json: Path to JSON output file
            output_md: Path to Markdown output file
        """
        # Save JSON
        output = {
            'best_strategy': best_strategy,
            'validation_results': validation_results
        }
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, default=str)
        
        # Generate Markdown report
        self._generate_report(output, output_md)
        
        print(f"Stage 3 results saved to {output_json} and {output_md}")
    
    def _generate_report(self, output: Dict, output_path: str):
        """Generate Stage 3 Markdown report."""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# Stage 3: Inference-Based Qualitative Validation Results\n\n")
            f.write("This stage provides qualitative validation of the best-performing chunking strategy.\n\n")
            
            best_strat = output.get('best_strategy')
            if best_strat:
                f.write(f"**Best Strategy:** {best_strat['strategy_name']}\n\n")
                f.write(f"- Strategy: {best_strat['strategy']}\n")
                f.write(f"- Chunk Size: {best_strat['chunk_size']}\n")
                f.write(f"- Chunk Overlap: {best_strat['chunk_overlap']}\n\n")
            
            validation = output.get('validation_results', {})
            if validation:
                f.write(f"**Validation Results:**\n\n")
                f.write(f"- Total queries evaluated: {validation.get('total_queries', 0)}\n")
                f.write(f"- Successful responses: {validation.get('successful_queries', 0)}\n")
                if validation.get('errors'):
                    f.write(f"- Errors: {len(validation.get('errors', []))}\n")
                f.write("\n**Note:** Review the query-response pairs in `stage3_results.json` for qualitative assessment.\n\n")
                
                # Show 15 random example results
                results = validation.get('results', [])
                if results:
                    f.write("### Sample Query-Response Pairs (15 Random Examples)\n\n")
                    # Pick 15 random examples (or all if fewer than 15)
                    num_examples = min(15, len(results))
                    random_examples = random.sample(results, num_examples)
                    
                    for i, result in enumerate(random_examples, 1):
                        query = result.get('query', {})
                        response = result.get('response', '')
                        f.write(f"**Example {i}:**\n\n")
                        f.write(f"- **Query:** {query.get('query', 'N/A')}\n")
                        f.write(f"- **Paper:** {query.get('title', 'N/A')}\n")
                        f.write(f"- **Response:** {response[:200]}{'...' if len(response) > 200 else ''}\n\n")

