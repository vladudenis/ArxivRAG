"""
Stage 2: Chunk-Level Evidence Retrieval Evaluation
Purpose: Determine whether retrieved chunks contain the correct evidence.
Uses content-based section matching (not just section_id) to avoid bias.
"""
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Set
from src.chunking import Chunker, ChunkMetadata
from src.embedder import PaperEmbedder
from src.storage_manager import StorageManager
from src.section_detector import SectionDetector

class Stage2Evaluator:
    """Evaluates chunk-level evidence retrieval performance."""
    
    def __init__(self, embedder: PaperEmbedder, storage: StorageManager):
        """
        Initialize Stage 2 evaluator.
        
        Args:
            embedder: PaperEmbedder instance
            storage: StorageManager instance
        """
        self.embedder = embedder
        self.storage = storage
        self.section_detector = SectionDetector()
    
    def _get_chunk_sections_by_content(
        self,
        chunk_text: str,
        chunk_start_pos: int,
        chunk_end_pos: int,
        paper_text: str,
        paper_sections: List[tuple]
    ) -> Set[str]:
        """
        Determine which sections a chunk actually contains based on content overlap.
        This avoids bias from section_id assignment.
        
        Args:
            chunk_text: The chunk text
            chunk_start_pos: Start character position of chunk in paper text
            chunk_end_pos: End character position of chunk in paper text
            paper_text: Full paper text
            paper_sections: List of (start_line_index, section_name, section_text) tuples
            
        Returns:
            Set of section names that this chunk overlaps with
        """
        if not paper_sections:
            return set()
        
        # Convert line indices to character positions
        lines = paper_text.split('\n')
        section_char_ranges = []
        
        for i, (section_start_line, section_name, section_text) in enumerate(paper_sections):
            # Calculate character position range for this section
            char_start = sum(len(line) + 1 for line in lines[:section_start_line])
            
            # Find end of section (start of next section or end of text)
            if i + 1 < len(paper_sections):
                section_end_line = paper_sections[i + 1][0]
            else:
                section_end_line = len(lines)
            
            char_end = sum(len(line) + 1 for line in lines[:section_end_line])
            
            section_char_ranges.append((char_start, char_end, section_name))
        
        # Find sections that overlap with this chunk
        overlapping_sections = set()
        
        # Check overlap between chunk and each section
        for section_start, section_end, section_name in section_char_ranges:
            # Check if chunk overlaps with section
            # Overlap exists if: chunk_start < section_end AND chunk_end > section_start
            if chunk_start_pos < section_end and chunk_end_pos > section_start:
                # Calculate overlap percentage
                overlap_start = max(chunk_start_pos, section_start)
                overlap_end = min(chunk_end_pos, section_end)
                overlap_size = max(0, overlap_end - overlap_start)
                chunk_size = chunk_end_pos - chunk_start_pos
                
                # If overlap is significant (>10% of chunk), consider it relevant
                if chunk_size > 0 and overlap_size / chunk_size > 0.1:
                    overlapping_sections.add(section_name)
        
        return overlapping_sections
    
    def _find_chunk_position_in_text(
        self,
        chunk_text: str,
        paper_text: str,
        paper_id: str
    ) -> tuple:
        """
        Find the character position of a chunk in the paper text.
        
        Args:
            chunk_text: The chunk text
            paper_text: Full paper text
            paper_id: Paper ID (for error messages)
            
        Returns:
            (start_pos, end_pos) tuple, or (0, len(paper_text)) if not found
        """
        # Try to find chunk in paper text
        # Use first 100 chars of chunk as anchor
        anchor = chunk_text[:min(100, len(chunk_text))].strip()
        
        if anchor:
            pos = paper_text.find(anchor)
            if pos != -1:
                return (pos, pos + len(chunk_text))
        
        # Fallback: if not found, return full text range
        # This is a conservative fallback - chunk might span entire paper
        return (0, len(paper_text))
    
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
        Evaluate a chunking strategy for chunk-level evidence retrieval.
        Uses content-based section matching to avoid bias.
        
        Args:
            papers: List of paper metadata dicts
            queries: List of query dicts with 'query', 'paper_id', 'gold_sections'
            strategy: Chunking strategy name
            chunk_size: Chunk size parameter
            chunk_overlap: Chunk overlap parameter
            top_k: Number of top chunks to retrieve
            
        Returns:
            Dictionary with metrics:
            - recall_at_k: Chunk-level recall@k
            - mrr: Mean Reciprocal Rank
            - precision_at_k: Precision@k
            - num_chunks: Total number of chunks generated
        """
        print(f"  Stage 2: Evaluating strategy '{strategy}'...")
        print(f"    Using content-based section matching (unbiased evaluation)")
        
        # 1. Chunk papers and store paper texts for section detection
        chunker = Chunker(strategy=strategy, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        all_chunks = []
        chunk_metadata = []
        paper_texts = {}  # paper_id -> full_text
        paper_sections_cache = {}  # paper_id -> list of sections
        
        for paper in papers:
            pdf_bytes = self.storage.get_paper_pdf(paper['id'])
            if not pdf_bytes:
                continue
            
            text = chunker.extract_text_from_pdf(pdf_bytes)
            paper_texts[paper['id']] = text
            
            # Detect sections for this paper
            sections = self.section_detector.detect_sections(text)
            paper_sections_cache[paper['id']] = sections
            
            paper_chunks = chunker.chunk_paper(paper['id'], text)
            
            for chunk_meta in paper_chunks:
                if len(chunk_meta.chunk_text) >= 50:  # Skip very small chunks
                    all_chunks.append(chunk_meta.chunk_text)
                    chunk_metadata.append(chunk_meta)
        
        if not all_chunks:
            return {
                'recall_at_k': 0.0,
                'mrr': 0.0,
                'precision_at_k': 0.0,
                'num_chunks': 0
            }
        
        print(f"    Generated {len(all_chunks)} chunks")
        
        # 2. Embed chunks
        print(f"    Embedding {len(all_chunks)} chunks...")
        chunk_embeddings = self.embedder.embed_texts(all_chunks)
        
        # 3. Pre-compute chunk section mappings (content-based)
        print(f"    Computing content-based section mappings for chunks...")
        chunk_sections_map = {}  # chunk_index -> set of section names
        
        for idx, chunk_meta in enumerate(chunk_metadata):
            paper_id = chunk_meta.paper_id
            if paper_id not in paper_texts:
                continue
            
            paper_text = paper_texts[paper_id]
            sections = paper_sections_cache[paper_id]
            
            # Find chunk position in paper text
            chunk_start, chunk_end = self._find_chunk_position_in_text(
                chunk_meta.chunk_text,
                paper_text,
                paper_id
            )
            
            # Get sections this chunk overlaps with (content-based)
            chunk_sections = self._get_chunk_sections_by_content(
                chunk_meta.chunk_text,
                chunk_start,
                chunk_end,
                paper_text,
                sections
            )
            
            chunk_sections_map[idx] = chunk_sections
        
        # 4. Evaluate each query
        all_recalls = []
        all_precisions = []
        reciprocal_ranks = []
        
        for query_dict in queries:
            query_text = query_dict['query']
            correct_paper_id = query_dict['paper_id']
            gold_sections = set(query_dict.get('gold_sections', []))
            
            # Normalize gold sections for case-insensitive matching
            gold_sections_normalized = {s.lower().strip() for s in gold_sections} if gold_sections else set()
            
            # Embed query
            query_emb = self.embedder.embed_texts([query_text])
            
            # Calculate similarities
            scores = cosine_similarity(query_emb, chunk_embeddings)[0]
            
            # Get top-k indices
            top_k_indices = np.argsort(scores)[::-1][:top_k]
            
            # Find relevant chunks using content-based section matching
            relevant_chunk_indices = set()
            for idx, chunk_meta in enumerate(chunk_metadata):
                if chunk_meta.paper_id == correct_paper_id:
                    if not gold_sections_normalized:
                        # If no gold sections specified, all chunks from correct paper are relevant
                        relevant_chunk_indices.add(idx)
                    else:
                        # Get sections this chunk actually contains (content-based)
                        chunk_sections = chunk_sections_map.get(idx, set())
                        
                        # Check if any chunk section matches any gold section
                        is_relevant = False
                        for chunk_section in chunk_sections:
                            chunk_section_normalized = chunk_section.lower().strip()
                            for gold_section in gold_sections_normalized:
                                if (chunk_section_normalized == gold_section or
                                    gold_section in chunk_section_normalized or
                                    chunk_section_normalized in gold_section):
                                    is_relevant = True
                                    break
                            if is_relevant:
                                break
                        
                        if is_relevant:
                            relevant_chunk_indices.add(idx)
            
            if not relevant_chunk_indices:
                # No relevant chunks found, skip this query
                all_recalls.append(0.0)
                all_precisions.append(0.0)
                reciprocal_ranks.append(0.0)
                continue
            
            # Calculate recall@k
            retrieved_relevant = len(set(top_k_indices) & relevant_chunk_indices)
            recall = retrieved_relevant / len(relevant_chunk_indices)
            all_recalls.append(recall)
            
            # Calculate precision@k
            precision = retrieved_relevant / top_k if top_k > 0 else 0.0
            all_precisions.append(precision)
            
            # Calculate MRR
            mrr = 0.0
            for rank, chunk_idx in enumerate(top_k_indices, start=1):
                if chunk_idx in relevant_chunk_indices:
                    mrr = 1.0 / rank
                    break
            reciprocal_ranks.append(mrr)
        
        metrics = {
            'recall_at_k': np.mean(all_recalls) if all_recalls else 0.0,
            'mrr': np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0,
            'precision_at_k': np.mean(all_precisions) if all_precisions else 0.0,
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
            print(f"  Recall@5: {metrics['recall_at_k']:.4f}, MRR: {metrics['mrr']:.4f}, Precision@5: {metrics['precision_at_k']:.4f}")
        
        return results
    
    def save_results(self, results: List[Dict], output_json: str = 'stage2_results.json', output_md: str = 'stage2_results.md'):
        """
        Save Stage 2 results to JSON and generate Markdown report.
        
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
        
        print(f"Stage 2 results saved to {output_json} and {output_md}")
    
    def _generate_report(self, results: List[Dict], output_path: str):
        """Generate Stage 2 Markdown report."""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# Stage 2: Chunk-Level Evidence Retrieval Evaluation Results\n\n")
            f.write("This stage evaluates whether retrieved chunks contain the correct evidence.\n\n")
            
            f.write("| Strategy | Recall@5 | MRR | Precision@5 |\n")
            f.write("|----------|----------|-----|-------------|\n")
            
            for result in results:
                metrics = result['metrics']
                f.write(f"| {result['strategy_name']} | "
                       f"{metrics['recall_at_k']:.4f} | "
                       f"{metrics['mrr']:.4f} | "
                       f"{metrics['precision_at_k']:.4f} |\n")
    
    @staticmethod
    def select_best_strategy(stage1_results: List[Dict], stage2_results: List[Dict]) -> Dict:
        """
        Select best strategy based on Stage 1 and Stage 2 metrics.
        
        Args:
            stage1_results: List of Stage 1 result dicts
            stage2_results: List of Stage 2 result dicts
            
        Returns:
            Best strategy dictionary with stage1 and stage2 metrics
        """
        # Match results by strategy_name
        stage1_dict = {r['strategy_name']: r for r in stage1_results}
        stage2_dict = {r['strategy_name']: r for r in stage2_results}
        
        # Score each strategy (weighted combination of metrics)
        strategy_scores = []
        
        for strategy_name in stage1_dict.keys():
            if strategy_name not in stage2_dict:
                continue
            
            stage1_metrics = stage1_dict[strategy_name]['metrics']
            stage2_metrics = stage2_dict[strategy_name]['metrics']
            
            # Combine Stage 1 and Stage 2 metrics
            # Weight: Stage 2 is more important (chunk-level evidence)
            stage1_score = (
                stage1_metrics['paper_recall_at_k'] * 0.3 +
                stage1_metrics['paper_mrr'] * 0.2
            )
            stage2_score = (
                stage2_metrics['recall_at_k'] * 0.3 +
                stage2_metrics['mrr'] * 0.1 +
                stage2_metrics['precision_at_k'] * 0.1
            )
            
            total_score = stage1_score + stage2_score
            
            # Combine into full result dict
            combined_result = {
                'strategy_name': strategy_name,
                'strategy': stage1_dict[strategy_name]['strategy'],
                'chunk_size': stage1_dict[strategy_name]['chunk_size'],
                'chunk_overlap': stage1_dict[strategy_name]['chunk_overlap'],
                'stage1': stage1_metrics,
                'stage2': stage2_metrics
            }
            
            strategy_scores.append({
                'strategy': combined_result,
                'score': total_score
            })
        
        # Sort by score
        strategy_scores.sort(key=lambda x: x['score'], reverse=True)
        
        best = strategy_scores[0]['strategy']
        
        print(f"\nBest strategy selected: {best['strategy_name']}")
        print(f"  Score: {strategy_scores[0]['score']:.4f}")
        print(f"  Stage 1 - Paper Recall@5: {best['stage1']['paper_recall_at_k']:.4f}")
        print(f"  Stage 2 - Recall@5: {best['stage2']['recall_at_k']:.4f}, MRR: {best['stage2']['mrr']:.4f}")
        
        return best

