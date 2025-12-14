"""
Test query generation and management for RAG evaluation.
"""
import random
from typing import List, Dict

class QueryGenerator:
    """Generates test queries from paper abstracts."""
    
    def __init__(self, papers: List[Dict]):
        """
        Initialize with list of papers.
        
        Args:
            papers: List of paper dictionaries with 'id', 'abstract', 'title' fields
        """
        self.papers = papers
    
    def generate_queries(self, num_queries: int = None) -> List[Dict]:
        """
        Generate test queries from paper abstracts.
        
        For this implementation, we use abstracts as queries and expect
        the RAG system to reconstruct them from the paper chunks.
        
        Args:
            num_queries: Number of queries to generate. If None, uses all papers.
            
        Returns:
            List of query dictionaries with 'query', 'reference', 'paper_id', 'type'
        """
        queries = []
        
        papers_to_use = self.papers if num_queries is None else random.sample(
            self.papers, min(num_queries, len(self.papers))
        )
        
        for paper in papers_to_use:
            # Use abstract as both query and reference answer
            # The RAG system should reconstruct the abstract from paper chunks
            query_dict = {
                'query': paper['abstract'],
                'reference': paper['abstract'],  # Reference for ROUGE/BLEU
                'paper_id': paper['id'],
                'title': paper['title'],
                'type': 'abstract_reconstruction'
            }
            queries.append(query_dict)
        
        return queries
    
    def generate_custom_queries(self) -> List[Dict]:
        """
        Generate custom queries for more diverse evaluation.
        This is a placeholder for future enhancement.
        """
        # TODO: Could add question generation, summarization tasks, etc.
        return []

def load_test_queries(papers: List[Dict], num_queries: int = None) -> List[Dict]:
    """
    Convenience function to load test queries.
    
    Args:
        papers: List of paper dictionaries
        num_queries: Number of queries to generate
        
    Returns:
        List of query dictionaries
    """
    generator = QueryGenerator(papers)
    return generator.generate_queries(num_queries)
