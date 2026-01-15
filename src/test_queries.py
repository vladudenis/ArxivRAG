"""
Test query generation for RAG evaluation.
Generates 3-6 natural-language claims per paper from abstract/conclusion.
"""
import random
import re
from typing import List, Dict, Set
from src.section_detector import SectionDetector

class QueryGenerator:
    """Generates test queries (claims) from paper abstracts and conclusions."""
    
    def __init__(self, papers: List[Dict]):
        """
        Initialize with list of papers.
        
        Args:
            papers: List of paper dictionaries with 'id', 'abstract', 'title' fields
        """
        self.papers = papers
        self.section_detector = SectionDetector()
    
    def generate_queries(self, num_papers: int = None) -> List[Dict]:
        """
        Generate test queries (claims) from paper abstracts and conclusions.
        Generates 3-6 claims per paper covering:
        - Contribution
        - Method
        - Result
        - Optional limitation
        
        Args:
            num_papers: Number of papers to generate queries for. If None, uses all papers.
            
        Returns:
            List of query dictionaries with:
            - 'query': The claim/query text
            - 'paper_id': ID of the paper this query relates to
            - 'claim_type': Type of claim (contribution, method, result, limitation)
            - 'gold_sections': List of section names where evidence should be found
        """
        queries = []
        
        papers_to_use = self.papers if num_papers is None else random.sample(
            self.papers, min(num_papers, len(self.papers))
        )
        
        for paper in papers_to_use:
            paper_queries = self._generate_paper_queries(paper)
            queries.extend(paper_queries)
        
        return queries
    
    def _generate_paper_queries(self, paper: Dict) -> List[Dict]:
        """
        Generate queries for a single paper.
        
        Args:
            paper: Paper dictionary
            
        Returns:
            List of query dictionaries for this paper
        """
        queries = []
        abstract = paper.get('abstract', '')
        title = paper.get('title', '')
        
        if not abstract:
            return queries
        
        # Extract claims from abstract
        claims = self._extract_claims_from_abstract(abstract, title)
        
        # Determine gold sections for each claim type
        for claim_type, claim_text in claims:
            gold_sections = self._get_gold_sections_for_claim_type(claim_type)
            
            query_dict = {
                'query': claim_text,
                'paper_id': paper['id'],
                'title': title,
                'claim_type': claim_type,
                'gold_sections': gold_sections
            }
            queries.append(query_dict)
        
        return queries
    
    def _extract_claims_from_abstract(self, abstract: str, title: str) -> List[tuple]:
        """
        Extract claims from abstract text.
        Uses heuristics to identify contribution, method, result, and limitation statements.
        
        Args:
            abstract: Abstract text
            title: Paper title
            
        Returns:
            List of tuples: (claim_type, claim_text)
        """
        claims = []
        sentences = self._split_into_sentences(abstract)
        
        # Pattern matching for different claim types
        contribution_patterns = [
            r'(?:we|this paper|this work|we propose|we introduce|we present|we develop|we design)',
            r'(?:novel|new|first|original|innovative)',
            r'(?:contribution|contribute|propose|introduce|present|develop|design)'
        ]
        
        method_patterns = [
            r'(?:method|approach|algorithm|technique|framework|model|architecture)',
            r'(?:using|based on|leverage|employ|utilize)',
            r'(?:train|learn|optimize|compute|calculate)'
        ]
        
        result_patterns = [
            r'(?:achieve|obtain|demonstrate|show|exhibit|outperform|improve)',
            r'(?:accuracy|performance|result|evaluation|experiment)',
            r'(?:%|percent|improvement|better than|state-of-the-art|SOTA)'
        ]
        
        limitation_patterns = [
            r'(?:limitation|limit|constraint|challenge|difficulty|drawback)',
            r'(?:future work|future research|remain|open question)',
            r'(?:however|although|but|yet|while)'
        ]
        
        # Classify sentences
        contribution_sentences = []
        method_sentences = []
        result_sentences = []
        limitation_sentences = []
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            
            # Check for contribution
            if any(re.search(pattern, sentence_lower) for pattern in contribution_patterns):
                contribution_sentences.append(sentence)
            
            # Check for method
            if any(re.search(pattern, sentence_lower) for pattern in method_patterns):
                method_sentences.append(sentence)
            
            # Check for results
            if any(re.search(pattern, sentence_lower) for pattern in result_patterns):
                result_sentences.append(sentence)
            
            # Check for limitations
            if any(re.search(pattern, sentence_lower) for pattern in limitation_patterns):
                limitation_sentences.append(sentence)
        
        # Generate claims (prioritize contribution, method, result)
        if contribution_sentences:
            claims.append(('contribution', contribution_sentences[0]))
        
        if method_sentences:
            claims.append(('method', method_sentences[0]))
        
        if result_sentences:
            claims.append(('result', result_sentences[0]))
        
        # Add more method/result claims if available
        if len(method_sentences) > 1 and len(claims) < 6:
            claims.append(('method', method_sentences[1]))
        
        if len(result_sentences) > 1 and len(claims) < 6:
            claims.append(('result', result_sentences[1]))
        
        # Add limitation if available and we have space
        if limitation_sentences and len(claims) < 6:
            claims.append(('limitation', limitation_sentences[0]))
        
        # If no patterns matched, create generic claims from first few sentences
        if not claims and sentences:
            if len(sentences) >= 1:
                claims.append(('contribution', sentences[0]))
            if len(sentences) >= 2:
                claims.append(('method', sentences[1]))
            if len(sentences) >= 3:
                claims.append(('result', sentences[2]))
        
        # Ensure we have at least 3 claims
        while len(claims) < 3 and len(sentences) > len(claims):
            remaining_sentences = [s for s in sentences if s not in [c[1] for c in claims]]
            if remaining_sentences:
                claim_type = 'method' if len(claims) % 2 == 0 else 'result'
                claims.append((claim_type, remaining_sentences[0]))
        
        return claims[:6]  # Max 6 claims per paper
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting (can be enhanced with NLTK)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _get_gold_sections_for_claim_type(self, claim_type: str) -> List[str]:
        """
        Determine which sections should contain evidence for a claim type.
        
        Args:
            claim_type: Type of claim (contribution, method, result, limitation)
            
        Returns:
            List of section names where evidence should be found
        """
        section_mapping = {
            'contribution': ['Introduction', 'Abstract'],
            'method': ['Method', 'Methodology', 'Approach', 'Architecture', 'Implementation'],
            'result': ['Experiments', 'Results', 'Evaluation', 'Discussion'],
            'limitation': ['Discussion', 'Conclusion', 'Future Work']
        }
        
        return section_mapping.get(claim_type, ['Full Text'])

def load_test_queries(papers: List[Dict], num_papers: int = None) -> List[Dict]:
    """
    Convenience function to load test queries.
    
    Args:
        papers: List of paper dictionaries
        num_papers: Number of papers to generate queries for
        
    Returns:
        List of query dictionaries
    """
    generator = QueryGenerator(papers)
    return generator.generate_queries(num_papers)
