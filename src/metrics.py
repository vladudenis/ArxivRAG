"""
Evaluation metrics for RAG system.
Implements ROUGE, BLEU, BERTScore, and Recall@k.
"""
import numpy as np
from typing import List, Dict, Union
from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU
from bert_score import score as bert_score_fn

class MetricsCalculator:
    """Calculate various evaluation metrics for RAG outputs."""
    
    def __init__(self):
        """Initialize metrics calculators."""
        # ROUGE scorer with multiple variants
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'], 
            use_stemmer=True
        )
        
        # BLEU scorer
        self.bleu_scorer = BLEU()
    
    def calculate_rouge(self, prediction: str, reference: str) -> Dict[str, float]:
        """
        Calculate ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L).
        
        Args:
            prediction: Generated text
            reference: Reference text
            
        Returns:
            Dictionary with rouge1, rouge2, rougeL F1 scores
        """
        scores = self.rouge_scorer.score(reference, prediction)
        
        return {
            'rouge1': scores['rouge1'].fmeasure,
            'rouge2': scores['rouge2'].fmeasure,
            'rougeL': scores['rougeL'].fmeasure
        }
    
    def calculate_bleu(self, prediction: str, reference: str) -> float:
        """
        Calculate BLEU score.
        
        Args:
            prediction: Generated text
            reference: Reference text
            
        Returns:
            BLEU score (0-100)
        """
        # sacrebleu expects list of references
        score = self.bleu_scorer.sentence_score(prediction, [reference])
        return score.score
    
    def calculate_bertscore(
        self, 
        predictions: List[str], 
        references: List[str],
        lang: str = 'en',
        model_type: str = None
    ) -> Dict[str, List[float]]:
        """
        Calculate BERTScore for semantic similarity.
        
        Args:
            predictions: List of generated texts
            references: List of reference texts
            lang: Language code
            model_type: Specific BERT model to use (None for default)
            
        Returns:
            Dictionary with precision, recall, f1 lists
        """
        # BERTScore can be slow, so we batch process
        P, R, F1 = bert_score_fn(
            predictions, 
            references, 
            lang=lang,
            model_type=model_type,
            verbose=False
        )
        
        return {
            'precision': P.tolist(),
            'recall': R.tolist(),
            'f1': F1.tolist()
        }
    
    def calculate_recall_at_k(
        self, 
        retrieved_ids: List[str], 
        relevant_ids: List[str], 
        k: int
    ) -> float:
        """
        Calculate Recall@k for retrieval.
        
        Args:
            retrieved_ids: List of retrieved document IDs (ordered by relevance)
            relevant_ids: List of relevant document IDs
            k: Number of top results to consider
            
        Returns:
            Recall@k score (0-1)
        """
        if not relevant_ids:
            return 0.0
        
        # Take top k retrieved documents
        top_k = retrieved_ids[:k]
        
        # Count how many relevant docs are in top k
        relevant_retrieved = len(set(top_k) & set(relevant_ids))
        
        # Recall = relevant retrieved / total relevant
        recall = relevant_retrieved / len(relevant_ids)
        
        return recall
    
    def calculate_all_metrics(
        self,
        prediction: str,
        reference: str,
        retrieved_ids: List[str],
        relevant_ids: List[str],
        k: int = 5
    ) -> Dict[str, float]:
        """
        Calculate all metrics for a single query.
        
        Args:
            prediction: Generated text
            reference: Reference text
            retrieved_ids: Retrieved document IDs
            relevant_ids: Relevant document IDs
            k: Top-k for recall calculation
            
        Returns:
            Dictionary with all metric scores
        """
        metrics = {}
        
        # ROUGE scores
        rouge_scores = self.calculate_rouge(prediction, reference)
        metrics.update(rouge_scores)
        
        # BLEU score
        metrics['bleu'] = self.calculate_bleu(prediction, reference)
        
        # Recall@k
        metrics[f'recall@{k}'] = self.calculate_recall_at_k(
            retrieved_ids, relevant_ids, k
        )
        
        return metrics

def aggregate_metrics(all_metrics: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """
    Aggregate metrics across multiple queries.
    
    Args:
        all_metrics: List of metric dictionaries from multiple queries
        
    Returns:
        Dictionary with mean, median, std for each metric
    """
    if not all_metrics:
        return {}
    
    # Get all metric names
    metric_names = all_metrics[0].keys()
    
    aggregated = {}
    for metric_name in metric_names:
        values = [m[metric_name] for m in all_metrics if metric_name in m]
        
        if values:
            aggregated[metric_name] = {
                'mean': np.mean(values),
                'median': np.median(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
    
    return aggregated
