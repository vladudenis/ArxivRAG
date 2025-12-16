"""
Results analysis and reporting for RAG experiments.
"""
import json
from typing import List, Dict
from pathlib import Path

class ResultsAnalyzer:
    """Analyze and report on RAG experiment results."""
    
    def __init__(self, results: List[Dict]):
        """
        Initialize with experiment results.
        
        Args:
            results: List of result dictionaries from experiments
        """
        self.results = results
    
    def generate_markdown_report(self, output_path: str = "experiment_results.md"):
        """
        Generate comprehensive markdown report.
        
        Args:
            output_path: Path to save markdown file
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# RAG Evaluation Results\n\n")
            f.write("Comparison of different chunking strategies for Retrieval-Augmented Generation.\n\n")
            
            # Summary table
            f.write("## Summary Metrics\n\n")
            f.write(self._generate_summary_table())
            f.write("\n\n")
            
            # Detailed metrics for each strategy
            f.write("## Detailed Results by Strategy\n\n")
            for result in self.results:
                f.write(self._generate_strategy_section(result))
                f.write("\n")
            
            # Best performing strategies
            f.write("## Best Performing Strategies\n\n")
            f.write(self._generate_best_strategies())
            f.write("\n")
            
            # Insights
            f.write("## Key Insights\n\n")
            f.write(self._generate_insights())
        
        print(f"Report saved to {output_path}")
    
    def _generate_summary_table(self) -> str:
        """Generate summary comparison table."""
        lines = []
        
        # Header
        lines.append("| Strategy | Chunks | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU | BERTScore F1 | Recall@5 |")
        lines.append("|----------|--------|---------|---------|---------|------|--------------|----------|")
        
        # Rows
        for result in self.results:
            strategy_name = result.get('strategy', 'Unknown')
            num_chunks = result.get('num_chunks', 0)
            metrics = result.get('metrics', {})
            
            rouge1 = metrics.get('rouge1', {}).get('mean', 0)
            rouge2 = metrics.get('rouge2', {}).get('mean', 0)
            rougeL = metrics.get('rougeL', {}).get('mean', 0)
            bleu = metrics.get('bleu', {}).get('mean', 0)
            bert_f1 = metrics.get('bertscore_f1', {}).get('mean', 0)
            recall5 = metrics.get('recall@5', {}).get('mean', 0)
            
            lines.append(
                f"| {strategy_name} | {num_chunks} | "
                f"{rouge1:.4f} | {rouge2:.4f} | {rougeL:.4f} | "
                f"{bleu:.2f} | {bert_f1:.4f} | {recall5:.4f} |"
            )
        
        return "\n".join(lines)
    
    def _generate_strategy_section(self, result: Dict) -> str:
        """Generate detailed section for a strategy."""
        lines = []
        
        strategy = result.get('strategy', 'Unknown')
        chunk_size = result.get('chunk_size', 0)
        chunk_overlap = result.get('chunk_overlap', 0)
        num_chunks = result.get('num_chunks', 0)
        num_queries = result.get('num_queries', 0)
        metrics = result.get('metrics', {})
        
        lines.append(f"### {strategy}")
        lines.append(f"\n**Configuration:**")
        lines.append(f"- Chunk Size: {chunk_size}")
        lines.append(f"- Chunk Overlap: {chunk_overlap}")
        lines.append(f"- Total Chunks: {num_chunks}")
        lines.append(f"- Queries Evaluated: {num_queries}")
        
        lines.append(f"\n**Metrics:**")
        
        for metric_name, stats in metrics.items():
            if isinstance(stats, dict):
                mean = stats.get('mean', 0)
                std = stats.get('std', 0)
                lines.append(f"- **{metric_name}**: {mean:.4f} (±{std:.4f})")
        
        return "\n".join(lines)
    
    def _generate_best_strategies(self) -> str:
        """Identify best strategies for each metric."""
        lines = []
        
        metric_names = ['rouge1', 'rouge2', 'rougeL', 'bleu', 'recall@5']
        
        for metric_name in metric_names:
            best_result = None
            best_score = -1
            
            for result in self.results:
                metrics = result.get('metrics', {})
                if metric_name in metrics:
                    score = metrics[metric_name].get('mean', 0)
                    if score > best_score:
                        best_score = score
                        best_result = result
            
            if best_result:
                strategy = best_result.get('strategy', 'Unknown')
                lines.append(f"- **{metric_name}**: {strategy} ({best_score:.4f})")
        
        return "\n".join(lines)
    
    def _generate_insights(self) -> str:
        """Generate insights from results."""
        lines = []
        
        # Find strategy with most chunks
        max_chunks_result = max(self.results, key=lambda r: r.get('num_chunks', 0))
        min_chunks_result = min(self.results, key=lambda r: r.get('num_chunks', 0))
        
        lines.append(f"- **Chunk Count Range**: {min_chunks_result.get('strategy')} generated "
                    f"{min_chunks_result.get('num_chunks')} chunks, while "
                    f"{max_chunks_result.get('strategy')} generated {max_chunks_result.get('num_chunks')} chunks.")
        
        # Find overall best strategy (by average of normalized metrics)
        best_overall = self._find_best_overall_strategy()
        if best_overall:
            lines.append(f"- **Best Overall Strategy**: {best_overall} (based on average performance across all metrics)")
        
        lines.append(f"- **Evaluation Coverage**: Tested {len(self.results)} different chunking strategies")
        
        return "\n".join(lines)
    
    def _find_best_overall_strategy(self) -> str:
        """Find best overall strategy by averaging normalized metrics."""
        if not self.results:
            return None
        
        # Collect all metric values for normalization
        metric_names = ['rouge1', 'rouge2', 'rougeL', 'bleu', 'recall@5']
        metric_ranges = {name: {'min': float('inf'), 'max': float('-inf')} for name in metric_names}
        
        # Find min/max for each metric
        for result in self.results:
            metrics = result.get('metrics', {})
            for metric_name in metric_names:
                if metric_name in metrics:
                    value = metrics[metric_name].get('mean', 0)
                    metric_ranges[metric_name]['min'] = min(metric_ranges[metric_name]['min'], value)
                    metric_ranges[metric_name]['max'] = max(metric_ranges[metric_name]['max'], value)
        
        # Calculate normalized average for each strategy
        best_strategy = None
        best_score = -1
        
        for result in self.results:
            metrics = result.get('metrics', {})
            normalized_scores = []
            
            for metric_name in metric_names:
                if metric_name in metrics:
                    value = metrics[metric_name].get('mean', 0)
                    min_val = metric_ranges[metric_name]['min']
                    max_val = metric_ranges[metric_name]['max']
                    
                    # Normalize to [0, 1]
                    if max_val > min_val:
                        normalized = (value - min_val) / (max_val - min_val)
                    else:
                        normalized = 1.0
                    
                    normalized_scores.append(normalized)
            
            if normalized_scores:
                avg_score = sum(normalized_scores) / len(normalized_scores)
                if avg_score > best_score:
                    best_score = avg_score
                    best_strategy = result.get('strategy')
        
        return best_strategy
    
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            import numpy as np
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(ResultsAnalyzer.NumpyEncoder, self).default(obj)

    def save_json(self, output_path: str = "experiment_results.json"):
        """Save results as JSON."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, cls=self.NumpyEncoder)
        print(f"JSON results saved to {output_path}")
