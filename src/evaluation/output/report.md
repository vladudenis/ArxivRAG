# ArxivRAG Evaluation Report

**Generated:** 2026-03-22 18:16:39 UTC

---

## Summary

| Metric | Value |
|--------|-------|
| Examples evaluated | 5 |
| Total records (examples × strategies) | 20 |
| Overall pass rate | 90.0% |
| Overall hallucination rate | 10.0% |

**Best strategy by composite score:** `STRUCTURE_AWARE_OVERLAP` (score: 4.940)

## Strategy Comparison

| Strategy | Composite | Groundedness | Correctness | Completeness | Citation | Pass | Halluc. | Latency (ms) |
|----------|-----------|--------------|-------------|---------------|----------|------|---------|---------------|
| STRUCTURE_AWARE_OVERLAP | 4.940 | 5.00 | 5.00 | 5.00 | 4.60 | 100% | 0% | 247175 |
| FIXED_WINDOW_OVERLAP | 4.700 | 4.40 | 5.00 | 5.00 | 4.40 | 80% | 20% | 247175 |
| SECTION_LEVEL_CHUNKING | 4.640 | 4.60 | 4.80 | 4.80 | 4.20 | 100% | 0% | 247175 |
| SEMANTIC_PARAGRAPH_GROUPING | 4.330 | 4.20 | 4.60 | 4.40 | 4.00 | 80% | 20% | 247175 |

## Additional Metrics by Strategy

| Strategy | Must-include hit | Recall@k | ROUGE-L | BLEU |
|----------|------------------|----------|---------|------|
| STRUCTURE_AWARE_OVERLAP | 30.00% | 0.00% | 0.068 | 0.45 |
| FIXED_WINDOW_OVERLAP | 30.00% | 0.00% | 0.063 | 0.64 |
| SECTION_LEVEL_CHUNKING | 30.00% | 0.00% | 0.070 | 0.59 |
| SEMANTIC_PARAGRAPH_GROUPING | 10.00% | 0.00% | 0.073 | 0.92 |

## Per-Strategy Details

### STRUCTURE_AWARE_OVERLAP

- **Composite score:** 4.9400
- **LLM judge:** correctness=5.00, groundedness=5.00, completeness=5.00, citation_quality=4.60
- **Pass rate:** 100.0% | **Hallucination rate:** 0.0%
- **Must-include hit rate:** 30.00%
- **Recall@k:** 0.00%
- **ROUGE-L F1:** 0.0678 | **BLEU:** 0.45
- **Avg latency:** 247175.42 ms

### FIXED_WINDOW_OVERLAP

- **Composite score:** 4.7000
- **LLM judge:** correctness=5.00, groundedness=4.40, completeness=5.00, citation_quality=4.40
- **Pass rate:** 80.0% | **Hallucination rate:** 20.0%
- **Must-include hit rate:** 30.00%
- **Recall@k:** 0.00%
- **ROUGE-L F1:** 0.0633 | **BLEU:** 0.64
- **Avg latency:** 247175.42 ms

### SECTION_LEVEL_CHUNKING

- **Composite score:** 4.6400
- **LLM judge:** correctness=4.80, groundedness=4.60, completeness=4.80, citation_quality=4.20
- **Pass rate:** 100.0% | **Hallucination rate:** 0.0%
- **Must-include hit rate:** 30.00%
- **Recall@k:** 0.00%
- **ROUGE-L F1:** 0.0699 | **BLEU:** 0.59
- **Avg latency:** 247175.42 ms

### SEMANTIC_PARAGRAPH_GROUPING

- **Composite score:** 4.3300
- **LLM judge:** correctness=4.60, groundedness=4.20, completeness=4.40, citation_quality=4.00
- **Pass rate:** 80.0% | **Hallucination rate:** 20.0%
- **Must-include hit rate:** 10.00%
- **Recall@k:** 0.00%
- **ROUGE-L F1:** 0.0734 | **BLEU:** 0.92
- **Avg latency:** 247175.42 ms
