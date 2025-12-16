# RAG Evaluation Results

Comparison of different chunking strategies for Retrieval-Augmented Generation.

## Summary Metrics

| Strategy | Chunks | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU | BERTScore F1 | Recall@5 |
|----------|--------|---------|---------|---------|------|--------------|----------|
| fixed | 14067 | 0.6947 | 0.5596 | 0.5812 | 43.13 | 0.0000 | 1.0000 |
| recursive | 14614 | 0.6968 | 0.5377 | 0.5953 | 42.41 | 0.0000 | 1.0000 |
| token | 7777 | 0.0000 | 0.0000 | 0.0000 | 0.00 | 0.0000 | 1.0000 |
| sentence | 15360 | 0.6207 | 0.4330 | 0.4690 | 30.89 | 0.0000 | 1.0000 |
| paragraph | 113 | 0.0000 | 0.0000 | 0.0000 | 0.00 | 0.0000 | 1.0000 |

## Detailed Results by Strategy

### fixed

**Configuration:**
- Chunk Size: 500
- Chunk Overlap: 50
- Total Chunks: 14067
- Queries Evaluated: 20

**Metrics:**
- **rouge1**: 0.6947 (±0.1447)
- **rouge2**: 0.5596 (±0.2231)
- **rougeL**: 0.5812 (±0.2263)
- **bleu**: 43.1338 (±25.5068)
- **recall@5**: 1.0000 (±0.0000)
### recursive

**Configuration:**
- Chunk Size: 500
- Chunk Overlap: 50
- Total Chunks: 14614
- Queries Evaluated: 20

**Metrics:**
- **rouge1**: 0.6968 (±0.1541)
- **rouge2**: 0.5377 (±0.2202)
- **rougeL**: 0.5953 (±0.2117)
- **bleu**: 42.4121 (±24.2462)
- **recall@5**: 1.0000 (±0.0000)
### token

**Configuration:**
- Chunk Size: 256
- Chunk Overlap: 32
- Total Chunks: 7777
- Queries Evaluated: 20

**Metrics:**
- **rouge1**: 0.0000 (±0.0000)
- **rouge2**: 0.0000 (±0.0000)
- **rougeL**: 0.0000 (±0.0000)
- **bleu**: 0.0000 (±0.0000)
- **recall@5**: 1.0000 (±0.0000)
### sentence

**Configuration:**
- Chunk Size: 500
- Chunk Overlap: 100
- Total Chunks: 15360
- Queries Evaluated: 20

**Metrics:**
- **rouge1**: 0.6207 (±0.0796)
- **rouge2**: 0.4330 (±0.1146)
- **rougeL**: 0.4690 (±0.1170)
- **bleu**: 30.8860 (±13.6027)
- **recall@5**: 1.0000 (±0.0000)
### paragraph

**Configuration:**
- Chunk Size: 0
- Chunk Overlap: 100
- Total Chunks: 113
- Queries Evaluated: 20

**Metrics:**
- **rouge1**: 0.0000 (±0.0000)
- **rouge2**: 0.0000 (±0.0000)
- **rougeL**: 0.0000 (±0.0000)
- **bleu**: 0.0000 (±0.0000)
- **recall@5**: 1.0000 (±0.0000)
## Best Performing Strategies

- **rouge1**: recursive (0.6968)
- **rouge2**: fixed (0.5596)
- **rougeL**: recursive (0.5953)
- **bleu**: fixed (43.1338)
- **recall@5**: fixed (1.0000)
## Key Insights

- **Chunk Count Range**: paragraph generated 113 chunks, while sentence generated 15360 chunks.
- **Best Overall Strategy**: fixed (based on average performance across all metrics)
- **Evaluation Coverage**: Tested 5 different chunking strategies