# Stage 1: Document-Level Retrieval Evaluation Results

This stage evaluates whether queries retrieve chunks from the correct paper(s).

## Phase 1: Fixed Corpus Benchmark

| Strategy | Paper Recall@5 | Paper MRR | Queries with Correct Paper |
|----------|----------------|-----------|----------------------------|
| fixed_token | 0.9670 | 0.9372 | 0.9670 |
| section | 0.9579 | 0.9250 | 0.9579 |
| paragraph | 0.9780 | 0.9676 | 0.9780 |
| sentence_sliding | 0.9835 | 0.9612 | 0.9835 |
| section_hybrid | 0.9579 | 0.9397 | 0.9579 |
