# ArxivRAG - RAG for arXiv Papers

A Retrieval-Augmented Generation (RAG) system for querying arXiv papers. Enter topics for search and your question; the system fetches relevant papers, embeds them, and answers using RAG.

## Overview

- **Topics + query workflow**: Enter comma-separated topics for arXiv search and a natural language question for embedding/retrieval.
- **Pre-filtering**: arXiv keyword search (topics, max 20 results) → abstract similarity filter (query, top 6 papers) → download and embed.
- **Multi-strategy chunking**: Each query returns answers from all four chunking strategies; the UI shows a paginated view (1/4, 2/4, …) to compare them.
- **Storage**: MinIO for PDFs, Qdrant for embeddings and paper metadata.

## Folder Structure

- `src/` - Core Python backend logic for retrieval, chunking, embedding, storage, and pipeline orchestration.
- `src/api/` - FastAPI application (`/query`, `/health`) and API-layer services/clients.
- `src/evaluation/` - Retrieval-only benchmarking pipeline, dataset template, corpus freezing, manifests, metrics, and outputs.
- `frontend/` - Next.js web UI (chat interface, topics input, source display, and API integration).
- `docker-compose.yml` - Local infrastructure setup for MinIO and Qdrant.
- `requirements.txt` - Python dependencies for the backend and evaluation pipeline.

## Chunking Strategies

1. **STRUCTURE_AWARE_OVERLAP** (default, recommended)
   - Preserves academic structure (Abstract, sections, subsections)
   - Target 500–800 tokens, max 900; 10–15% overlap
   - Abstract and Conclusion as standalone chunks; References excluded

2. **SEMANTIC_PARAGRAPH_GROUPING**
   - Groups paragraphs by embedding similarity (threshold ~0.75)
   - Min 300, max 900 tokens; topic-coherent chunks

3. **FIXED_WINDOW_OVERLAP** (baseline)
   - Sliding window: 700 tokens, 150 overlap
   - Simple and fast

4. **SECTION_LEVEL_CHUNKING**
   - One chunk per subsection; split at midpoint if >1500 tokens
   - Maximum semantic integrity

## Setup

### Prerequisites

- Python 3.8+
- Node.js 20+ (for the Next.js frontend)
- Docker and Docker Compose

### 1. Install Dependencies

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Install Frontend Dependencies

```powershell
cd frontend
npm install
cd ..
```

### 3. Copy Environment Template

```powershell
copy .env.example .env
```

### 4. Configure Environment

Create `.env` in the project root:

```env
HF_TOKEN=your_huggingface_token_here

# DeepSeek API (required for query answering)
LLM_BASE_URL=https://api.deepseek.com
LLM_API_KEY=your_deepseek_api_key_here
```

Optional (only if your API is not running on `http://localhost:8000`):

Create `frontend/.env.local`:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### 5. Start Infrastructure

```powershell
docker compose up -d
```

This starts:

- **MinIO** (ports 9000, 9001): Stores PDF files
- **Qdrant** (port 6333): Vector database for embeddings and paper metadata

## Usage

### Web interface (FastAPI + Next.js)

1. **Start the API** (from project root):

```powershell
uvicorn src.api.main:app --reload --port 8000
```

2. **Start the frontend**:

```powershell
cd frontend
npm run dev
```

3. Open http://localhost:3000 for a ChatGPT-like chat interface.

Note: each `/query` request rebuilds the temporary index from the newly fetched papers (across all chunking strategies), so data is intentionally not persisted between queries.

**API**:

- `POST /query` — Request body: `{"query": "...", "topics": "..."}`. Both fields required. Optional: `include_debug_context: true` to return retrieved chunks per strategy (for evaluation). Topics: comma-separated terms for arXiv search. Query: natural language question for embedding and retrieval. Returns `{"results": [{"strategy", "strategy_label", "answer", "sources"}, ...]}` with one entry per chunking strategy.
- `GET /health` — Health check.

**Features**:

- Topics field: type terms and press comma to lock each term
- Web interface shows cited sources (paper title, arXiv link) when the LLM uses them
- Paginated strategy view: navigate between answers from each chunking strategy (page 1/4, 2/4, …) to compare results

### Evaluation (retrieval-only)

Search and abstract-filter sizes are centralized in [`src/rag_constants.py`](src/rag_constants.py) (`ARXIV_SEARCH_MAX_RESULTS=20`, `ABSTRACT_FILTER_TOP_K=6`). Chunk retrieval defaults to `RETRIEVAL_CHUNK_TOP_K=8` with dense or hybrid (dense + BM25) retrieval and optional cross-encoder re-ranking.

**1. Freeze a corpus** (full reset of snapshot + arXiv search/filter + PDF download to MinIO bucket `eval-frozen-corpus`):

```powershell
docker compose up -d
python -m src.evaluation.freeze_corpus --dataset src/evaluation/dataset.template.jsonl --snapshot-id v1
```

**2. Run retrieval evaluation** (indexes frozen PDFs, uses frozen paper IDs as gold docs, compares retrieval against dataset `gold_passages`):

```powershell
python -m src.evaluation.run_retrieval_eval --dataset src/evaluation/dataset.template.jsonl --snapshot-id v1 --phase all --output-dir src/evaluation/output
```

Output: `src/evaluation/output/retrieval_eval.json`. Each configuration reports five metrics — `hit_at_k` (Success@k: ≥1 relevant chunk in top-k), `paper_coverage` (paper-level coverage = |retrieved papers ∩ gold| / |gold|), `precision_at_k`, `mrr`, and `average_precision` (aggregates to MAP) — plus `metrics_ci` (bootstrap 95% confidence intervals) and `per_query` scores (aligned to top-level `example_ids`). A `significance` block reports paired-bootstrap and Wilcoxon tests for pre-registered comparisons.

Note on metrics: `hit_at_k` corresponds to the metric the practical report labelled "Recall@k". `paper_coverage` is a stricter paper-level recall that is structurally capped at `min(k, |gold|)/|gold|` (so it is best compared at `k ≥ |gold|`); `hit_at_k` and `average_precision` are the recommended headline metrics for cross-configuration comparison.

**Graded relevance:** a chunk's grade is `1.0` if its paper ID is a gold document, otherwise its max cosine similarity to any gold passage (clipped to `[0,1]`). Binary relevance (used by `hit_at_k`, `precision_at_k`, `mrr`, `average_precision`) thresholds this grade at `PASSAGE_RELEVANCE_THRESHOLD`. Relevance is scored by a fixed **judge embedder** (`--judge-embedding-model`, default `google/embeddinggemma-300m`) so labels stay constant when the retrieval embedding model varies.

**3. Compare embedding models (Phase 3, optional):** add `--embedding-models` to run a per-model sweep that re-indexes the frozen corpus with the Phase 1 winning chunker and the best Phase 2 retrieval config, scoring all models against the fixed judge embedder. Results are written under the `phase3` key.

If you have **already** run Phases 1–2 (i.e. `retrieval_eval.json` exists), use `--phase 3` to run **only** the embedding comparison and merge it into the existing file — this skips re-computing Phases 1–2 (hours of work) and preserves their results. `--phase 3` requires `--winner-strategy` (and reuses the best Phase 2 config from the existing file, falling back to dense/top-k=10/no-rerank):

```powershell
python -m src.evaluation.run_retrieval_eval --dataset src/evaluation/dataset.template.jsonl --snapshot-id v1 --phase 3 --winner-strategy FIXED_WINDOW_OVERLAP --embedding-models "google/embeddinggemma-300m,BAAI/bge-small-en-v1.5,BAAI/bge-base-en-v1.5" --output-dir src/evaluation/output
```

To run everything in one pass instead (recomputes Phases 1–2), use `--phase all` with `--embedding-models`. Either way, each model triggers a full re-embed + re-index of the frozen corpus, so Phase 3 runtime scales with the number of models.

**4. Generate LaTeX tables:** turn a results file into Overleaf-ready booktabs tables (Phase 1/2/3 + significance):

```powershell
python -m src.evaluation.make_tables --input src/evaluation/output/retrieval_eval.json --output src/evaluation/output/thesis_tables.tex
```

**Dataset format** (JSONL): `id` (optional), `topics`, `question`, `gold_passages` (optional, used by retrieval evaluation), `metadata` (optional). `gold_docs` from the dataset are ignored; gold docs come from the frozen snapshot mapping (`topics` + `question` → paper IDs).
