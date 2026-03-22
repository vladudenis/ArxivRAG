# ArxivRAG - RAG for arXiv Papers

A Retrieval-Augmented Generation (RAG) system for querying arXiv papers. Enter topics for search and your question; the system fetches relevant papers, embeds them, and answers using RAG.

## Overview

- **Topics + query workflow**: Enter comma-separated topics for arXiv search and a natural language question for embedding/retrieval.
- **Pre-filtering**: arXiv keyword search (topics) → abstract similarity filter (query) → download and embed.
- **Multi-strategy chunking**: Each query returns answers from all four chunking strategies; the UI shows a paginated view (1/4, 2/4, …) to compare them.
- **Storage**: MinIO for PDFs, Qdrant for embeddings and paper metadata.

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
- Docker and Docker Compose

### 1. Install Dependencies

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Configure Environment

Create `.env` in the project root:

```env
HF_TOKEN=your_huggingface_token_here

# DeepSeek API (required for query answering)
LLM_BASE_URL=https://api.deepseek.com
LLM_API_KEY=your_deepseek_api_key_here
```

### 3. Start Infrastructure

```powershell
docker-compose up -d
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

**API**:

- `POST /query` — Request body: `{"query": "...", "topics": "..."}`. Both fields required. Optional: `include_debug_context: true` to return retrieved chunks per strategy (for evaluation). Topics: comma-separated terms for arXiv search. Query: natural language question for embedding and retrieval. Returns `{"results": [{"strategy", "strategy_label", "answer", "sources"}, ...]}` with one entry per chunking strategy.
- `GET /health` — Health check.

**Features**:

- Topics field: type terms and press comma to lock each term
- Web interface shows cited sources (paper title, arXiv link) when the LLM uses them
- Paginated strategy view: navigate between answers from each chunking strategy (page 1/4, 2/4, …) to compare results

## Evaluation (LLM-as-Judge)

The `src/evaluation` folder contains an LLM-as-judge pipeline for benchmarking RAG performance across chunking strategies.

**What it does**:

- Loads a JSONL dataset and calls `POST /query` with `include_debug_context=true`
- Scores each strategy with LLM judge metrics (correctness, groundedness, completeness, citation_quality, hallucination, verdict)
- Computes deterministic metrics (must_include_hit_rate, recall_at_k, citation_count) and text overlap (ROUGE-L, BLEU) when reference answers exist
- Writes `results.jsonl`, `summary.json`, and `report.md` to `src/evaluation/output`

**Dataset format** (one JSON object per line in `src/evaluation/dataset.template.jsonl`):

```json
{"id":"ex_001","topics":"rag, hallucination","question":"How does RAG reduce hallucinations?","reference_answer":"...","must_include":["retrieving evidence"],"relevant_papers":["1706.03762"],"metadata":{"difficulty":"easy"}}
```

**Run evaluation**:

1. Start the API (see Usage above)
2. Run: `python -m src.evaluation.run_eval --dataset src/evaluation/dataset.template.jsonl --base-url http://localhost:8000 --output-dir src/evaluation/output`
