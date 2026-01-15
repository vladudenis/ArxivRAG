# ArxivRAG - Chunking Evaluation Pipeline

A robust chunking evaluation pipeline for Retrieval-Augmented Generation (RAG) systems operating on arXiv papers. This system identifies the best-performing chunking strategy based strictly on retrieval quality metrics, then validates it qualitatively with an inference model.

## Overview

The pipeline is designed to be **general-purpose and paper-agnostic** (not limited to ML or math papers), although initial experiments run on arXiv machine learning papers that may contain LaTeX math and symbols. Math and symbols are preserved as raw text; no OCR is required.

### Key Principles

- **Fixed Components**: Only the chunking strategy varies. All other components remain fixed:

  - Embedding model
  - Vector database
  - Similarity metric
  - Retrieval parameters (top-k)
  - Query generation logic

- **Three Strictly Separated Stages**:

  1. **Stage 1**: Document-level retrieval evaluation (paper identity)
  2. **Stage 2**: Chunk-level evidence retrieval evaluation (section-based gold evidence)
  3. **Stage 3**: Inference-based qualitative validation (only for best strategy)

## Features

### Five Chunking Strategies

1. **Fixed-Length Token Chunks** (`fixed_token`)

   - ~512 tokens with 10-20% overlap (~64 tokens)
   - Baseline strategy for comparison

2. **Section-Based Chunking** (`section`)

   - One chunk per logical section (Introduction, Method, Experiments, etc.)
   - Preserves document structure

3. **Paragraph-Based Chunking** (`paragraph`)

   - One chunk per paragraph
   - Merges very short paragraphs if needed

4. **Sentence-Aware Sliding Window** (`sentence_sliding`)

   - Fixed size chunks (~512 characters)
   - Sentence-boundary-aware overlap (~64 characters)
   - Respects sentence boundaries

5. **Section + Size-Capped Hybrid** (`section_hybrid`)
   - Split by section first
   - Subdivide sections exceeding max token length (512 tokens)

### Chunk Metadata

Each chunk stores:

- `paper_id`: Unique paper identifier
- `section_id`: Section name (e.g., "Introduction", "Method")
- `chunk_id`: Unique chunk identifier
- `chunk_text`: The chunk content

## Setup

### Prerequisites

- Python 3.8+
- Docker and Docker Compose
- NVIDIA GPU (optional, for vLLM inference)

### 1. Install Dependencies

```powershell
# Create virtual environment (if not exists)
python -m venv .venv

# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt
```

### 2. Configure Environment

Create/update `.env` file in the project root:

```env
HF_TOKEN=your_huggingface_token_here
VLLM_BASE_URL=http://localhost:8000/v1
VLLM_MODEL=TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

**Note**: `HF_TOKEN` is required for downloading embedding models from HuggingFace. `VLLM_*` variables are optional and only needed for Stage 3 inference validation.

### 3. Start Infrastructure Services

Start all required services using Docker Compose:

```powershell
# Start all services in background
docker-compose up -d

# Check status
docker-compose ps

# View logs (useful for debugging)
docker-compose logs -f
```

This starts:

- **DynamoDB Local** (port 8001): Stores paper metadata
- **MinIO** (ports 9000, 9001): Stores PDF files
- **Qdrant** (port 6333): Vector database for embeddings
- **vLLM** (port 8000): LLM inference server (optional, for Stage 3)

**Note**: On first run, vLLM will download the model, which may take several minutes.

## Usage

### Running the Evaluation

```powershell
# Full pipeline: Download papers and run all stages
python experiment_runner.py --skip-ingestion false

# Full pipeline: Reuse existing papers and run all stages
python experiment_runner.py --skip-ingestion true

# Run a single stage independently
python experiment_runner.py --skip-ingestion true --stage 1  # Stage 1 only
python experiment_runner.py --skip-ingestion true --stage 2  # Stage 2 only
python experiment_runner.py --skip-ingestion true --stage 3  # Stage 3 only
```

**What it does**:

1. Downloads 100 papers from arXiv (if `--skip-ingestion false`)
2. Generates 3-6 queries per paper from abstracts
3. Evaluates all 5 chunking strategies using Stages 1 & 2
4. Selects best strategy based on retrieval metrics
5. Runs Stage 3 qualitative validation (if vLLM available)

### Command-Line Options

- `--skip-ingestion {true|false}`:

  - `true`: Reuse existing papers in storage
  - `false`: Download and process new papers

- `--stage {1|2|3}` (optional):
  - Run a single stage in isolation
  - `1`: Run Stage 1 only (document-level retrieval evaluation)
  - `2`: Run Stage 2 only (chunk-level evidence retrieval evaluation)
  - `3`: Run Stage 3 only (inference-based qualitative validation)
  - If not specified, runs all stages in sequence
  - **Note**: Stage 3 uses section-based chunking by default if no best strategy is found

## Pipeline Architecture

### Stage 1: Document-Level Retrieval Evaluation

**Purpose**: Determine whether queries retrieve chunks from the correct paper(s).

**Process**:

1. Chunk papers using each strategy
2. Embed chunks
3. Build vector index
4. For each query, retrieve top-k chunks
5. Check if at least one retrieved chunk belongs to the correct paper

**Metrics**:

- **Paper Recall@k**: Percentage of queries with correct paper in top-k
- **Paper MRR**: Mean Reciprocal Rank for paper retrieval
- **Queries with Correct Paper**: Percentage of queries with ≥1 correct paper

**Note**: Only paper identity matters at this stage; chunk content is ignored.

### Stage 2: Chunk-Level Evidence Retrieval Evaluation

**Purpose**: Determine whether retrieved chunks contain the correct evidence.

**Query Generation**:

- Automatically generates 3-6 natural-language claims per paper from abstracts
- Claims cover: contribution, method, result, optional limitation
- Each claim corresponds to one query

**Gold Evidence Definition**:

- For each query, defines a set of gold sections (e.g., Method, Experiments)
- Uses **content-based section matching** to determine if a chunk contains relevant evidence
- A retrieved chunk is considered relevant if it overlaps with gold sections based on content analysis
- This ensures fair evaluation across all chunking strategies (not biased toward section-based chunking)
- Sentence-level annotation is not required

**Metrics**:

- **Recall@k** (chunk-level): Percentage of relevant chunks retrieved
- **MRR**: Mean Reciprocal Rank
- **Precision@k**: Precision of retrieved chunks

### Stage 3: Inference-Based Qualitative Validation

**Purpose**: Qualitatively validate the best-performing chunking strategy.

**Process**:

1. Attach fixed inference model (vLLM)
2. Run full RAG on a small subset of papers (max 10)
3. Generate responses using retrieved chunks
4. Manually inspect outputs for:
   - Faithfulness
   - Completeness
   - Citation correctness

**Important**:

- Stage 3 is **ONLY** run for the best strategy selected from Stages 1 & 2 (when running full pipeline)
- When running Stage 3 independently (`--stage 3`), it uses section-based chunking by default if no best strategy is found
- Uses top-3 chunks (reduced from top-5) to fit within TinyLlama's 2048 token context limit
- Inference results **MUST NOT** be used to rank chunking strategies
- This stage remains strictly separate from retrieval evaluation

## Reproducibility

The system ensures reproducibility through:

- Fixed random seed (42) for paper selection
- Deterministic chunking strategies
- Fixed embedding model and parameters
- Consistent query generation logic

## Constraints

- **No model training/fine-tuning**: Uses pre-trained models only
- **Re-embed on chunking change**: Embeddings are regenerated when chunking strategy changes
- **Fixed queries**: Same queries used across all chunking strategies
- **Math preservation**: LaTeX and symbols preserved as raw text
- **Deterministic**: Experiments are reproducible
