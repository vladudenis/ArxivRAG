"""
FastAPI backend for ArxivRAG.
POST /query - run RAG pipeline and return answer + cited sources.
"""
import asyncio
from contextlib import asynccontextmanager
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

from src.api.rag_service import RAGService
from src.api.rag_storage import RAGStorage, clear_all_data
from src.embedder import PaperEmbedder
from src.llm_client import DeepSeekClient
from src.arxiv_retriever import ArxivRetriever


# Global RAG service (initialized on startup)
rag_service: RAGService | None = None


class QueryRequest(BaseModel):
    query: str
    topics: str
    include_debug_context: bool = False


class SourceResponse(BaseModel):
    paper_id: str
    title: str
    section: str
    arxiv_url: str


class StrategyResultResponse(BaseModel):
    strategy: str
    strategy_label: str
    answer: str
    sources: list[SourceResponse]
    debug_context: list[dict[str, Any]] | None = None


class QueryResponse(BaseModel):
    results: list[StrategyResultResponse]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize RAG components on startup."""
    global rag_service

    storage = RAGStorage()
    storage.init_db()
    storage.init_bucket()

    embedder = PaperEmbedder(model_name="google/embeddinggemma-300m")
    test_emb = embedder.embed_texts(["test"])
    embedding_dim = test_emb.shape[1]
    storage.init_qdrant(vector_size=embedding_dim)

    retriever = ArxivRetriever()

    try:
        llm_client = DeepSeekClient()
    except Exception as e:
        print(f"WARNING: DeepSeek client init failed: {e}")
        llm_client = None

    if llm_client:
        rag_service = RAGService(
            storage=storage,
            embedder=embedder,
            retriever=retriever,
            llm_client=llm_client,
            embedding_dim=embedding_dim,
        )
    else:
        rag_service = None

    yield

    # Cleanup on shutdown
    if rag_service:
        clear_all_data(storage, vector_size=embedding_dim)


app = FastAPI(
    title="ArxivRAG API",
    description="Query arXiv papers via RAG",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    """Run RAG pipeline for the given query."""
    if not rag_service:
        raise HTTPException(
            status_code=503,
            detail="RAG service not available. Check LLM_BASE_URL and LLM_API_KEY.",
        )

    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    if not request.topics or not request.topics.strip():
        raise HTTPException(status_code=400, detail="Topics cannot be empty.")

    # Run blocking RAG pipeline in thread pool to avoid freezing the event loop
    result: dict[str, Any] = await asyncio.to_thread(
        rag_service.query,
        request.query.strip(),
        request.topics.strip(),
        request.include_debug_context,
    )

    results = [
        StrategyResultResponse(
            strategy=r["strategy"],
            strategy_label=r["strategy_label"],
            answer=r["answer"],
            sources=[
                SourceResponse(
                    paper_id=s["paper_id"],
                    title=s["title"],
                    section=s.get("section", ""),
                    arxiv_url=s["arxiv_url"],
                )
                for s in r.get("sources", [])
            ],
            debug_context=r.get("debug_context"),
        )
        for r in result.get("results", [])
    ]

    return QueryResponse(results=results)


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check."""
    return {"status": "ok"}
