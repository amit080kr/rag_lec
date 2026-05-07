import logging
from typing import List, Dict, Any
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from document_manager import DocumentManager
from document_processor import DocumentProcessor
from vector_engine import VectorEngine
from hybrid_retriever import HybridRetriever

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Global Services ---
COLLECTION_NAME = "documents_hybrid_search"
vector_engine: VectorEngine = None
retriever: HybridRetriever = None
processor: DocumentProcessor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager runs before the server starts and after it shuts down.
    Each uvicorn worker will initialize its own local embedding models and Qdrant clients.
    """
    global vector_engine, retriever, processor
    logger.info("Initializing background services (Models & Qdrant Clients)...")
    
    vector_engine = VectorEngine(collection_name=COLLECTION_NAME)
    retriever = HybridRetriever(collection_name=COLLECTION_NAME)
    processor = DocumentProcessor()
    
    # Quick health check
    if not vector_engine.health_check():
        logger.error("CRITICAL: Failed to connect to Qdrant during startup.")
        
    yield # Application runs here
    
    logger.info("Shutting down RAG API worker gracefully...")

# Initialize FastAPI App
app = FastAPI(
    title="RAG Pipeline API",
    description="Production API for Incremental Ingestion and Hybrid RAG Search",
    version="1.0.0",
    lifespan=lifespan
)

# 3. Middleware for CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, specify exact domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Global Exception Handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catches all unhandled exceptions globally to prevent server crashes and return standard JSON."""
    logger.error(f"Unhandled exception on {request.method} {request.url}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"message": "Internal Server Error", "details": str(exc)},
    )

# --- API Models ---
class IngestRequest(BaseModel):
    directory_path: str
    
class QueryRequest(BaseModel):
    query: str
    hybrid_top_k: int = 20
    rerank_top_k: int = 5
    confidence_threshold: float = 0.4

# --- Endpoints ---

@app.get("/")
async def root():
    """Welcome endpoint for simple browser verification."""
    return {
        "status": "online",
        "message": "RAG Pipeline API is running. Access the API documentation at /docs",
        "endpoints": ["POST /ingest", "POST /query"]
    }

@app.get("/health")
async def health_check():
    """Returns the health status of the API and its dependent services."""
    qdrant_status = vector_engine.health_check() if vector_engine else False
    return {
        "status": "healthy" if qdrant_status else "degraded",
        "qdrant_connected": qdrant_status
    }

@app.post("/ingest")
async def ingest_documents(request: IngestRequest):
    """
    1. Scans the local directory using DocumentManager.
    2. Chunks new/modified files using DocumentProcessor.
    3. Upserts chunks in batches to Qdrant using VectorEngine.
    """
    try:
        manager = DocumentManager(directory=request.directory_path)
        files_to_process = manager.get_files_to_process()
        
        if not files_to_process:
            return {"message": "No new or modified files found.", "files_processed": 0}
            
        total_chunks = 0
        for file_path in files_to_process:
            logger.info(f"Ingesting file: {file_path}")
            chunks = processor.process_file(file_path)
            if chunks:
                vector_engine.upsert_documents(chunks, batch_size=100)
                total_chunks += len(chunks)
                
        return {
            "message": "Ingestion successful", 
            "files_processed": len(files_to_process),
            "total_chunks_upserted": total_chunks
        }
    except ValueError as ve:
        # Expected errors like invalid directory
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")

@app.post("/query")
async def query_documents(request: QueryRequest):
    """
    2. Executes a Hybrid Search (Dense + Sparse BM25 + RRF) and re-ranks with FlashRank.
    Returns the top reranked document chunks.
    """
    try:
        results = retriever.search(
            query=request.query, 
            hybrid_top_k=request.hybrid_top_k, 
            rerank_top_k=request.rerank_top_k,
            confidence_threshold=request.confidence_threshold
        )
        return {"query": request.query, "results": results}
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")

# 4. Uvicorn multiple workers logic (optimized for 4-core EC2)
if __name__ == "__main__":
    # When running directly via `python main.py`
    # We use workers=4 to maximize throughput on a 4-core instance.
    uvicorn.run("main:app", host="0.0.0.0", port=8000, workers=4, log_level="info")
