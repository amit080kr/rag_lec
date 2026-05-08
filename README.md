# Enterprise Hybrid RAG System

An advanced, production-ready Retrieval-Augmented Generation (RAG) system built for AWS EC2. It features a 4-step agentic pipeline, strict Role-Based Access Control (RBAC) multi-tenancy, FinOps semantic caching, and a Data Flywheel telemetry system.

## Key Architecture

- **Backend Framework**: FastAPI (Python)
- **Vector Database**: Qdrant (Dockerized)
- **LLM Inference**: Groq API (`llama-3.1-8b-instant`) for ultra-low latency generation.
- **Local Embeddings**: FastEmbed (`BAAI/bge-small-en-v1.5`) running entirely on local CPU to save API costs and protect data privacy.
- **Hybrid Search**: Fuses Dense Vector Search with Sparse Exact-Match Text (Local BM25) using **Reciprocal Rank Fusion (RRF)**.
- **Reranking**: Ultra-lightweight Cross-Encoder (`ms-marco-MiniLM-L-12-v2` via FlashRank) for ultimate precision.

## Enterprise Features

- **Strict Multi-Tenancy (RBAC)**: All documents are tagged with `tenant_id` and `access_level`. Qdrant enforces a strict Payload Filter *before* retrieval, making it mathematically impossible for the LLM to hallucinate unauthorized data.
- **Security Firewall**: A dedicated prompt-injection LLM intercepts every query before processing.
- **FinOps Semantic Cache**: Redundant queries (Cosine Similarity > 0.95) are short-circuited instantly, bypassing the DB and LLM to save cloud costs.
- **Data Flywheel (Telemetry)**: Asynchronously logs query latency, traces, and retrieved chunks to SQLite. Supports exporting negatively-reviewed queries to JSONL for continuous fine-tuning.

## 📂 Repository Structure

```text
rag_lec/
├── main.py                     # FastAPI backend and core API endpoints
├── hybrid_retriever.py         # RRF, FastEmbed, and FlashRank cross-encoder orchestration
├── vector_engine.py            # Local dense vector generation and Qdrant upserting
├── document_processor.py       # Document chunking and metadata extraction (PDF/CSV)
├── semantic_cache.py           # FinOps caching to prevent redundant LLM/DB calls
├── telemetry.py                # Asynchronous 'Data Flywheel' logger to SQLite
├── llm_context_builder.py      # LLM Security Firewall and Prompt Injection protection
├── generate_hard_corpus.py     # Evaluation: Generates needle-in-haystack Qdrant dataset
├── generate_golden_dataset.py  # Evaluation: Generates "messy" human-like queries 
├── evaluate_retriever.py       # Evaluation: Calculates Recall@1/5 and MRR metrics
├── docker-compose.yml          # Production deployment orchestration
└── docs/                       # Architecture Decision Records (ADRs)
```

## AWS EC2 Production Deployment

This project is fully automated via GitHub Actions to deploy directly to an AWS EC2 instance.

1. Ensure your EC2 instance has Docker and Docker Compose installed.
2. Ensure you have configured an 8GB Swap File if running on a free-tier (1GB RAM) instance to prevent Out-Of-Memory errors during ONNX model loading.
3. Configure your GitHub Repository Secrets (`HOST`, `USERNAME`, `KEY`, `GROQ_API_KEY`).
4. Push to `main`! The GitHub Action will automatically SSH into the server, pull the latest code, and restart the `docker-compose` cluster.

## Documentation

Please see `docs/ADR-001-RAG-Architecture.md` for a comprehensive breakdown of our Engineering Architecture Decision Records.
