# ADR-001: RAG Pipeline Core Architecture Decisions

**Date:** 2026-05-08
**Context:** We are deploying an enterprise-grade Hybrid RAG application onto an AWS EC2 instance. The architecture must balance scalable performance, low latency, robust security (prompt injection & RBAC), and cost efficiency (FinOps). 

This document records the foundational architectural decisions made to satisfy these production constraints.

---

## Decision 1: Cloud Deployment (AWS EC2 & Docker Compose)

### Decision
We are using AWS EC2 instance for the cloud deployment. The entire stack (FastAPI Backend and Qdrant Vector Database) is containerized and orchestrated using `docker-compose`.

### Consequences
- **Positive (Availability):** The system is now a scalable web service accessible via a public IP, rather than a local prototype.
- **Positive (Consistency):** Docker ensures the environment (Python dependencies, ONNX runtime, Qdrant binary) behaves exactly the same in production as it did in development.
- **Negative:** Introduces cloud infrastructure management overhead and requires secure SSH key management.

---

## Decision 2: Groq API for High-Speed LLM Inference

### Decision
We chose to offload LLM processing (Security Firewall, Query Rewriting, and the Analytical Answering Engine) to the **Groq API** using the `llama-3.1-8b-instant` model, rather than running LLMs locally.

### Consequences
- **Positive (Latency):** Groq's specialized LPU hardware provides unparalleled inference speeds (often >800 tokens/second), reducing total user wait time to under 1-2 seconds per query.
- **Positive (Resource Efficiency):** Removes the need for expensive, high-RAM GPU instances on AWS. We can run the API and Vector DB on a lightweight 1GB/2GB RAM EC2 instance (augmented with swap space).
- **Negative:** Introduces a dependency on a third-party SaaS API and requires managing the `GROQ_API_KEY` securely in the production environment.

---

## Decision 3: Local CPU Embeddings (FastEmbed)

### Decision
We chose to generate dense vectors using **FastEmbed** (`BAAI/bge-small-en-v1.5`) running entirely locally on the EC2 CPU, rather than offloading to an external embedding API like `text-embedding-3-small`.

### Consequences
- **Positive (FinOps):** We eliminate recurring API costs associated with embedding queries and documents.
- **Positive (Data Privacy):** Enterprise documents are never transmitted over the internet for vectorization.
- **Negative:** We sacrifice a negligible margin of embedding accuracy compared to state-of-the-art closed models, mitigated by our use of Hybrid Search.

---

## Decision 4: Reciprocal Rank Fusion (RRF) for Hybrid Search

### Decision
Instead of relying purely on dense semantic search, we implemented a **Hybrid Search** strategy fusing Dense Vectors with Sparse Exact-Match Text (Local BM25), combined mathematically using **Reciprocal Rank Fusion (RRF)**. We further refine this using a local cross-encoder (`ms-marco-MiniLM-L-12-v2` via FlashRank).

### Consequences
- **Positive (Recall Accuracy):** Pure semantic search struggles with highly specific acronyms and employee IDs. By fusing it with BM25, exact-matches are pulled to the top.
- **Positive (Robustness):** RRF provides an elegant, parameter-free ranking algorithm.

---

## Decision 5: Multi-Tenancy and Strict RBAC at the Database Layer

### Decision
We implemented Multi-Tenancy and Role-Based Access Control (RBAC). Every uploaded chunk is stamped with a `tenant_id` and `access_level`. During retrieval, the FastAPI dependency injects a strict `models.Filter` payload directly into the Qdrant query.

### Consequences
- **Positive (Absolute Isolation):** By filtering at the database layer, it is mathematically impossible for the LLM to hallucinate or expose data belonging to another tenant or a restricted access level.
- **Negative:** Requires strict metadata hygiene; any document ingested without proper tags is completely invisible to the system.

---

## Decision 6: Telemetry Flywheel & Semantic Caching

### Decision
We integrated an asynchronous SQLite telemetry logger (`telemetry_logs`) and an in-memory `SemanticCache` (Cosine Similarity > 0.95).

### Consequences
- **Positive (Continuous Improvement):** We can export negatively-reviewed queries via JSONL to continuously fine-tune our retrieval strategies.
## Decision 6: Prompt Injection Security & Context Building

### Decision
To prevent malicious users from bypassing instructions (Prompt Injection) or extracting internal system prompts, we implemented a strict context-wrapping strategy in `llm_context_builder.py`.

### Consequences
- **Positive (Security):** User queries are explicitly isolated at the very bottom of the prompt (`User Query: {query}\nAnswer:`). We enforce the rule: "Answer the user's query using strictly the following context." This structure prevents the user query from acting as system instructions and overriding the behavior.
- **Positive (Token Safety):** The context builder uses `tiktoken` to dynamically truncate context chunks that exceed the strict 4096-token limit, preventing buffer overflow attacks that attempt to break the LLM parser.

---

## Decision 7: Hardened Synthetic Evaluation

### Decision
We rejected a naive 100% accuracy test in favor of a "Needle in a Haystack" synthetic evaluation. We populated Qdrant with 80 Golden Facts and 160 adversarial "Distractor" facts (e.g., outdated or conflicting policies) and queried them with 162 "messy", human-like queries.

### Consequences
- **Performance Baseline Established:** The pipeline achieved **19.75% Recall@1** and **39.51% Recall@5** with a **0.2784 MRR**. 
- **Negative (Limits Exposed):** The adversarial test exposed that our lightweight Cross-Encoder (`ms-marco-MiniLM-L-12-v2`) struggles when queries lack context and documents are highly similar. 
- **Positive (Clear Roadmap):** This realistic benchmark proves the necessity for our next architectural upgrades: replacing the reranker with a heavier model (`bge-reranker-large`) and implementing a pre-retrieval LLM Query Rewriter.