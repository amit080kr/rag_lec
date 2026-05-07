# ADR-001: RAG Pipeline Core Architecture Decisions

**Date:** 2026-05-07
**Status:** Accepted
**Context:** We are deploying an enterprise-grade Hybrid RAG application onto an AWS EC2 instance. We must optimize for cost (FinOps), execution latency, robust security against prompt injections, and maximal retrieval accuracy for complex corporate terminologies.

This document records three foundational architectural decisions made to satisfy these constraints.

---

## Decision 1: Local CPU Embeddings vs. External API (e.g., OpenAI)

### Decision
We chose to generate dense vectors using **FastEmbed** (`BAAI/bge-small-en-v1.5`) running entirely locally on CPU, rather than offloading to an external embedding API like `text-embedding-3-small`.

### Consequences
- **Positive (FinOps):** We eliminate recurring API costs associated with embedding queries and documents. At scale, this represents thousands of dollars saved monthly.
- **Positive (Data Privacy):** Enterprise documents are never transmitted over the internet for vectorization, ensuring absolute data residency and compliance with strict infosec policies.
- **Positive (Latency):** FastEmbed's ONNX runtime executes in single-digit milliseconds on CPU, removing external network round-trips.
- **Negative:** We sacrifice a negligible margin of embedding accuracy compared to state-of-the-art closed models, but this is mitigated by our use of Hybrid Search.

---

## Decision 2: Reciprocal Rank Fusion (RRF) for Hybrid Search

### Decision
Instead of relying purely on dense semantic search, we implemented a **Hybrid Search** strategy fusing Dense Vectors with Sparse Exact-Match Text (Local BM25), combined mathematically using **Reciprocal Rank Fusion (RRF)**.

### Consequences
- **Positive (Recall Accuracy):** Pure semantic search struggles with highly specific acronyms, employee IDs, and proprietary serial numbers. By fusing it with BM25, we ensure exact-matches are pulled to the top alongside semantically related concepts.
- **Positive (Robustness):** RRF does not require calibrating arbitrary weight scores (e.g., 0.7 dense / 0.3 sparse). It ranks purely based on reciprocal positioning, providing an elegant, parameter-free ranking algorithm.
- **Negative:** Retrieval time is marginally increased as Qdrant must execute two distinct search queries (vector search + payload text match) before local BM25 scoring and RRF fusion occur.

---

## Decision 3: Generic Error Messages for the Security Firewall

### Decision
We implemented a strict LLM-based Cybersecurity Firewall that analyzes queries for prompt injection. When malicious intent is detected, we deliberately suppress the detailed LLM reasoning (e.g., *"I blocked this because it asked for my system prompt"*) and instead return a silent, generic error: `"I'm sorry, I cannot process this request."`

### Consequences
- **Positive (Security Posture):** By masking the exact reason for rejection, we prevent "Prompt Probing." Attackers cannot iteratively guess our firewall's logic or internal prompt rules based on verbose error messages.
- **Positive (Simplicity):** Frontend systems do not need complex error-parsing logic; they simply render a polite, generic failure.
- **Negative:** Legitimate users who accidentally trigger the firewall might be confused as to why their query failed, impacting User Experience. We accept this trade-off for the sake of enterprise security.
