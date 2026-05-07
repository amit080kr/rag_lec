# Production-Grade Hybrid RAG Pipeline

A high-performance, production-ready Retrieval-Augmented Generation (RAG) system built with **FastAPI**, **Qdrant**, and **Hybrid Search** (Dense Vector + Sparse BM25). This pipeline is optimized for local CPU-bound performance and efficient deployment on AWS EC2.

## 🚀 Key Features

- **Incremental Ingestion**: Uses SHA-256 hashing and SQLite to track file changes, ensuring only new or modified documents (PDF/CSV) are processed.
- **Hybrid Retrieval Engine**:
  - **Dense Search**: Semantic retrieval using `fastembed` (BAAI/bge-small-en-v1.5).
  - **Sparse BM25**: Local keyword-based scoring for precise term matching.
  - **RRF (Reciprocal Rank Fusion)**: Native fusion of dense and sparse results.
- **Advanced Re-ranking**: Integrated **FlashRank** (`ms-marco-MiniLM-L-12-v2`) for ultra-lightweight cross-encoder re-scoring with <100ms latency.
- **Production Guardrails**:
  - **OOD (Out-of-Distribution) Detection**: Standardized "I don't know" responses for low-confidence queries.
  - **Context Overflow Protection**: Uses `tiktoken` to trim least-relevant chunks, preventing LLM context window crashes.
- **Evaluation Suite**: Automated "Golden Dataset" generation via LLM and a metrics engine calculating **Recall@K**, **MRR**, and **Latency**.
- **Dockerized Deployment**: Multi-worker FastAPI setup optimized for 4-core EC2 instances with hard resource limits.

## 🛠 Tech Stack

- **Vector Database**: [Qdrant](https://qdrant.tech/)
- **API Framework**: [FastAPI](https://fastapi.tiangolo.com/)
- **Embeddings**: [FastEmbed](https://qdrant.github.io/fastembed/)
- **Re-ranking**: [FlashRank](https://github.com/PrithivirajDamodaran/FlashRank)
- **Document Processing**: PyMuPDF (PDF), Pandas (CSV), LangChain (Text Splitters)
- **Logging**: Structured JSON logging and dedicated OOD audit logs.

## 📁 Project Structure

```text
├── main.py                 # FastAPI Application (Entry point)
├── document_manager.py     # Incremental file tracker (SQLite/SHA-256)
├── document_processor.py   # PDF/CSV parsing and token-based chunking
├── vector_engine.py        # Qdrant upsert and embedding logic
├── hybrid_retriever.py     # Hybrid Search (Dense+Sparse) + RRF + Reranking
├── llm_context_builder.py  # tiktoken-based context trimming utility
├── qdrant_initializer.py   # Collection setup script (Payload/Full-text indexes)
├── generate_golden_dataset.py # LLM-based test query generator
├── evaluate_retriever.py   # Recall@K and MRR evaluation script
├── Dockerfile              # Python-slim production image
├── docker-compose.yml      # Orchestration (API + Qdrant)
└── aws_ec2_deployment_guide.md # Step-by-step AWS setup instructions
```

## ⚙️ Setup & Installation

### Local Development

1. **Clone the repository**:
   ```bash
   git clone https://github.com/amit080kr/rag_lec.git
   cd rag_lec
   ```

2. **Setup Virtual Environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/python  # On Mac/Linux
   pip install -r requirements.txt
   ```

3. **Configure Environment**:
   Set your Groq API key for dataset generation and LLM tasks.
   ```bash
   export GROQ_API_KEY="your-groq-api-key"
   ```

4. **Start Qdrant**:
   ```bash
   docker compose up -d qdrant
   ```

5. **Initialize Collection**:
   ```bash
   python qdrant_initializer.py
   ```

6. **Run the API**:
   ```bash
   python main.py
   ```

---

## 🐳 Docker Deployment (Automated via GitHub Actions)

To prevent hardcoding API keys, this repository uses **GitHub Actions** and **GitHub Secrets** for secure, automated deployment to AWS EC2.

1. **Configure GitHub Secrets**:
   Go to your repository **Settings > Secrets and variables > Actions** and add:
   - `GROQ_API_KEY`: Your Groq API key
   - `EC2_HOST`: The Public IP of your EC2 instance
   - `EC2_USERNAME`: Usually `ubuntu`
   - `EC2_SSH_KEY`: The raw text of your `.pem` key

2. **Automated Deployment**:
   Whenever you push to the `main` branch, GitHub Actions will automatically SSH into your EC2 instance, generate a secure, invisible `.env` file containing your `GROQ_API_KEY`, and trigger `docker compose up --build`. No keys are ever stored in the codebase!
   *The API will be available at `http://localhost:8000` (or your EC2 Public IP).*

2. **Resource Management**: 
   The `docker-compose.yml` is configured with hard limits (3.0 CPU / 4GB RAM for the API) to prevent OOM crashes on shared hosting environments.

---

## 📊 API Endpoints

- **`GET /`**: API Status and welcome message.
- **`POST /ingest`**: Trigger incremental ingestion of a local directory.
  - Body: `{"directory_path": "./data"}`
- **`POST /query`**: Execute Hybrid Search with re-ranking.
  - Body: `{"query": "Your search term", "confidence_threshold": 0.4}`
- **`GET /docs`**: Interactive Swagger documentation.

---

## 🧪 Evaluation

1. **Generate Golden Dataset**: Ensure Ollama or Groq is configured in `generate_golden_dataset.py`.
   ```bash
   python generate_golden_dataset.py
   ```
2. **Run Metrics**:
   ```bash
   python evaluate_retriever.py
   ```
   *Outputs a summary table of Recall@1, Recall@5, MRR, and Latency.*
