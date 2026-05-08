import logging
import os
from typing import List, Dict, Any, Tuple
# pyrefly: ignore [missing-import]
from qdrant_client import QdrantClient
from qdrant_client.http import models
from fastembed import TextEmbedding
from rank_bm25 import BM25Okapi
from flashrank import Ranker, RerankRequest
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Dedicated logger for low-confidence / OOD queries
ood_logger = logging.getLogger("ood_queries")
ood_logger.setLevel(logging.INFO)
ood_fh = logging.FileHandler("low_confidence_queries.log")
ood_fh.setFormatter(logging.Formatter('%(asctime)s - OOD QUERY: %(message)s'))
ood_logger.addHandler(ood_fh)
ood_logger.propagate = False # Prevent double-logging to console

class HybridRetriever:
    """
    Performs Hybrid Search using Dense Vector Search and Local BM25 Scoring on
    documents filtered by Qdrant's Full-Text Match. Combines results using RRF.
    """
    
    def __init__(
        self, 
        collection_name: str, 
        qdrant_host: str = os.getenv("QDRANT_HOST", "localhost"), 
        qdrant_port: int = 6333,
        embedding_model_name: str = "BAAI/bge-small-en-v1.5",
        rrf_k: int = 60
    ):
        self.collection_name = collection_name
        self.qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)
        
        logger.info(f"Loading local embedding model: {embedding_model_name}...")
        self.embedding_model = TextEmbedding(model_name=embedding_model_name)
        self.rrf_k = rrf_k
        
        logger.info("Loading ultra-lightweight re-ranker model: ms-marco-MiniLM-L-12-v2...")
        self.ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="/tmp")
        
        logger.info("HybridRetriever initialized successfully.")

    def _dense_search(self, query: str, tenant_id: str, allowed_access_levels: List[str], limit: int = 60) -> List[models.ScoredPoint]:
        """Performs a dense vector search in Qdrant with tenant and RBAC filtering."""
        logger.info("Executing dense vector search...")
        # Embed query text
        query_vector = list(self.embedding_model.embed([query]))[0].tolist()
        
        # Build the payload filter
        query_filter = models.Filter(
            must=[
                models.FieldCondition(key="tenant_id", match=models.MatchValue(value=tenant_id)),
                models.FieldCondition(key="access_level", match=models.MatchAny(any=allowed_access_levels))
            ]
        )
        
        results = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=limit,
            with_payload=True
        )
        return results.points

    def _sparse_bm25_search(self, query: str, tenant_id: str, allowed_access_levels: List[str], limit: int = 60) -> List[Tuple[str, float, dict]]:
        """
        Since Qdrant's TextIndexParams provides boolean filtering (MatchText)
        and not BM25 scoring, we fetch matched candidates and rank them locally 
        using BM25Okapi to generate accurate sparse ranking.
        """
        logger.info("Executing sparse full-text search and local BM25 ranking...")
        
        # 1. Fetch candidate documents that contain the query words and match the tenant/access filters
        # We use MatchText which leverages the Full-text index we created
        scroll_results, _ = self.qdrant_client.scroll(
            collection_name=self.collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="content",
                        match=models.MatchText(text=query)
                    ),
                    models.FieldCondition(key="tenant_id", match=models.MatchValue(value=tenant_id)),
                    models.FieldCondition(key="access_level", match=models.MatchAny(any=allowed_access_levels))
                ]
            ),
            limit=limit,
            with_payload=True
        )
        
        if not scroll_results:
            return []

        # 2. Tokenize candidates and query for BM25
        tokenized_corpus = [point.payload.get("content", "").lower().split() for point in scroll_results]
        tokenized_query = query.lower().split()
        
        # 3. Calculate BM25 scores
        bm25 = BM25Okapi(tokenized_corpus)
        doc_scores = bm25.get_scores(tokenized_query)
        
        # 4. Pair scores with points and sort descending
        scored_candidates = []
        for i, point in enumerate(scroll_results):
            scored_candidates.append((str(point.id), doc_scores[i], point.payload))
            
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        return scored_candidates

    def search(
        self, 
        query: str, 
        tenant_id: str,
        allowed_access_levels: List[str],
        hybrid_top_k: int = 20, 
        rerank_top_k: int = 5,
        confidence_threshold: float = 0.4
    ) -> List[Dict[str, Any]]:
        """
        Executes Dense and Sparse searches, fuses them with Reciprocal Rank Fusion (RRF),
        and returns the top_k results after re-ranking with FlashRank.
        """
        logger.info(f"Initiating Hybrid Search for query: '{query}'")
        
        # Fetch up to 60 candidates from both strategies
        dense_results = self._dense_search(query, tenant_id, allowed_access_levels, limit=60)
        sparse_results = self._sparse_bm25_search(query, tenant_id, allowed_access_levels, limit=60)
        
        rrf_scores: Dict[str, float] = {}
        payloads: Dict[str, dict] = {}
        
        # 1. Rank Dense Results
        for rank, hit in enumerate(dense_results, start=1):
            point_id = str(hit.id)
            rrf_scores[point_id] = rrf_scores.get(point_id, 0.0) + 1.0 / (self.rrf_k + rank)
            payloads[point_id] = hit.payload
            
        # 2. Rank Sparse BM25 Results
        for rank, (point_id, bm25_score, payload) in enumerate(sparse_results, start=1):
            # Only count if the BM25 score is actually > 0
            if bm25_score > 0:
                rrf_scores[point_id] = rrf_scores.get(point_id, 0.0) + 1.0 / (self.rrf_k + rank)
                payloads[point_id] = payload

        # 3. Sort by combined RRF score descending
        sorted_results = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
        
        # 4. Prepare candidates for FlashRank
        candidates = []
        for point_id, score in sorted_results[:hybrid_top_k]:
            candidates.append({
                "id": point_id,
                "text": payloads[point_id].get("content", ""),
                "meta": payloads[point_id]
            })
            
        if not candidates:
            logger.info("No candidates found in database matching the filters. Skipping reranking.")
            return []
            
        logger.info(f"Hybrid Search complete. Re-ranking the top {len(candidates)} candidates...")
        
        # 5. Ultra-lightweight Re-ranking
        start_time = time.time()
        
        # Create the RerankRequest. FlashRank automatically sorts the returned list by 'score'
        rerank_request = RerankRequest(query=query, passages=candidates)
        reranked_results = self.ranker.rerank(rerank_request)
        
        latency_ms = (time.time() - start_time) * 1000
        logger.info(f"Re-ranking completed in {latency_ms:.2f} ms")
        
        # OOD Check: If the absolutely best passage is still below our confidence threshold,
        # it means the query is Out-Of-Distribution (or out of context).
        if reranked_results and reranked_results[0]["score"] < confidence_threshold:
            max_score = reranked_results[0]["score"]
            logger.warning(f"OOD query detected. Top score {max_score:.4f} is below threshold {confidence_threshold}.")
            ood_logger.info(f"'{query}' (Max Score: {max_score:.4f})")
            
            return [{
                "id": "ood-response",
                "relevance_score": 0.0,
                "payload": {
                    "content": "I don't have enough context in my knowledge base to accurately answer this question."
                },
                "rerank_latency_ms": round(latency_ms, 2)
            }]
        
        # 6. Format Final Output
        final_results = []
        for res in reranked_results[:rerank_top_k]:
            final_results.append({
                "id": res["id"],
                "relevance_score": round(res["score"], 4),
                "payload": res["meta"],
                "rerank_latency_ms": round(latency_ms, 2)
            })
            
        return final_results

if __name__ == "__main__":
    # Test the Hybrid Retriever
    retriever = HybridRetriever(collection_name="documents_hybrid_search")
    
    # Execute a test query against the mock chunks we upserted previously
    test_query = "test content for chunk 5"
    results = retriever.search(
        query=test_query, 
        tenant_id="test_tenant", 
        allowed_access_levels=["public", "internal"], 
        hybrid_top_k=20, 
        rerank_top_k=5
    )
    
    print(f"\n--- Top Results for '{test_query}' ---")
    for r in results:
        print(f"ID: {r['id']} | Relevance: {r['relevance_score']} | Rerank Latency: {r['rerank_latency_ms']}ms | Content: {r['payload'].get('content')}")
