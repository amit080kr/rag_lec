import json
import logging
import time
from tabulate import tabulate
from hybrid_retriever import HybridRetriever

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_dataset(file_path: str = "golden_dataset.json"):
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"Dataset '{file_path}' not found. Please run generate_golden_dataset.py first.")
        return []

def evaluate(dataset, top_k_hybrid=20, top_k_rerank=5):
    logger.info("Initializing HybridRetriever for Evaluation...")
    # Initialize the retriever
    retriever = HybridRetriever(collection_name="documents_hybrid_search")
    
    total_queries = len(dataset)
    if total_queries == 0:
        return
        
    hits_at_1 = 0
    hits_at_k = 0
    reciprocal_ranks = []
    total_latency_ms = 0
    
    logger.info(f"Starting evaluation of {total_queries} queries from Golden Dataset...")
    
    for idx, item in enumerate(dataset, 1):
        query = item["question"]
        expected_doc_id = item["doc_id"]
        
        logger.info(f"Evaluating query {idx}/{total_queries}: '{query}'")
        
        # Track overall latency including dense+sparse search and reranking
        start_time = time.time()
        
        # Run retrieval pipeline
        # We fetch top 20 candidates, and rerank down to 5
        results = retriever.search(
            query=query, 
            tenant_id="default", 
            allowed_access_levels=["public", "internal", "confidential"],
            hybrid_top_k=top_k_hybrid, 
            rerank_top_k=top_k_rerank
        )
        
        latency_ms = (time.time() - start_time) * 1000
        total_latency_ms += latency_ms
        
        # Calculate Rank
        rank = 0
        for i, res in enumerate(results, 1):
            if res["id"] == expected_doc_id:
                rank = i
                break
                
        # Metrics Calculation
        if rank == 1:
            hits_at_1 += 1
        if rank > 0:
            hits_at_k += 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)
            
    # Aggregations
    recall_at_1 = (hits_at_1 / total_queries) * 100
    recall_at_k = (hits_at_k / total_queries) * 100
    mrr = sum(reciprocal_ranks) / total_queries
    avg_latency = total_latency_ms / total_queries
    
    # Print results table
    table = [
        ["Total Queries Evaluated", total_queries],
        ["Recall@1", f"{recall_at_1:.2f}%"],
        [f"Recall@{top_k_rerank}", f"{recall_at_k:.2f}%"],
        ["Mean Reciprocal Rank (MRR)", f"{mrr:.4f}"],
        ["Avg Pipeline Latency", f"{avg_latency:.2f} ms"]
    ]
    
    print("\n" + "="*60)
    print("           RETRIEVAL PIPELINE EVALUATION RESULTS           ")
    print("="*60)
    print(tabulate(table, headers=["Metric", "Value"], tablefmt="fancy_grid"))
    print("="*60 + "\n")

if __name__ == "__main__":
    dataset = load_dataset()
    if dataset:
        evaluate(dataset)
