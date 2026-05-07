import logging
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)

class SemanticCache:
    """
    A lightweight, in-memory semantic cache designed to short-circuit the RAG pipeline.
    It stores the embedded query vectors and their resulting LLM JSON response.
    When a new query arrives, it calculates cosine similarity against cached vectors.
    If similarity > threshold, it returns the cached response, saving expensive vector DB
    and LLM generation costs (FinOps optimization).
    
    In a production system, this would be backed by Redis or Memcached.
    """
    
    def __init__(self, similarity_threshold: float = 0.95):
        self.similarity_threshold = similarity_threshold
        # Stores tuples of (query_vector, answer_json_string)
        self.cache = []
        logger.info(f"Initialized SemanticCache with threshold {self.similarity_threshold}")
        
    def _cosine_similarity(self, vec1: list, vec2: list) -> float:
        """Calculates cosine similarity between two lists of floats."""
        a = np.array(vec1)
        b = np.array(vec2)
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
            
        return float(dot_product / (norm_a * norm_b))
        
    def check_cache(self, query_vector: list) -> Optional[dict]:
        """
        Iterates over the cache to find a semantically identical query.
        Returns the cached answer dictionary if a match is found, else None.
        """
        for cached_vector, cached_answer in self.cache:
            sim = self._cosine_similarity(query_vector, cached_vector)
            if sim >= self.similarity_threshold:
                logger.info(f"Semantic Cache HIT! Similarity: {sim:.4f}")
                return cached_answer
                
        return None
        
    def add_to_cache(self, query_vector: list, answer: dict):
        """Adds a new query vector and its resulting answer to the cache."""
        # Simple cap to prevent infinite memory growth for the demo
        if len(self.cache) >= 1000:
            self.cache.pop(0) # Remove oldest
            
        self.cache.append((query_vector, answer))
        logger.debug("Added new response to Semantic Cache.")
