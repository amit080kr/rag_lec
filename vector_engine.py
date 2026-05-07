import logging
import uuid
import os
from typing import List
from qdrant_client import QdrantClient
from qdrant_client.http import models
from fastembed import TextEmbedding

# Import the DocumentChunk model defined previously
from document_processor import DocumentChunk

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VectorEngine:
    """
    Handles local embedding generation and batch upserting into Qdrant.
    """
    
    def __init__(
        self, 
        collection_name: str, 
        qdrant_host: str = os.getenv("QDRANT_HOST", "localhost"), 
        qdrant_port: int = 6333,
        embedding_model_name: str = "BAAI/bge-small-en-v1.5"
    ):
        self.collection_name = collection_name
        self.qdrant_host = qdrant_host
        self.qdrant_port = qdrant_port
        
        logger.info(f"Connecting to Qdrant at {self.qdrant_host}:{self.qdrant_port}...")
        self.qdrant_client = QdrantClient(host=self.qdrant_host, port=self.qdrant_port)
        
        logger.info(f"Loading local embedding model: {embedding_model_name}...")
        # fastembed runs purely locally, bound to CPU
        self.embedding_model = TextEmbedding(model_name=embedding_model_name)
        logger.info("Embedding model loaded successfully.")

    def health_check(self) -> bool:
        """Verifies if the Qdrant service is reachable."""
        try:
            # We can check by calling the collections API
            self.qdrant_client.get_collections()
            logger.info("Health check passed: Qdrant service is reachable.")
            return True
        except Exception as e:
            logger.error(f"Health check failed: Qdrant service is unreachable. Error: {e}")
            return False

    def upsert_documents(self, chunks: List[DocumentChunk], batch_size: int = 100) -> None:
        """
        Embeds text and upserts document chunks into Qdrant in batches to prevent memory spikes.
        """
        if not chunks:
            logger.warning("No chunks provided to upsert.")
            return

        total_chunks = len(chunks)
        logger.info(f"Starting upsert of {total_chunks} chunks to '{self.collection_name}' in batches of {batch_size}...")

        for i in range(0, total_chunks, batch_size):
            batch = chunks[i:i + batch_size]
            
            # Extract texts for the current batch
            texts = [chunk.text for chunk in batch]
            
            # Generate embeddings locally using fastembed CPU
            logger.info(f"Generating embeddings for batch {i // batch_size + 1}...")
            # embed() returns a generator. We evaluate it to get the dense vectors
            embeddings = list(self.embedding_model.embed(texts))
            
            points = []
            for j, chunk in enumerate(batch):
                # Qdrant requires a unique UUID or integer for each point. 
                # We use uuid5 based on the source_file and chunk_index to make updates idempotent
                point_id = str(uuid.uuid5(
                    uuid.NAMESPACE_URL, 
                    f"{chunk.metadata.source_file}_{chunk.metadata.chunk_index}"
                ))
                
                # Format payload to include all metadata + full text (for BM25 hybrid search)
                payload = {
                    "content": chunk.text,  # Maps to the full-text index we initialized
                    "source_file": chunk.metadata.source_file,
                    "page_number": chunk.metadata.page_number,
                    "chunk_index": chunk.metadata.chunk_index,
                    "summary_hint": chunk.metadata.summary_hint
                }
                
                points.append(
                    models.PointStruct(
                        id=point_id,
                        vector=embeddings[j].tolist(),
                        payload=payload
                    )
                )
            
            # Upsert into Qdrant
            try:
                self.qdrant_client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
                logger.info(f"Successfully upserted batch {i // batch_size + 1} ({len(points)} chunks).")
            except Exception as e:
                logger.error(f"Failed to upsert batch {i // batch_size + 1}: {e}")
                raise

        logger.info(f"Finished upserting all {total_chunks} chunks.")


if __name__ == "__main__":
    # Example usage / Test logic
    from document_processor import ChunkMetadata, DocumentChunk
    
    # Run a quick health check
    engine = VectorEngine(collection_name="documents_hybrid_search")
    if engine.health_check():
        # Create some dummy DocumentChunks to test the upsert loop
        test_chunks = []
        for idx in range(1, 15):
            meta = ChunkMetadata(
                source_file="test_doc.csv",
                page_number=1,
                chunk_index=idx,
                summary_hint="This is a test summary."
            )
            chunk = DocumentChunk(text=f"This is the test content for chunk {idx}", metadata=meta)
            test_chunks.append(chunk)
            
        # Test batch size of 5 to demonstrate chunking behavior for our 14 items
        engine.upsert_documents(test_chunks, batch_size=5)
