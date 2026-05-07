import logging
import os
from qdrant_client import QdrantClient
from qdrant_client.http import models

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def initialize_qdrant_collection(
    collection_name: str,
    host: str = os.getenv("QDRANT_HOST", "localhost"),
    port: int = 6333
):
    """
    Initializes a Qdrant collection for Hybrid Search with Scalar Quantization,
    Payload Indexes, and a Full-text Index.
    """
    logger.info(f"Connecting to Qdrant at {host}:{port}...")
    client = QdrantClient(host=host, port=port)
    
    # 1. Create or recreate collection
    if client.collection_exists(collection_name):
        logger.info(f"Collection '{collection_name}' already exists. Deleting it to recreate...")
        client.delete_collection(collection_name)
        
    logger.info(f"Creating collection '{collection_name}' with Vector Size 384 and Scalar Quantization...")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=384,  # BAAI/bge-small-en-v1.5 vector size
            distance=models.Distance.COSINE
        ),
        # 2. Enable Scalar Quantization to reduce memory usage on EC2
        quantization_config=models.ScalarQuantization(
            scalar=models.ScalarQuantizationConfig(
                type=models.ScalarType.INT8,
                quantile=0.99,
                always_ram=True
            )
        )
    )
    
    # 3. Create Payload Indexes for metadata
    # Explicitly creating indexes for fields that will be used in filtering
    logger.info("Creating Payload Indexes for metadata fields...")
    client.create_payload_index(
        collection_name=collection_name,
        field_name="source_file",
        field_schema=models.PayloadSchemaType.KEYWORD
    )
    
    client.create_payload_index(
        collection_name=collection_name,
        field_name="page_number",
        field_schema=models.PayloadSchemaType.INTEGER
    )
    
    client.create_payload_index(
        collection_name=collection_name,
        field_name="chunk_index",
        field_schema=models.PayloadSchemaType.INTEGER
    )

    # 4. Create Full-text index on the 'content' field to support BM25/Sparse/Hybrid search
    logger.info("Creating Full-text index on 'content' field for BM25 support...")
    client.create_payload_index(
        collection_name=collection_name,
        field_name="content",
        field_schema=models.TextIndexParams(
            type="text",
            tokenizer=models.TokenizerType.WORD,
            min_token_len=2,
            max_token_len=15,
            lowercase=True,
        )
    )
    
    logger.info(f"Successfully initialized Qdrant collection: {collection_name}")

if __name__ == "__main__":
    COLLECTION_NAME = "documents_hybrid_search"
    try:
        initialize_qdrant_collection(COLLECTION_NAME)
    except Exception as e:
        logger.error(f"Failed to initialize Qdrant collection: {e}")
