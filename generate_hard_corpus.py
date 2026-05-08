import os
import time
import logging
from openai import OpenAI
from document_processor import ChunkMetadata, DocumentChunk
from vector_engine import VectorEngine

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Fetch the Groq API key from the environment
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    logger.warning("GROQ_API_KEY environment variable not found.")

CLIENT = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=GROQ_API_KEY)
MODEL_NAME = "llama-3.1-8b-instant"

def generate_corpus_batch(batch_num: int) -> list[DocumentChunk]:
    """Generates 10 topics per batch. For each topic: 1 Golden, 2 Distractors."""
    
    prompt = f"""
    You are an expert HR and IT policy generator for an enterprise company.
    Generate a JSON list of 10 distinct topics (e.g., "Remote Work Policy", "Expense Reimbursement", "VPN Setup").
    For each topic, provide:
    1. A "golden_chunk": The official, current policy for 2024.
    2. A "distractor_1": The outdated policy from 2023 (semantically similar but factually different numbers/rules).
    3. A "distractor_2": The policy for contractors or a different department (semantically similar but different facts).

    Ensure chunks are 2-3 sentences long.

    Output STRICTLY in the following JSON format:
    [
      {{
        "topic": "Expense Reimbursement",
        "golden_chunk": "The 2024 dinner expense limit is $75 per day. Receipts are required for all transactions over $15.",
        "distractor_1": "The 2023 dinner expense limit is $50 per day. Receipts are required for all transactions over $25.",
        "distractor_2": "For external contractors, the dinner expense limit is $30 per day and pre-approval is required."
      }}
    ]
    Do not output any markdown formatting, only raw JSON.
    """
    
    try:
        completion = CLIENT.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,
            response_format={"type": "json_object"} # We will actually use json block but Groq allows json_object
        )
        
        response_text = completion.choices[0].message.content.strip()
        
        # Parse JSON manually if it's wrapped in a dict
        import json
        try:
            data = json.loads(response_text)
            if isinstance(data, dict):
                # Groq json_object often returns a dict with a root key
                key = list(data.keys())[0]
                items = data[key]
            else:
                items = data
        except json.JSONDecodeError:
            # Fallback cleanup
            start = response_text.find('[')
            end = response_text.rfind(']') + 1
            items = json.loads(response_text[start:end])
            
        chunks = []
        for i, item in enumerate(items):
            topic = item.get("topic", f"Topic_{i}")
            
            # Golden
            meta_golden = ChunkMetadata(
                source_file=f"{topic}_2024.pdf",
                page_number=1,
                chunk_index=batch_num * 100 + i * 3,
                tenant_id="default",
                access_level="public",
                summary_hint=topic
            )
            chunks.append(DocumentChunk(text=item.get("golden_chunk", ""), metadata=meta_golden))
            
            # Distractor 1
            meta_d1 = ChunkMetadata(
                source_file=f"{topic}_2023_Archive.pdf",
                page_number=1,
                chunk_index=batch_num * 100 + i * 3 + 1,
                tenant_id="default",
                access_level="public",
                summary_hint=topic + " (Archived)"
            )
            chunks.append(DocumentChunk(text=item.get("distractor_1", ""), metadata=meta_d1))
            
            # Distractor 2
            meta_d2 = ChunkMetadata(
                source_file=f"{topic}_Contractors.pdf",
                page_number=1,
                chunk_index=batch_num * 100 + i * 3 + 2,
                tenant_id="default",
                access_level="public",
                summary_hint=topic + " (Contractors)"
            )
            chunks.append(DocumentChunk(text=item.get("distractor_2", ""), metadata=meta_d2))
            
        return chunks
        
    except Exception as e:
        logger.error(f"Error generating corpus batch: {e}")
        return []

def main():
    logger.info("Starting generation of Hard Corpus...")
    all_chunks = []
    
    # Generate 8 batches of 10 topics = 80 topics = 240 total chunks (80 golden, 160 distractors)
    for i in range(8):
        logger.info(f"Generating batch {i+1}/8...")
        batch_chunks = generate_corpus_batch(i)
        all_chunks.extend(batch_chunks)
        time.sleep(1) # Prevent rate limits
        
    if not all_chunks:
        logger.error("Failed to generate any chunks.")
        return
        
    logger.info(f"Generated {len(all_chunks)} total chunks. Upserting to Qdrant...")
    
    engine = VectorEngine(collection_name="documents_hybrid_search")
    
    # Wipe the collection to ensure a clean slate
    logger.info("Wiping existing Qdrant collection to create a clean Needle-in-a-Haystack test...")
    if engine.qdrant_client.collection_exists("documents_hybrid_search"):
        engine.qdrant_client.delete_collection("documents_hybrid_search")
        
    # VectorEngine upsert will automatically recreate the collection if we call _initialize_collection
    # But wait, `VectorEngine.upsert_documents` does not recreate if it doesn't exist?
    # Ah, VectorEngine in init DOES check, but we deleted it AFTER init. 
    # Let's initialize it manually.
    try:
        from qdrant_initializer import QdrantInitializer
        init = QdrantInitializer()
        init.initialize_collections()
        logger.info("Recreated collection successfully.")
    except Exception as e:
        logger.warning(f"Failed to run QdrantInitializer: {e}")
    
    # Now upsert
    engine.upsert_documents(all_chunks, batch_size=50)
    logger.info("Hard Corpus successfully deployed to Qdrant.")

if __name__ == "__main__":
    main()
