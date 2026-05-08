import json
import logging
import os
from qdrant_client import QdrantClient
from openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Fetch the Groq API key from the environment
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    logger.warning("GROQ_API_KEY environment variable not found. The LLM generation will fail unless set.")

# Configure the OpenAI client to use the Groq endpoint
CLIENT = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=GROQ_API_KEY)
MODEL_NAME = "llama-3.1-8b-instant" # Fast Groq model suitable for dataset generation

def get_sample_chunks(collection_name: str = "documents_hybrid_search", limit: int = 10):
    """Fetches a sample of document chunks from Qdrant."""
    logger.info(f"Fetching {limit} chunks from Qdrant collection: {collection_name}")
    client = QdrantClient(host=os.getenv("QDRANT_HOST", "localhost"), port=6333)
    
    # We use scroll to just grab the first N records
    response, _ = client.scroll(
        collection_name=collection_name,
        limit=limit,
        with_payload=True
    )
    return response

def generate_questions_for_chunk(chunk_content: str) -> list[str]:
    """Uses LLM to generate exactly 2 questions that this chunk perfectly answers."""
    prompt = f"""
    You are an expert dataset creator. 
    Read the following document chunk and generate EXACTLY TWO distinct, realistic user questions.
    The chunk must be the perfect and definitive answer to both questions.
    
    Document Chunk:
    {chunk_content}
    
    Output ONLY the two questions, one per line, with no bullet points, numbers, or extra text.
    """
    
    try:
        completion = CLIENT.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=150
        )
        
        response_text = completion.choices[0].message.content.strip()
        questions = [q.strip() for q in response_text.split('\n') if q.strip()]
        
        # Fallback if the LLM adds weird formatting
        if len(questions) < 2:
            logger.warning(f"LLM did not generate 2 questions. Output:\n{response_text}")
        return questions[:2]
        
    except Exception as e:
        logger.error(f"Error generating questions from LLM: {e}\n(Ensure GROQ_API_KEY is set correctly).")
        return []

def main():
    chunks = get_sample_chunks(limit=10)
    if not chunks:
        logger.error("No chunks found in Qdrant. Please run vector_engine.py first.")
        return
        
    dataset = []
    
    logger.info(f"Starting generation of Golden Dataset using LLM model: {MODEL_NAME}...")
    for idx, point in enumerate(chunks, 1):
        content = point.payload.get("content", "")
        if not content:
            continue
            
        logger.info(f"Processing chunk {idx}/{len(chunks)} (ID: {point.id})")
        questions = generate_questions_for_chunk(content)
        
        for q in questions:
            # Clean up leading numbers or dashes if the LLM ignored formatting rules
            if q[0].isdigit() or q.startswith('- '):
                q = q.lstrip('1234567890.- )')
                
            dataset.append({
                "question": q,
                "context": content,
                "doc_id": str(point.id)
            })
            
    # Save to JSON
    output_file = "golden_dataset.json"
    if dataset:
        with open(output_file, "w") as f:
            json.dump(dataset, f, indent=4)
        logger.info(f"Successfully generated Golden Dataset with {len(dataset)} items. Saved to {output_file}")
    else:
        logger.warning("No questions were generated. Check your LLM connection.")

if __name__ == "__main__":
    main()
