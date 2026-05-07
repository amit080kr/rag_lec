import tiktoken
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def build_safe_context(
    query: str, 
    retrieved_chunks: List[Dict[str, Any]], 
    max_tokens: int = 4096, 
    model_name: str = "gpt-4o"
) -> str:
    """
    Safely builds the final LLM prompt context without overflowing the context window.
    Prioritizes highest-scoring chunks and iteratively trims the least relevant chunks
    until the total tokens fit within the max limit.
    """
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        logger.warning(f"Model '{model_name}' not found. Defaulting to 'cl100k_base' encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
        
    # Define standard prompt wrappers
    system_prompt = "Answer the user's query using strictly the following context:\n\n"
    query_prompt = f"\n\nUser Query: {query}\nAnswer:"
    
    base_tokens = len(encoding.encode(system_prompt + query_prompt))
    
    # 1. Handle Out-Of-Distribution queries natively
    if retrieved_chunks and retrieved_chunks[0].get("id") == "ood-response":
        logger.info("OOD response detected. Bypassing context builder.")
        return retrieved_chunks[0]["payload"].get("content", "")
            
    # 2. Iteratively add chunks starting from highest relevance (assuming input is pre-sorted)
    accepted_chunks = []
    current_tokens = base_tokens
    
    for chunk in retrieved_chunks:
        content = chunk.get("payload", {}).get("content", "")
        formatted_chunk = f"---\n{content}\n"
        chunk_tokens = len(encoding.encode(formatted_chunk))
        
        if current_tokens + chunk_tokens <= max_tokens:
            accepted_chunks.append(formatted_chunk)
            current_tokens += chunk_tokens
        else:
            logger.warning(
                f"Context overflow prevented. Dropped lower-relevance chunk {chunk.get('id')} "
                f"due to token limits. (Current: {current_tokens}, Attempted: {current_tokens + chunk_tokens})"
            )
            # Stop prioritizing further chunks since we've hit the limit
            break
            
    final_context = system_prompt + "".join(accepted_chunks) + query_prompt
    logger.info(f"Built safe context containing {len(accepted_chunks)}/{len(retrieved_chunks)} chunks. Total Tokens: {current_tokens}/{max_tokens}")
    
    return final_context
