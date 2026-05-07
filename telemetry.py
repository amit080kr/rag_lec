import sqlite3
import json
import logging
import os

logger = logging.getLogger(__name__)

DB_PATH = "telemetry.db"

def init_db():
    """Initialize the SQLite database for telemetry logging."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS telemetry_logs (
                    trace_id TEXT PRIMARY KEY,
                    original_query TEXT NOT NULL,
                    retrieved_chunk_ids TEXT,
                    llm_answer TEXT,
                    latency_ms REAL,
                    is_helpful BOOLEAN,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
    except Exception as e:
        logger.error(f"Failed to initialize telemetry DB: {e}")

# Call init_db immediately upon module import
init_db()

def log_query(trace_id: str, original_query: str, retrieved_chunk_ids: list, llm_answer: str, latency_ms: float):
    """Logs the results of a RAG query to the telemetry DB."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO telemetry_logs (trace_id, original_query, retrieved_chunk_ids, llm_answer, latency_ms)
                VALUES (?, ?, ?, ?, ?)
            """, (
                trace_id, 
                original_query, 
                json.dumps(retrieved_chunk_ids), 
                llm_answer, 
                latency_ms
            ))
            conn.commit()
            logger.info(f"Telemetry logged for trace_id: {trace_id}")
    except Exception as e:
        logger.error(f"Failed to log telemetry: {e}")

def update_feedback(trace_id: str, is_helpful: bool) -> bool:
    """Updates the is_helpful flag for a given trace_id."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE telemetry_logs SET is_helpful = ? WHERE trace_id = ?
            """, (is_helpful, trace_id))
            conn.commit()
            return cursor.rowcount > 0
    except Exception as e:
        logger.error(f"Failed to update feedback for trace {trace_id}: {e}")
        return False
