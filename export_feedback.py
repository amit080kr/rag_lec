import sqlite3
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DB_PATH = "telemetry.db"
OUTPUT_FILE = "negative_feedback.jsonl"

def export_negative_feedback():
    """Queries telemetry_logs for queries where is_helpful=False and exports to JSONL."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            # is_helpful is stored as BOOLEAN (0 or 1 in SQLite)
            cursor.execute("""
                SELECT trace_id, original_query, retrieved_chunk_ids, llm_answer, latency_ms, timestamp
                FROM telemetry_logs
                WHERE is_helpful = 0
            """)
            
            rows = cursor.fetchall()
            
            if not rows:
                logger.info("No negative feedback found to export.")
                return

            with open(OUTPUT_FILE, "w") as f:
                for row in rows:
                    record = {
                        "trace_id": row[0],
                        "original_query": row[1],
                        "retrieved_chunk_ids": json.loads(row[2]) if row[2] else [],
                        "llm_answer": row[3],
                        "latency_ms": row[4],
                        "timestamp": row[5]
                    }
                    f.write(json.dumps(record) + "\n")
                    
            logger.info(f"Successfully exported {len(rows)} records to {OUTPUT_FILE}")
            
    except Exception as e:
        logger.error(f"Failed to export negative feedback: {e}")

if __name__ == "__main__":
    export_negative_feedback()
