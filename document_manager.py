import sqlite3
import hashlib
import logging
from pathlib import Path
from typing import List, Optional, Union
from datetime import datetime, timezone

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DocumentManager:
    """
    Manages incremental ingestion of PDF and CSV documents from a local directory.
    Uses a SQLite database to track file hashes and timestamps, ensuring only
    new or modified files are returned for processing.
    """

    def __init__(self, directory: Union[str, Path], db_path: Union[str, Path] = "document_tracking.db", chunk_size: int = 8192):
        self.directory = Path(directory)
        self.db_path = Path(db_path)
        self.chunk_size = chunk_size
        
        if not self.directory.exists() or not self.directory.is_dir():
            logger.error(f"Directory does not exist or is not a directory: {self.directory}")
            raise ValueError(f"Invalid directory: {self.directory}")

        self._init_db()

    def _init_db(self) -> None:
        """Initializes the SQLite database and creates the tracking table if it doesn't exist."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS document_tracker (
                        file_path TEXT PRIMARY KEY,
                        file_hash TEXT NOT NULL,
                        last_processed_timestamp TEXT NOT NULL
                    )
                """)
                conn.commit()
            logger.info(f"Initialized tracking database at {self.db_path}")
        except sqlite3.Error as e:
            logger.error(f"Database initialization failed: {e}")
            raise

    def _calculate_hash(self, file_path: Path) -> str:
        """Calculates the SHA-256 hash of a file's content reading in chunks."""
        hasher = hashlib.sha256()
        try:
            with open(file_path, 'rb') as f:
                while chunk := f.read(self.chunk_size):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except IOError as e:
            logger.error(f"Failed to read file for hashing {file_path}: {e}")
            raise

    def get_files_to_process(self) -> List[Path]:
        """
        Scans the directory for PDFs and CSVs, compares them against the database,
        and returns a list of files that are new or have been modified.
        """
        logger.info(f"Scanning directory {self.directory} for PDF and CSV files...")
        
        # 1. Scan directory for PDFs and CSVs
        files_to_check: List[Path] = []
        files_to_check.extend(self.directory.rglob("*.pdf"))
        files_to_check.extend(self.directory.rglob("*.csv"))
        
        files_to_process: List[Path] = []
        now_str = datetime.now(timezone.utc).isoformat()

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                for file_path in files_to_check:
                    try:
                        current_hash = self._calculate_hash(file_path)
                        path_str = str(file_path.absolute())
                        
                        # Check database for existing record
                        cursor.execute(
                            "SELECT file_hash FROM document_tracker WHERE file_path = ?",
                            (path_str,)
                        )
                        result = cursor.fetchone()
                        
                        if result is None:
                            # New file
                            logger.info(f"New file detected: {file_path.name}")
                            files_to_process.append(file_path)
                            cursor.execute(
                                "INSERT INTO document_tracker (file_path, file_hash, last_processed_timestamp) VALUES (?, ?, ?)",
                                (path_str, current_hash, now_str)
                            )
                        elif result[0] != current_hash:
                            # Modified file
                            logger.info(f"Modified file detected (hash changed): {file_path.name}")
                            files_to_process.append(file_path)
                            cursor.execute(
                                "UPDATE document_tracker SET file_hash = ?, last_processed_timestamp = ? WHERE file_path = ?",
                                (current_hash, now_str, path_str)
                            )
                        else:
                            # File unchanged
                            logger.debug(f"File unchanged, skipping: {file_path.name}")
                            
                    except Exception as e:
                        logger.error(f"Error processing file {file_path}: {e}")
                        continue
                
                conn.commit()
                
        except sqlite3.Error as e:
            logger.error(f"Database error during file processing: {e}")
            raise

        logger.info(f"Scan complete. Found {len(files_to_process)} file(s) that need processing.")
        return files_to_process

if __name__ == "__main__":
    # Example Usage
    import tempfile
    import os
    
    # Create a dummy environment to test
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create some mock files
        pdf_file = Path(temp_dir) / "test1.pdf"
        pdf_file.write_text("dummy pdf content")
        
        csv_file = Path(temp_dir) / "test2.csv"
        csv_file.write_text("col1,col2\n1,2")
        
        # Initialize the manager
        manager = DocumentManager(directory=temp_dir, db_path=Path(temp_dir) / "test.db")
        
        # First run: Should find both files
        logger.info("--- FIRST RUN ---")
        files = manager.get_files_to_process()
        logger.info(f"Files to process: {[f.name for f in files]}")
        
        # Second run: Should find no files
        logger.info("--- SECOND RUN ---")
        files = manager.get_files_to_process()
        logger.info(f"Files to process: {[f.name for f in files]}")
        
        # Modify a file
        logger.info("--- MODIFYING A FILE ---")
        csv_file.write_text("col1,col2\n1,2\n3,4")
        
        # Third run: Should find only the modified CSV
        logger.info("--- THIRD RUN ---")
        files = manager.get_files_to_process()
        logger.info(f"Files to process: {[f.name for f in files]}")
