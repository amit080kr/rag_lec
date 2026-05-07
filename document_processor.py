import fitz  # PyMuPDF
import pandas as pd
from pydantic import BaseModel, Field
from pathlib import Path
from typing import List, Optional, Union
import logging
from langchain_text_splitters import TokenTextSplitter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Pydantic Models ---

class ChunkMetadata(BaseModel):
    source_file: str = Field(description="Name of the source file")
    page_number: Optional[int] = Field(description="Page number (1-indexed) if applicable (e.g., PDF)")
    chunk_index: int = Field(description="Sequential index of the chunk within the document")
    summary_hint: str = Field(description="First 100 characters of the document acting as a summary hint")

class DocumentChunk(BaseModel):
    text: str = Field(description="The text content of the chunk")
    metadata: ChunkMetadata = Field(description="Metadata associated with this chunk")


# --- Processor Class ---

class DocumentProcessor:
    """
    Processes PDFs (using PyMuPDF) and CSVs (using pandas) by chunking them
    and returning a list of Pydantic DocumentChunk objects.
    """
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Using tiktoken-based TokenTextSplitter for production-level token splitting
        self.text_splitter = TokenTextSplitter(
            chunk_size=self.chunk_size, 
            chunk_overlap=self.chunk_overlap
        )
        
    def process_file(self, file_path: Union[str, Path]) -> List[DocumentChunk]:
        """Routes the file to the appropriate processor based on extension."""
        path = Path(file_path)
        if not path.exists():
            logger.error(f"File not found: {path}")
            raise FileNotFoundError(f"File not found: {path}")
            
        ext = path.suffix.lower()
        if ext == '.pdf':
            return self._process_pdf(path)
        elif ext == '.csv':
            return self._process_csv(path)
        else:
            logger.warning(f"Unsupported file type: {ext}. Skipping {path.name}.")
            return []

    def _process_pdf(self, file_path: Path) -> List[DocumentChunk]:
        """Extracts text from a PDF, splits it into chunks, and attaches metadata."""
        logger.info(f"Processing PDF: {file_path.name}")
        chunks: List[DocumentChunk] = []
        
        try:
            doc = fitz.open(file_path)
        except Exception as e:
            logger.error(f"Failed to open PDF {file_path.name}: {e}")
            return chunks

        # Extract summary hint (first 100 chars of the document)
        summary_hint = ""
        if len(doc) > 0:
            first_page_text = doc[0].get_text()
            summary_hint = first_page_text[:100].replace('\n', ' ').strip()
            
        global_chunk_idx = 0
        
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text()
            if not text.strip():
                continue
                
            # Split the text of the current page into chunks
            page_chunks = self.text_splitter.split_text(text)
            
            for chunk_text in page_chunks:
                meta = ChunkMetadata(
                    source_file=file_path.name,
                    page_number=page_num,
                    chunk_index=global_chunk_idx,
                    summary_hint=summary_hint
                )
                chunks.append(DocumentChunk(text=chunk_text, metadata=meta))
                global_chunk_idx += 1
                
        doc.close()
        logger.info(f"Generated {len(chunks)} chunks from {file_path.name}")
        return chunks

    def _process_csv(self, file_path: Path) -> List[DocumentChunk]:
        """Extracts text from a CSV, formatting rows as 'Key: Value', and splits it."""
        logger.info(f"Processing CSV: {file_path.name}")
        chunks: List[DocumentChunk] = []
        
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            logger.error(f"Failed to read CSV {file_path.name}: {e}")
            return chunks
            
        # Convert rows into 'Key: Value' string format
        row_strings = []
        for _, row in df.iterrows():
            row_str = "\n".join([f"{col}: {val}" for col, val in row.items()])
            row_strings.append(row_str)
            
        # Join all rows to form the full document text, separated by a distinct delimiter
        full_text = "\n\n---\n\n".join(row_strings)
        
        # Summary hint (first 100 chars of the full text)
        summary_hint = full_text[:100].replace('\n', ' ').strip()
        
        # Split the full text into chunks
        split_chunks = self.text_splitter.split_text(full_text)
        
        for idx, chunk_text in enumerate(split_chunks):
            meta = ChunkMetadata(
                source_file=file_path.name,
                page_number=None, # CSVs do not have a traditional page number
                chunk_index=idx,
                summary_hint=summary_hint
            )
            chunks.append(DocumentChunk(text=chunk_text, metadata=meta))
            
        logger.info(f"Generated {len(chunks)} chunks from {file_path.name}")
        return chunks

if __name__ == "__main__":
    # Example usage / Test logic
    import tempfile
    
    with tempfile.TemporaryDirectory() as temp_dir:
        dir_path = Path(temp_dir)
        
        # 1. Create a dummy CSV
        csv_file = dir_path / "data.csv"
        csv_file.write_text("Name,Age,Role\nAlice,30,Engineer\nBob,25,Designer\nCharlie,35,Manager")
        
        # 2. Process
        # We use a small chunk_size to demonstrate chunking works on our tiny test file
        processor = DocumentProcessor(chunk_size=10, chunk_overlap=2) 
        results = processor.process_file(csv_file)
        
        print(f"Total chunks created: {len(results)}")
        for chunk in results:
            print(f"\n--- Chunk {chunk.metadata.chunk_index} ---")
            print(f"Metadata: {chunk.metadata.model_dump()}")
            print(f"Text:\n{chunk.text}")
