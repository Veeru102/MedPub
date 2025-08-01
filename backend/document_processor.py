from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_core.documents import Document
import os
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        try:
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
            )
        except Exception as e:
            logger.error(f"Failed to initialize text splitter: {e}")
            raise

    def extract_metadata(self, pdf_path: str) -> Dict[str, Any]:
        """Extracts metadata from a PDF file."""
        if not os.path.exists(pdf_path):
            logger.error(f"PDF file not found: {pdf_path}")
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
        try:
            loader = PDFPlumberLoader(pdf_path)
            doc = loader.load()[0]  # Get first page to extract metadata
            metadata = doc.metadata
            result = {
                "title": metadata.get("title", ""),
                "author": metadata.get("author", ""),
                "subject": metadata.get("subject", ""),
                "keywords": metadata.get("keywords", ""),
                "page_count": metadata.get("page_count", 0),
                "creation_date": metadata.get("creationDate", ""),
            }
            return result
        except Exception as e:
            logger.error(f"Error extracting metadata from {pdf_path}: {e}")
            raise

    def process_pdf(self, pdf_path: str) -> List[Document]:
        """Processes a PDF file and returns chunks as Langchain Document objects with metadata."""
        if not os.path.exists(pdf_path):
            logger.error(f"PDF file not found: {pdf_path}")
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
        try:
            # Loads PDF using PDFPlumberLoader.
            logger.info(f"Loading PDF file: {pdf_path}")
            loader = PDFPlumberLoader(pdf_path)
            pages = loader.load()
            
            if not pages:
                logger.warning(f"No pages extracted from PDF: {pdf_path}")
                return []
                
            logger.info(f"Successfully loaded {len(pages)} pages from {pdf_path}")
            
            # Splits text into chunks (Langchain Documents).
            logger.info("Splitting text into chunks...")
            lc_documents = self.text_splitter.split_documents(pages)
            
            if not lc_documents:
                logger.warning("No chunks created after splitting text")
                return []
                
            logger.info(f"Created {len(lc_documents)} chunks")
            
            # Extracts metadata and adds to each document.
            try:
                metadata = self.extract_metadata(pdf_path)
                logger.info("Successfully extracted metadata")
            except Exception as e:
                logger.warning(f"Failed to extract metadata, using empty metadata: {e}")
                metadata = {}
            
            for doc in lc_documents:
                doc.metadata = {**doc.metadata, **metadata, "filename": os.path.basename(pdf_path)}
            
            return lc_documents
            
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {e}")
            raise

    def get_chunk_with_context(self, chunks: List[str], chunk_index: int, context_size: int = 2) -> str:
        """Gets a chunk with surrounding context."""
        if not chunks:
            logger.warning("No chunks provided for context retrieval")
            return ""
            
        if chunk_index < 0 or chunk_index >= len(chunks):
            logger.error(f"Invalid chunk index {chunk_index} for chunks of length {len(chunks)}")
            raise ValueError(f"Chunk index {chunk_index} out of range [0, {len(chunks)-1}]")
            
        start_idx = max(0, chunk_index - context_size)
        end_idx = min(len(chunks), chunk_index + context_size + 1)
        return "\n".join(chunks[start_idx:end_idx]) 