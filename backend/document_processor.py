import fitz  # PyMuPDF
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
import os

class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

    def extract_metadata(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract metadata from PDF file
        """
        try:
            doc = fitz.open(pdf_path)
            metadata = doc.metadata
            return {
                "title": metadata.get("title", ""),
                "author": metadata.get("author", ""),
                "subject": metadata.get("subject", ""),
                "keywords": metadata.get("keywords", ""),
                "page_count": len(doc),
                "creation_date": metadata.get("creationDate", ""),
            }
        except Exception as e:
            print(f"Error extracting metadata: {e}")
            return {}

    def process_pdf(self, pdf_path: str) -> List[Document]:
        """
        Process a PDF file and return chunks as Langchain Document objects with metadata
        """
        try:
            # Load PDF using LangChain's PyMuPDFLoader
            loader = PyMuPDFLoader(pdf_path)
            pages = loader.load()
            
            # Split text into chunks (Langchain Documents)
            lc_documents = self.text_splitter.split_documents(pages)
            
            # Extract metadata and add to each document
            metadata = self.extract_metadata(pdf_path)
            for doc in lc_documents:
                 doc.metadata = {**doc.metadata, **metadata, "filename": os.path.basename(pdf_path)}
            
            return lc_documents
            
        except Exception as e:
            print(f"Error processing PDF: {e}")
            raise

    def get_chunk_with_context(self, chunks: List[str], chunk_index: int, context_size: int = 2) -> str:
        """
        Get a chunk with surrounding context
        """
        start_idx = max(0, chunk_index - context_size)
        end_idx = min(len(chunks), chunk_index + context_size + 1)
        return "\n".join(chunks[start_idx:end_idx]) 