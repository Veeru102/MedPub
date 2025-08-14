from fastapi import FastAPI, WebSocket, UploadFile, File, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from typing import List, Optional, Dict, Any
from langchain_core.documents import Document
import os
from dotenv import load_dotenv
import json
from datetime import datetime
from pydantic import BaseModel
import logging
import numpy as np
import sys
import asyncio
import nltk # Added for NLTK path configuration

# Import arXiv router
from arxiv_search import router as arxiv_router

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Debugging information.
logger.info(f"Python path: {sys.path}")
logger.info(f"Current working directory: {os.getcwd()}")
try:
    import fitz
    logger.info("Successfully imported fitz")
except ImportError as e:
    logger.error(f"Failed to import fitz: {e}")

# Load environment var
load_dotenv(override=True)  

# Update ARXIV_LOAD_LIMIT to 2500000
os.environ["ARXIV_LOAD_LIMIT"] = "100000"




# Verify OpenAI API key is set
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    logger.error("OPENAI_API_KEY environment variable is not set")
    logger.error("Please set the OPENAI_API_KEY in your .env file")
    raise ValueError("OPENAI_API_KEY environment variable is not set")
else:
    # Only show first and last 4 char
    key_preview = f"{api_key[:4]}...{api_key[-4:]}"
    logger.info(f"OpenAI API Key loaded successfully (format: {key_preview})")
    logger.info(f"API Key length: {len(api_key)} characters")

app = FastAPI(title="MedCopilot API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "https://medscopefrontend.onrender.com"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include arXiv search router
app.include_router(arxiv_router)

# Create uploads directory if it doesn't exist
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Mount static files directory to serve PDFs
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

# Stores Langchain Document objects with chunks and metadata
processed_documents: Dict[str, List[Document]] = {}
document_sections: Dict[str, Dict[str, Any]] = {}  # Store section information

# Global variables for lazy loading
doc_processor = None
enhanced_processor = None
llm_service = None
rage_engine = None

def get_doc_processor():
    """Lazy load DocumentProcessor"""
    global doc_processor
    if doc_processor is None:
        from document_processor import DocumentProcessor
        doc_processor = DocumentProcessor()
    return doc_processor

def get_enhanced_processor():
    """Lazy load EnhancedDocumentProcessor"""
    global enhanced_processor
    if enhanced_processor is None:
        from enhanced_document_processor import EnhancedDocumentProcessor
        enhanced_processor = EnhancedDocumentProcessor()
    return enhanced_processor

def get_llm_service():
    """Lazy load LLMService"""
    global llm_service
    if llm_service is None:
        from llm_services import LLMService
        llm_service = LLMService()
    return llm_service

def get_rage_engine():
    """Lazy load RAGEngine"""
    global rage_engine
    if rage_engine is None:
        from rag_engine import RAGEngine
        rage_engine = RAGEngine()
    return rage_engine

# Minimal startup event - no heavy loading
@app.on_event("startup")
async def startup_event():
    logger.info("Application startup: Minimal initialization complete")
    logger.info("Heavy components will be loaded on-demand to save memory")


class SummarizeRequest(BaseModel):
    filename: str
    audience_type: Optional[str] = "clinician"  # patient, clinician, researcher

class ExplanationRequest(BaseModel):
    filename: str
    sentence: str

class ExplainTextRequest(BaseModel):
    filename: str
    selected_text: str
    context: str
    question: str
    audience_type: Optional[str] = "patient"

class QueryDocRequest(BaseModel):
    question: str
    document_id: str

class ChatRequest(BaseModel):
    question: str
    chat_history: Optional[List[Dict[str, str]]] = None  # List of {"human": "...", "ai": "..."}
    filenames: Optional[List[str]] = None

class SynthesizeRequest(BaseModel):
    filenames: List[str]
    synthesis_type: Optional[str] = "comparison"  # comparison, evolution, consensus, methods

class QueryRequest(BaseModel):
    query: str
    filenames: Optional[List[str]] = None

class DeleteRequest(BaseModel):
    filename: str

@app.get("/")
async def root():
    return {"message": "Welcome to MedCopilot API"}

@app.get("/files/{filename}")
async def serve_pdf(filename: str):
    """Serves PDF files from the uploads directory."""
    file_path = os.path.join(UPLOAD_DIR, filename)
    
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise HTTPException(status_code=404, detail="File not found")
    
    if not filename.lower().endswith('.pdf'):
        logger.error(f"Invalid file type requested: {filename}")
        raise HTTPException(status_code=400, detail="Only PDF files are served")
    
    try:
        return FileResponse(
            path=file_path,
            media_type='application/pdf',
            filename=filename
        )
    except Exception as e:
        logger.error(f"Error serving file {filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Error serving file: {str(e)}")

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Uploads a PDF file for processing."""
    if not file.filename.endswith('.pdf'):
        logger.warning(f"Invalid file type attempted: {file.filename}")
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    # Creates a unique filename.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{file.filename}"
    file_path = os.path.join(UPLOAD_DIR, filename)
    
    # Saves the file.
    try:
        contents = await file.read()
        with open(file_path, "wb") as f:
            f.write(contents)
        logger.info(f"File saved successfully: {file_path}")
    except Exception as e:
        logger.error(f"Error saving file {filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Error saving file: {str(e)}")
    
    # Processes the file into Langchain Documents.
    try:
        logger.info(f"Processing file: {filename}")
        # Use enhanced processor for better chunking and section detection
        lc_documents, doc_info = get_enhanced_processor().process_pdf_enhanced(file_path)
        
        if not lc_documents:
            logger.warning(f"No documents extracted from {filename}")
            raise HTTPException(status_code=400, detail="Could not extract any content from the PDF")
            
        logger.info(f"Successfully processed {len(lc_documents)} chunks from {filename}")
        
        # Stores processed documents and section information.
        processed_documents[filename] = lc_documents
        document_sections[filename] = doc_info
        
        # Extracts topics for clustering.
        full_text = '\n'.join([doc.page_content for doc in lc_documents[:5]])  # Uses first 5 chunks.
        topics = await get_llm_service().extract_key_topics(full_text)
        doc_info['topics'] = topics
        
        # Update vector store with rate limiting
        try:
            all_processed_documents = []
            for doc_list in processed_documents.values():
                all_processed_documents.extend(doc_list)
            
            if all_processed_documents:
                logger.info(f"Updating vector store with {len(all_processed_documents)} documents...")
                get_rage_engine().create_vector_store_from_documents(all_processed_documents)
                get_rage_engine().setup_qa_chain()
                logger.info("Vector store updated successfully")
            
            return {"filename": filename, "message": "File uploaded and processed successfully"}
            
        except Exception as e:
            logger.error(f"Error updating vector store: {e}")
            # Even if vector store update fails, processed documents are kept.
            return {
                "filename": filename,
                "message": "File uploaded and processed, but vector store update failed. Some features may be limited.",
                "warning": str(e)
            }
            
    except Exception as e:
        logger.error(f"Error processing PDF {filename}: {e}")
        # Cleans up the uploaded file if processing fails.
        try:
            os.remove(file_path)
            logger.info(f"Cleaned up {file_path} after processing error")
        except OSError as cleanup_error:
            logger.error(f"Error cleaning up file {file_path}: {cleanup_error}")

        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time communication."""
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            # Process the message and send response
            response = {"message": "Received your message", "data": data}
            await websocket.send_json(response)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await websocket.close()

@app.post("/summarize")
async def summarize_paper(request: SummarizeRequest):
    """Summarizes a specific paper with audience-specific formatting."""
    logger.info(f"Received summarize request for file: {request.filename}, audience: {request.audience_type}")
    filename = request.filename
    
    if filename not in processed_documents:
        logger.error(f"Processed data not found for file: {filename}")
        raise HTTPException(status_code=404, detail="Processed data not found for this file")
        
    document_chunks = processed_documents[filename]
    doc_info = document_sections.get(filename, {})
    
    if not document_chunks:
        logger.warning(f"No chunks found for file: {filename}")
        raise HTTPException(status_code=400, detail="No content available for summarization")
    
    try:
        # Gets sections if available.
        sections = doc_info.get('sections', {})
        
        # Uses LLM service for audience-specific summary.
        chunks_text = [doc.page_content for doc in document_chunks]
        full_text = '\n'.join(chunks_text)
        
        summary_result = await get_llm_service().generate_summary(
            text=full_text,
            audience=request.audience_type,
            sections=sections
        )
        
        if summary_result.get("status") == "error":
            logger.error(f'Summarization failed for {filename}: {summary_result.get("error")}')
            raise HTTPException(status_code=500, detail=f"Summarization failed: {summary_result.get('error')}")

        return {"message": summary_result.get("summary", "Summarization failed"), "audience": request.audience_type}
        
    except Exception as e:
        logger.error(f"Error summarizing {filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Error during summarization: {str(e)}")

@app.post("/delete_file")
async def delete_file(request: DeleteRequest):
    """Deletes a specific uploaded file."""
    logger.info(f"Received delete request for file: {request.filename}")
    filename = request.filename
    file_path = os.path.join(UPLOAD_DIR, filename)

    if not os.path.exists(file_path):
        logger.warning(f"File not found for deletion: {file_path}")
        raise HTTPException(status_code=404, detail="File not found")

    try:
        os.remove(file_path)
        logger.info(f"File deleted successfully: {file_path}")
        
        # Removes from processed documents.
        if filename in processed_documents:
            del processed_documents[filename]
            logger.info(f"Removed {filename} from processed documents")
            
        # Updates vector store after deletion with rate limiting.
        try:
            all_processed_documents = []
            for doc_list in processed_documents.values():
                all_processed_documents.extend(doc_list)
            
            if all_processed_documents:
                logger.info(f"Updating vector store after file deletion with {len(all_processed_documents)} documents...")
                get_rage_engine().create_vector_store_from_documents(all_processed_documents)
                get_rage_engine().setup_qa_chain()
                logger.info("Vector store updated successfully")
                
        except Exception as e:
            logger.error(f"Error updating vector store after deletion: {e}")
            return {
                "message": "File deleted, but vector store update failed. Some features may be limited.",
                "warning": str(e)
            }
            
        return {"message": "File deleted successfully"}
        
    except OSError as e:
        logger.error(f"Error deleting file {file_path}: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting file: {str(e)}")

@app.post("/query")
async def query_papers(request: QueryRequest):
    """Performs a single-turn QA query."""
    logger.info(f"Received QA query request: {request.query}")
    
    if not request.filenames:
        raise HTTPException(status_code=400, detail="No papers specified for query")
        
    try:
        # Uses the updated query method.
        result = await get_rage_engine().query(request.query)
        return {"message": result["answer"], "sources": result["sources"]}
        
    except Exception as e:
        logger.error(f"Error during QA query: {e}")
        if not get_rage_engine().qa_chain:
            raise HTTPException(status_code=503, detail="RAG system not ready. Please upload and process a document.")
        else:
            raise HTTPException(status_code=500, detail=f"Error during RAG query: {e}")

@app.post("/chat")
async def chat_with_documents(request: ChatRequest):
    """Performs multi-turn conversational chat with RAG."""
    logger.info(f"Received chat request: {request.question}")
    
    if not request.filenames:
        logger.warning("No filenames specified, using all available documents")
        # Uses all available documents if none specified.
        request.filenames = list(processed_documents.keys())
    
    try:
        # Combine documents from all selected files
        all_documents = []
        for filename in request.filenames:
            if filename in processed_documents:
                all_documents.extend(processed_documents[filename])
            else:
                logger.warning(f"Document not found: {filename}")
        
        if not all_documents:
            raise HTTPException(
                status_code=400,
                detail="No valid documents found for chat context"
            )
        
        # Update RAG engine with combined documents
        get_rage_engine().create_vector_store_from_documents(all_documents)
        
        # Use the chat method
        result = await get_rage_engine().chat(request.question, request.chat_history)
        
        return {
            "answer": result["answer"],
            "sources": result["sources"],
            "chat_history": result["chat_history"]
        }
        
    except Exception as e:
        logger.error(f"Error during chat: {e}")
        if not get_rage_engine().chat_chain:
            raise HTTPException(status_code=503, detail="Chat system not ready. Please upload and process a document.")
        else:
            raise HTTPException(status_code=500, detail=f"Error during chat: {e}")

@app.post("/chat/clear")
async def clear_chat_history():
    """Clears the conversation history."""
    try:
        get_rage_engine().clear_memory()
        return {"message": "Chat history cleared successfully"}
    except Exception as e:
        logger.error(f"Error clearing chat history: {e}")
        raise HTTPException(status_code=500, detail=f"Error clearing chat history: {e}")

@app.post("/explanation")
async def get_sentence_explanation(request: ExplanationRequest):
    """Gets source passages and confidence scores for a specific sentence."""
    logger.info(f"Received explanation request for sentence in file: {request.filename}")
    filename = request.filename
    
    if filename not in processed_documents:
        logger.error(f"Processed data not found for file: {filename}")
        raise HTTPException(status_code=404, detail="Processed data not found for this file.")
    
    try:
        # Get the document chunks
        document_chunks = processed_documents[filename]
        
        # Get embeddings for the sentence
        sentence_embedding = await get_rage_engine().embeddings.aembed_query(request.sentence)
        sentence_embedding = np.array(sentence_embedding).reshape(1, -1)  # Reshape to 2D array
        
        # Calculate cosine similarity with all chunks
        similarities = []
        for doc in document_chunks:
            chunk_embedding = await get_rage_engine().embeddings.aembed_query(doc.page_content)
            chunk_embedding = np.array(chunk_embedding).reshape(1, -1)  # Reshape to 2D array
            similarity = cosine_similarity(sentence_embedding, chunk_embedding)[0][0]  # Get scalar value
            similarities.append({
                "content": doc.page_content,
                "similarity": float(similarity),
                "metadata": doc.metadata
            })
        
        # Sort by similarity and get top 3 most relevant chunks
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        top_chunks = similarities[:3]
        
        # Calculate overall confidence (average of top similarities)
        confidence = sum(chunk["similarity"] for chunk in top_chunks) / len(top_chunks) if top_chunks else 0
        
        return {
            "source_chunks": top_chunks,
            "confidence": confidence
        }
        
    except Exception as e:
        logger.error(f"Error getting sentence explanation: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting sentence explanation: {e}")

@app.post("/query-doc")
async def query_document(request: QueryDocRequest):
    """Queries a specific document with citations to source sections."""
    logger.info(f"Query document request: {request.question} for {request.document_id}")
    
    if request.document_id not in processed_documents:
        raise HTTPException(status_code=404, detail="Document not found")
    
    try:
        # Gets relevant chunks from the specific document.
        doc_chunks = processed_documents[request.document_id]
        
        # Uses RAG engine to find relevant chunks.
        query_embedding = await get_rage_engine().embeddings.aembed_query(request.question)
        
        # Calculates similarities and gets top chunks.
        similarities = []
        for doc in doc_chunks:
            chunk_embedding = await get_rage_engine().embeddings.aembed_query(doc.page_content)
            similarity = cosine_similarity(
                np.array(query_embedding).reshape(1, -1),
                np.array(chunk_embedding).reshape(1, -1)
            )[0][0]
            similarities.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "similarity": float(similarity)
            })
        
        # Gets top 5 most relevant chunks.
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        relevant_chunks = similarities[:5]
        
        # Generates answer with citations.
        answer_result = await get_llm_service().answer_with_citations(
            question=request.question,
            relevant_chunks=relevant_chunks
        )
        
        return {
            "answer": answer_result.get("answer", ""),
            "citations": answer_result.get("citations", []),
            "document_id": request.document_id
        }
        
    except Exception as e:
        logger.error(f"Error querying document: {e}")
        raise HTTPException(status_code=500, detail=f"Error querying document: {str(e)}")

@app.post("/explain-text")
async def explain_highlighted_text(request: ExplainTextRequest):
    """Explains highlighted text based on user question."""
    logger.info(f"Explain text request for {request.filename}")
    
    if request.filename not in processed_documents:
        raise HTTPException(status_code=404, detail="Document not found")
    
    try:
        result = await get_llm_service().explain_text(
            selected_text=request.selected_text,
            context=request.context,
            user_question=request.question,
            audience=request.audience_type
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error explaining text: {e}")
        raise HTTPException(status_code=500, detail=f"Error explaining text: {str(e)}")

@app.post("/synthesize-topic")
async def synthesize_topic(request: SynthesizeRequest):
    """Synthesizes findings across multiple papers."""
    logger.info(f"Synthesize request for {len(request.filenames)} files")
    
    papers_data = []
    for filename in request.filenames:
        if filename in processed_documents and filename in document_sections:
            doc_info = document_sections[filename]
            
            # Extract key info from each paper
            paper_data = {
                "title": doc_info.get("metadata", {}).get("title", filename),
                "year": doc_info.get("metadata", {}).get("creation_date", "Unknown"),
                "topics": doc_info.get("topics", []),
                "sections": doc_info.get("sections", {})
            }
            
            # Extract findings and methods from sections
            sections = doc_info.get("sections", {})
            for section_name, section_content in sections.items():
                if "finding" in section_name.lower() or "result" in section_name.lower():
                    paper_data["findings"] = section_content[:1000]
                elif "method" in section_name.lower():
                    paper_data["methods"] = section_content[:1000]
            
            papers_data.append(paper_data)
    
    if not papers_data:
        raise HTTPException(status_code=400, detail="No valid papers found for synthesis")
    
    try:
        synthesis_result = await get_llm_service().synthesize_papers(
            papers_data=papers_data,
            synthesis_type=request.synthesis_type
        )
        
        return synthesis_result
        
    except Exception as e:
        logger.error(f"Error synthesizing papers: {e}")
        raise HTTPException(status_code=500, detail=f"Error synthesizing papers: {str(e)}")

@app.get("/document-info/{filename}")
async def get_document_info(filename: str):
    """Gets detailed information about a processed document."""
    if filename not in document_sections:
        raise HTTPException(status_code=404, detail="Document information not found")
    
    doc_info = document_sections[filename]
    
    return {
        "filename": filename,
        "metadata": doc_info.get("metadata", {}),
        "sections": list(doc_info.get("sections", {}).keys()),
        "topics": doc_info.get("topics", []),
        "total_chunks": doc_info.get("total_chunks", 0),
        "chunking_method": doc_info.get("chunking_method", "unknown")
    }

@app.get("/related-documents/{filename}")
async def get_related_documents(filename: str):
    """Gets related documents based on similarity to the given document."""
    if filename not in processed_documents:
        raise HTTPException(status_code=404, detail="Document not found")
    
    try:
        import re
        # Get topics for the document
        doc_info = document_sections.get(filename, {})
        topics = doc_info.get('topics', [])
        
        # Find similar documents based on topics
        related_docs = []
        for other_filename, other_doc_info in document_sections.items():
            if other_filename == filename:
                continue
            
            other_topics = other_doc_info.get('topics', [])
            if not other_topics:
                continue
                
            # Calculate topic similarity (simple intersection)
            common_topics = list(set(topics) & set(other_topics))
            if common_topics:
                similarity_score = len(common_topics) / len(set(topics + other_topics))
                related_docs.append({
                    'filename': other_filename,
                    'title': other_doc_info.get('title', re.sub(r'^\d{8}_\d{6}_', '', other_filename)),
                    'topics': other_topics,
                    'common_topics': common_topics,
                    'similarity_score': similarity_score
                })
        
        # Sort by similarity and return top 5
        related_docs.sort(key=lambda x: x['similarity_score'], reverse=True)
        return {"related": related_docs[:5]}
        
    except Exception as e:
        logger.error(f"Error getting related documents for {filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Error finding related documents: {str(e)}")

@app.get("/similar-papers/{filename}")
async def get_similar_papers(filename: str, limit: int = 3):
    """Gets similar arXiv papers for an uploaded PDF."""
    logger.info(f"Getting similar papers for: {filename}")
    
    if filename not in processed_documents:
        raise HTTPException(status_code=404, detail="Document not found")
    
    try:
        # Extract text content from processed document chunks
        document_chunks = processed_documents[filename]
        if not document_chunks:
            raise HTTPException(status_code=400, detail="No content available for the document")
        
        # Use first few chunks or summary-like content for better matching
        # Take first 5 chunks to get representative content without overwhelming the search
        chunks_text = [doc.page_content for doc in document_chunks[:5]]
        search_text = '\n'.join(chunks_text)
        
        # Limit search text length to avoid API limits (around 1000-2000 characters)
        if len(search_text) > 2000:
            search_text = search_text[:2000]
        
        logger.info(f"Using {len(search_text)} characters for similarity search")
    
        # Import and call the existing arXiv search function
        from arxiv_search import search_similar_papers
        from arxiv_search import arxiv_state
        
        # Check if arXiv search is initialized
        if not arxiv_state.is_initialized:
            raise HTTPException(
                status_code=503, 
                detail="arXiv search system not ready. Please try again in a moment."
            )
        
        # Search for similar papers
        results_df = search_similar_papers(
            index=arxiv_state.faiss_index,
            metadata_df=arxiv_state.metadata_df,
            query_text=search_text,
            model=arxiv_state.sentence_model,
            k=limit
        )
        
        # Format results
        papers = []
        for _, row in results_df.iterrows():
            paper = {
                "id": row.get("id", ""),
                "title": row.get("title", ""),
                "abstract": row.get("abstract", ""),
                "similarity_score": float(row.get("similarity_score", 0.0)),
                "rank": int(row.get("rank", 0)),
                "arxiv_url": f"https://arxiv.org/abs/{row.get('id', '')}" if row.get("id") else None
            }
            papers.append(paper)
        
        logger.info(f"Found {len(papers)} similar papers for {filename}")
    
        return {
            "papers": papers,
            "total_found": len(papers),
            "filename": filename,
            "query_length": len(search_text)
        }
        
    except Exception as e:
        logger.error(f"Error finding similar papers for {filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Error finding similar papers: {str(e)}")

@app.get("/debug-chunks/{filename}")
async def debug_chunks(filename: str, start_idx: Optional[int] = 0, limit: Optional[int] = 5):
    """Debug endpoint to inspect document chunks."""
    if filename not in processed_documents:
        raise HTTPException(status_code=404, detail="Document not found")
        
    chunks = processed_documents[filename]
    total_chunks = len(chunks)
    
    # Get requested slice of chunks
    end_idx = min(start_idx + limit, total_chunks)
    selected_chunks = chunks[start_idx:end_idx]
    
    return {
        "total_chunks": total_chunks,
        "showing_chunks": f"{start_idx} to {end_idx-1}",
        "chunks": [
            {
                "content": chunk.page_content,
                "metadata": chunk.metadata,
                "length": len(chunk.page_content),
                "sentences": len(chunk.page_content.split('.'))
            }
            for chunk in selected_chunks
        ]
    }

@app.get("/debug-embeddings/{filename}")
async def debug_embeddings(filename: str, query: Optional[str] = None):
    """Debug endpoint to inspect embeddings and similarity scores."""
    if filename not in processed_documents:
        raise HTTPException(status_code=404, detail="Document not found")
        
    if not get_rage_engine().vector_store:
        raise HTTPException(status_code=400, detail="Vector store not initialized")
    
    try:
        # If query provided, show similarity to chunks
        if query:
            docs_and_scores = get_rage_engine().vector_store.similarity_search_with_score(query, k=5)
            return {
                "query": query,
                "results": [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "similarity_score": float(score)  # Convert numpy float to Python float
                    }
                    for doc, score in docs_and_scores
                ]
            }
        
        # Otherwise show general embedding stats
        return {
            "total_embeddings": get_rage_engine().vector_store._collection.count(),
            "embedding_dimension": get_rage_engine().vector_store._collection.dim,
            "index_type": "FAISS"
        }
        
    except Exception as e:
        logger.error(f"Error inspecting embeddings: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/debug-index-health")
async def check_index_health():
    """Checks FAISS index health and stats."""
    if not get_rage_engine().vector_store:
        raise HTTPException(status_code=400, detail="Vector store not initialized")
        
    try:
        # Get index stats
        index = get_rage_engine().vector_store.index
        
        # Basic health check - try a random query
        test_query = "This is a test query"
        test_embedding = get_rage_engine().embeddings.embed_query(test_query)
        
        # Try search
        D, I = index.search(np.array([test_embedding], dtype=np.float32), k=1)
        
        return {
            "status": "healthy",
            "total_vectors": index.ntotal,
            "dimension": index.d,
            "is_trained": index.is_trained,
            "test_search_successful": bool(len(D) > 0 and len(I) > 0),
            "index_path": get_rage_engine().faiss_index_path,
            "index_file_exists": os.path.exists(get_rage_engine().faiss_index_path)
        }
        
    except Exception as e:
        logger.error(f"Index health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@app.post("/debug-retrieval")
async def debug_retrieval(query: str, k: Optional[int] = 3):
    """Debug endpoint to inspect retrieval results."""
    if not get_rage_engine().vector_store:
        raise HTTPException(status_code=400, detail="Vector store not initialized")
        
    try:
        # Get raw retrieval results
        docs = get_rage_engine().vector_store.similarity_search_with_score(query, k=k)
        
        # Format results
        results = []
        for doc, score in docs:
            results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "similarity_score": float(score),
                "source_file": doc.metadata.get("filename", "unknown"),
                "section": doc.metadata.get("section", "unknown"),
                "chunk_index": doc.metadata.get("chunk_index", -1)
            })
            
        return {
            "query": query,
            "num_results": len(results),
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Retrieval debug failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000)) # Gets port from environment variable or defaults to 8000
    uvicorn.run(app, host="0.0.0.0", port=port)

