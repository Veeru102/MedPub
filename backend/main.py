from fastapi import FastAPI, WebSocket, UploadFile, File, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Any
import os
from dotenv import load_dotenv
import json
from datetime import datetime
from pydantic import BaseModel
import logging
import faiss
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from document_processor import DocumentProcessor
from rag_engine import RAGEngine
from enhanced_document_processor import EnhancedDocumentProcessor
from llm_services import LLMService
from langchain_core.documents import Document # Import Document
from langchain_community.document_loaders import PyMuPDFLoader # Updated import

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Verify OpenAI API key is set
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    logger.error("OPENAI_API_KEY environment variable is not set")
    logger.error("Please set the OPENAI_API_KEY environment variable in your Render dashboard")
    logger.error("Go to: Dashboard > Service > Environment and add OPENAI_API_KEY")
    raise ValueError("OPENAI_API_KEY environment variable is not set")
else:
    logger.info(f"OpenAI API Key loaded successfully (starts with: {api_key[:10]}...)")
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

# Create uploads directory if it doesn't exist
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Stores Langchain Document objects with chunks and metadata
processed_documents: Dict[str, List[Document]] = {}
document_sections: Dict[str, Dict[str, Any]] = {}  # Store section information
doc_processor = DocumentProcessor()
enhanced_processor = EnhancedDocumentProcessor()
llm_service = LLMService()

# Initialize RAGEngine globally
rage_engine = RAGEngine()

# Define startup event to load the FAISS index
@app.on_event("startup")
async def startup_event():
    logger.info("Application startup: Loading FAISS index...")
    try:
        rage_engine.load_vector_store()
    except Exception as e:
        logger.error(f"Failed to load FAISS index during startup: {e}")


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

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF file for processing
    """
    if not file.filename.endswith('.pdf'):
        logger.warning(f"Invalid file type attempted: {file.filename}")
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    # Create a unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{file.filename}"
    file_path = os.path.join(UPLOAD_DIR, filename)
    
    # Save the file
    try:
        contents = await file.read()
        with open(file_path, "wb") as f:
            f.write(contents)
        logger.info(f"File saved successfully: {file_path}")
    except Exception as e:
        logger.error(f"Error saving file {filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Error saving file: {str(e)}")
    
    # Process the file into Langchain Documents
    try:
        logger.info(f"Processing file: {filename}")
        # Use enhanced processor for better chunking and section detection
        lc_documents, doc_info = enhanced_processor.process_pdf_enhanced(file_path)
        
        if not lc_documents:
            logger.warning(f"No documents extracted from {filename}")
            raise HTTPException(status_code=400, detail="Could not extract any content from the PDF")
            
        logger.info(f"Successfully processed {len(lc_documents)} chunks from {filename}")
        
        # Store processed documents and section information
        processed_documents[filename] = lc_documents
        document_sections[filename] = doc_info
        
        # Extract topics for clustering
        full_text = '\n'.join([doc.page_content for doc in lc_documents[:5]])  # Use first 5 chunks
        topics = await llm_service.extract_key_topics(full_text)
        doc_info['topics'] = topics
        
        # Update vector store with rate limiting
        try:
            all_processed_documents = []
            for doc_list in processed_documents.values():
                all_processed_documents.extend(doc_list)
            
            if all_processed_documents:
                logger.info(f"Updating vector store with {len(all_processed_documents)} documents...")
                rage_engine.create_vector_store_from_documents(all_processed_documents)
                rage_engine.setup_qa_chain()
                logger.info("Vector store updated successfully")
            
            return {"filename": filename, "message": "File uploaded and processed successfully"}
            
        except Exception as e:
            logger.error(f"Error updating vector store: {e}")
            # Even if vector store update fails, we keep the processed documents
            return {
                "filename": filename,
                "message": "File uploaded and processed, but vector store update failed. Some features may be limited.",
                "warning": str(e)
            }
            
    except Exception as e:
        logger.error(f"Error processing PDF {filename}: {e}")
        # Clean up the uploaded file if processing fails
        try:
            os.remove(file_path)
            logger.info(f"Cleaned up {file_path} after processing error")
        except OSError as cleanup_error:
            logger.error(f"Error cleaning up file {file_path}: {cleanup_error}")

        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time communication
    """
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
    """
    Summarize a specific paper with audience-specific formatting
    """
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
        # Get sections if available
        sections = doc_info.get('sections', {})
        
        # Use LLM service for audience-specific summary
        chunks_text = [doc.page_content for doc in document_chunks]
        full_text = '\n'.join(chunks_text)
        
        summary_result = await llm_service.generate_summary(
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
    """
    Delete a specific uploaded file
    """
    logger.info(f"Received delete request for file: {request.filename}")
    filename = request.filename
    file_path = os.path.join(UPLOAD_DIR, filename)

    if not os.path.exists(file_path):
        logger.warning(f"File not found for deletion: {file_path}")
        raise HTTPException(status_code=404, detail="File not found")

    try:
        os.remove(file_path)
        logger.info(f"File deleted successfully: {file_path}")
        
        # Remove from processed documents
        if filename in processed_documents:
            del processed_documents[filename]
            logger.info(f"Removed {filename} from processed documents")
            
        # Update vector store after deletion with rate limiting
        try:
            all_processed_documents = []
            for doc_list in processed_documents.values():
                all_processed_documents.extend(doc_list)
            
            if all_processed_documents:
                logger.info(f"Updating vector store after file deletion with {len(all_processed_documents)} documents...")
                rage_engine.create_vector_store_from_documents(all_processed_documents)
                rage_engine.setup_qa_chain()
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
    """
    Query the uploaded papers using RAG
    """
    logger.info(f"Received query request: {request.model_dump_json()}")
    query = request.query
    filenames = request.filenames
    
    if not filenames:
        raise HTTPException(status_code=400, detail="No papers specified for query")
        
    # The RAGEngine is already initialized globally and attempts to load the index on startup.
    # The setup_qa_chain method is now called within the query method if needed.
    # We no longer need to recreate the vector store here for each query.

    # Instead of gathering documents here, the RAGEngine's retriever will use the
    # globally loaded/updated vector store which contains data from all processed docs.
    # However, the RAGEngine query method currently doesn't filter by filename. 
    # To implement multi-document querying where the query is scoped to selected files,
    # we would need to either: 
    # 1. Pass the selected filenames to the RAGEngine.query method and modify it
    #    to filter the retrieval by source filename metadata.
    # 2. Create a temporary filtered vector store for each query based on selected files.
    # Approach 1 is generally more efficient.

    # For now, the RAGEngine.query method will query against the index of ALL uploaded documents.
    # We will update the RAGEngine query method later to filter by filenames if needed for true multi-doc querying.

    # Ensure the RAGEngine is initialized and attempts to load the vector store
    # The query method itself now handles calling setup_qa_chain.
    # No need to explicitly call setup_qa_chain here.

    # The filenames are used by the frontend to indicate context, 
    # but the current RAGEngine queries the combined index.
    # TODO: Enhance RAGEngine.query to use the filenames list for retrieval filtering.

    logger.info(f"Querying with filenames: {filenames}. Using global vector store.")
    
    # Explicitly invoke the chain and process the output
    try:
        chain_output = await rage_engine.qa_chain.ainvoke({"question": query})
        
        # The answer is in chain_output['answer'], sources are in chain_output['source_documents']

        response_answer = chain_output.get("answer", "Could not find an answer.")
        response_sources = [
            {
                "content": doc.page_content,
                "metadata": doc.metadata
            }
            for doc in chain_output.get("source_documents", [])
        ]
        
        return {"message": response_answer, "sources": response_sources}

    except Exception as e:
        logger.error(f"Error during RAG query: {e}")
        # If QA chain is not initialized (e.g., no documents processed and index loading failed),
        # the above ainvoke call would fail. Catching it here and returning a specific message.
        if not rage_engine.qa_chain:
             raise HTTPException(status_code=503, detail="RAG system not ready. Please upload and process a document.")
        else:
             # Re-raise other exceptions
             raise HTTPException(status_code=500, detail=f"Error during RAG query: {e}")

@app.post("/explanation")
async def get_sentence_explanation(request: ExplanationRequest):
    """
    Get source passages and confidence scores for a specific sentence in the summary
    """
    logger.info(f"Received explanation request for sentence in file: {request.filename}")
    filename = request.filename
    
    if filename not in processed_documents:
        logger.error(f"Processed data not found for file: {filename}")
        raise HTTPException(status_code=404, detail="Processed data not found for this file.")
    
    try:
        # Get the document chunks
        document_chunks = processed_documents[filename]
        
        # Get embeddings for the sentence
        sentence_embedding = await rage_engine.embeddings.aembed_query(request.sentence)
        sentence_embedding = np.array(sentence_embedding).reshape(1, -1)  # Reshape to 2D array
        
        # Calculate cosine similarity with all chunks
        similarities = []
        for doc in document_chunks:
            chunk_embedding = await rage_engine.embeddings.aembed_query(doc.page_content)
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
    """
    Query a specific document with citations to source sections
    """
    logger.info(f"Query document request: {request.question} for {request.document_id}")
    
    if request.document_id not in processed_documents:
        raise HTTPException(status_code=404, detail="Document not found")
    
    try:
        # Get relevant chunks from the specific document
        doc_chunks = processed_documents[request.document_id]
        
        # Use RAG engine to find relevant chunks
        query_embedding = await rage_engine.embeddings.aembed_query(request.question)
        
        # Calculate similarities and get top chunks
        similarities = []
        for doc in doc_chunks:
            chunk_embedding = await rage_engine.embeddings.aembed_query(doc.page_content)
            similarity = cosine_similarity(
                np.array(query_embedding).reshape(1, -1),
                np.array(chunk_embedding).reshape(1, -1)
            )[0][0]
            similarities.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "similarity": float(similarity)
            })
        
        # Get top 5 most relevant chunks
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        relevant_chunks = similarities[:5]
        
        # Generate answer with citations
        answer_result = await llm_service.answer_with_citations(
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
    """
    Explain highlighted text based on user question
    """
    logger.info(f"Explain text request for {request.filename}")
    
    if request.filename not in processed_documents:
        raise HTTPException(status_code=404, detail="Document not found")
    
    try:
        result = await llm_service.explain_text(
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
    """
    Synthesize findings across multiple papers
    """
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
        synthesis_result = await llm_service.synthesize_papers(
            papers_data=papers_data,
            synthesis_type=request.synthesis_type
        )
        
        return synthesis_result
        
    except Exception as e:
        logger.error(f"Error synthesizing papers: {e}")
        raise HTTPException(status_code=500, detail=f"Error synthesizing papers: {str(e)}")

@app.get("/document-info/{filename}")
async def get_document_info(filename: str):
    """
    Get detailed information about a processed document
    """
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
    """
    Find related documents based on topic similarity
    """
    if filename not in document_sections:
        raise HTTPException(status_code=404, detail="Document not found")
    
    target_topics = set(document_sections[filename].get("topics", []))
    
    if not target_topics:
        return {"related": []}
    
    related = []
    for other_filename, other_info in document_sections.items():
        if other_filename != filename:
            other_topics = set(other_info.get("topics", []))
            
            # Calculate topic overlap
            overlap = len(target_topics.intersection(other_topics))
            if overlap > 0:
                related.append({
                    "filename": other_filename,
                    "title": other_info.get("metadata", {}).get("title", other_filename),
                    "common_topics": list(target_topics.intersection(other_topics)),
                    "similarity_score": overlap / len(target_topics.union(other_topics))
                })
    
    # Sort by similarity score
    related.sort(key=lambda x: x["similarity_score"], reverse=True)
    
    return {"related": related[:5]}  # Return top 5 related documents

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Remove the print statement for faiss version as it's not needed here
# print(faiss.__version__) 