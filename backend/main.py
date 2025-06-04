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
from langchain_core.documents import Document # Import Document
from langchain_community.document_loaders import PyMuPDFLoader # Updated import

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

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

# In-memory storage for processed documents (for demonstration)
# Stores Langchain Document objects with chunks and metadata
processed_documents: Dict[str, List[Document]] = {}
doc_processor = DocumentProcessor()
# Initialize RAGEngine globally to persist the vector store across requests
rage_engine = RAGEngine()

# Define startup event to load the FAISS index
@app.on_event("startup")
async def startup_event():
    logger.info("Application startup: Loading FAISS index...")
    rage_engine.load_vector_store()


class SummarizeRequest(BaseModel):
    filename: str

class ExplanationRequest(BaseModel):
    filename: str
    sentence: str

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
        logger.error(f"Error saving file: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    # Process the file into Langchain Documents
    try:
        # The process_pdf method in DocumentProcessor now returns a list of Documents
        lc_documents = doc_processor.process_pdf(file_path)
        
        # Store processed documents and update the global RAGEngine's vector store
        processed_documents[filename] = lc_documents
        
        # Pass the new documents to the RAGEngine to update the vector store
        # The RAGEngine should handle adding these to the existing index or creating a new one
        # For simplicity now, we recreate the index with all current documents. 
        # A more efficient approach would be incremental indexing.
        all_processed_documents = []
        for doc_list in processed_documents.values():
             all_processed_documents.extend(doc_list)
        
        if all_processed_documents:
             rage_engine.create_vector_store_from_documents(all_processed_documents)
             # Ensure the QA chain is updated with the new vector store
             rage_engine.setup_qa_chain()
        
        logger.info(f"File processed successfully: {filename} with {len(lc_documents)} chunks. Vector store updated.")
        return {"filename": filename, "message": "File uploaded and processed successfully"}
        
    except Exception as e:
        logger.error(f"Error processing PDF or updating vector store: {e}")
        # Clean up the uploaded file if processing fails
        try:
            os.remove(file_path)
            logger.info(f"Cleaned up {file_path} after processing error.")
        except OSError as cleanup_error:
            logger.error(f"Error cleaning up file {file_path}: {cleanup_error}")

        raise HTTPException(status_code=500, detail=f"Error processing PDF or updating vector store: {e}")

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
    Summarize a specific paper
    """
    logger.info(f"Received summarize request for file: {request.filename}")
    filename = request.filename
    
    if filename not in processed_documents:
        logger.error(f"Processed data not found for file: {filename}")
        raise HTTPException(status_code=404, detail="Processed data not found for this file.")
        
    # Get chunks for the specific document
    document_chunks = processed_documents[filename]
    
    # Initialize RAGEngine and summarize
    rage_engine = RAGEngine()
    # Pass the page_content of each Document object to the summarizer
    chunks_text = [doc.page_content for doc in document_chunks]
    summary_result = await rage_engine.summarize_paper(chunks_text)
    
    if summary_result.get("status") == "error":
        logger.error(f'Summarization failed for {filename}: {summary_result.get("error")}')
        raise HTTPException(status_code=500, detail=f"Summarization failed: {summary_result.get('error')}")

    return {"message": summary_result.get("summary", "Summarization failed.")}

@app.post("/delete_file")
async def delete_file(request: DeleteRequest):
    """
    Delete a specific uploaded file
    """
    logger.info(f"Received delete request for file: {request.filename}")
    filename = request.filename
    file_path = os.path.join(UPLOAD_DIR, filename)

    # Check if the file exists
    if not os.path.exists(file_path):
        logger.warning(f"File not found for deletion: {file_path}")
        raise HTTPException(status_code=404, detail="File not found.")

    # Delete the file from the filesystem
    try:
        os.remove(file_path)
        logger.info(f"File deleted successfully: {file_path}")
    except OSError as e:
        logger.error(f"Error deleting file {file_path}: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting file: {e}")

    # Remove the processed document data from in-memory storage
    if filename in processed_documents:
        del processed_documents[filename]
        logger.info(f"Removed processed data for: {filename}")

    return {"message": "File deleted successfully"}

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Remove the print statement for faiss version as it's not needed here
# print(faiss.__version__) 