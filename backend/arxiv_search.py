import asyncio
import logging
from typing import List, Dict, Any, Optional
import pandas as pd
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import threading
from datetime import datetime
from pathlib import Path

from arxiv_loader import load_arxiv_metadata
from arxiv_indexer import build_faiss_index, search_similar_papers

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Request models
class ArxivSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Search query text")
    limit: Optional[int] = Field(5, ge=1, le=20, description="Number of results to return")

class ArxivSearchResponse(BaseModel):
    papers: List[Dict[str, Any]]
    total_found: int
    search_time_ms: float
    query: str

class ArxivStatusResponse(BaseModel):
    status: str
    total_papers: int
    index_ready: bool
    last_updated: Optional[str]

# Global state for arXiv search
class ArxivSearchState:
    def __init__(self):
        self.faiss_index: Optional[faiss.IndexFlatIP] = None
        self.metadata_df: Optional[pd.DataFrame] = None
        self.sentence_model: Optional[SentenceTransformer] = None
        self.is_initialized = False
        self.is_loading = False
        self.last_updated: Optional[str] = None
        self.total_papers = 0
        self.lock = threading.Lock()

# Global instance
arxiv_state = ArxivSearchState()

# FastAPI router
router = APIRouter(prefix="/arxiv", tags=["arxiv"])

async def initialize_arxiv_search(force_reload: bool = False, force_rebuild_index: bool = False):
    """Initializes the arXiv search system with caching support.
    
    Args:
        force_reload: If True, bypass metadata cache and reload from Kaggle
        force_rebuild_index: If True, bypass FAISS index cache and rebuild
    """
    global arxiv_state
    
    with arxiv_state.lock:
        if arxiv_state.is_initialized or arxiv_state.is_loading:
            logger.info("ArXiv search already initialized or loading")
            return
        
        arxiv_state.is_loading = True
        logger.info("Starting arXiv search initialization...")
    
    try:
        start_time = pd.Timestamp.now()
        
        # Load arXiv metadata (with caching)
        if force_reload:
            logger.info("Force reload requested - bypassing metadata cache...")
        else:
            logger.info("Loading arXiv metadata (checking cache first)...")
            
        metadata_df = load_arxiv_metadata(force_reload=force_reload)
        
        if metadata_df.empty:
            logger.warning("No arXiv metadata loaded")
            return
        
        metadata_load_time = (pd.Timestamp.now() - start_time).total_seconds()
        logger.info(f"Loaded {len(metadata_df)} arXiv papers in {metadata_load_time:.2f}s")
        
        # Build FAISS index (with caching)
        index_start_time = pd.Timestamp.now()
        if force_rebuild_index:
            logger.info("Force rebuild requested - bypassing FAISS index cache...")
        else:
            logger.info("Building/loading FAISS index (checking cache first)...")
            
        faiss_index, processed_df = build_faiss_index(metadata_df, force_rebuild=force_rebuild_index)
        
        index_build_time = (pd.Timestamp.now() - index_start_time).total_seconds()
        logger.info(f"FAISS index ready with {faiss_index.ntotal} vectors in {index_build_time:.2f}s")
        
        # Load sentence transformer model
        model_start_time = pd.Timestamp.now()
        logger.info("Loading SentenceTransformer model...")
        sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
        model_load_time = (pd.Timestamp.now() - model_start_time).total_seconds()
        logger.info(f"SentenceTransformer model loaded in {model_load_time:.2f}s")
        
        # Update global state
        with arxiv_state.lock:
            arxiv_state.faiss_index = faiss_index
            arxiv_state.metadata_df = processed_df
            arxiv_state.sentence_model = sentence_model
            arxiv_state.is_initialized = True
            arxiv_state.is_loading = False
            arxiv_state.last_updated = datetime.now().isoformat()
            arxiv_state.total_papers = len(processed_df)
        
        total_time = (pd.Timestamp.now() - start_time).total_seconds()
        logger.info(f"ArXiv search initialized successfully with {len(processed_df)} papers in {total_time:.2f}s total")
        logger.info(f"Performance breakdown: metadata={metadata_load_time:.2f}s, index={index_build_time:.2f}s, model={model_load_time:.2f}s")
        
    except Exception as e:
        logger.error(f"Failed to initialize arXiv search: {e}")
        with arxiv_state.lock:
            arxiv_state.is_loading = False
        raise

@router.post("/search", response_model=ArxivSearchResponse)
async def search_arxiv_papers(request: ArxivSearchRequest):
    """Searches for similar arXiv papers using semantic similarity."""
    logger.info(f"ArXiv search request: '{request.query}' (limit: {request.limit})")
    
    # Check if system is initialized
    if not arxiv_state.is_initialized:
        if arxiv_state.is_loading:
            raise HTTPException(
                status_code=503, 
                detail="ArXiv search system is still loading. Please try again in a moment."
            )
        else:
            raise HTTPException(
                status_code=503, 
                detail="ArXiv search system not initialized. Please contact administrator."
            )
    
    # Validate query
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        start_time = pd.Timestamp.now()
        
        # Search for similar papers
        results_df = search_similar_papers(
            index=arxiv_state.faiss_index,
            metadata_df=arxiv_state.metadata_df,
            query_text=request.query.strip(),
            model=arxiv_state.sentence_model,
            k=request.limit
        )
        
        # Format results
        papers = []
        for _, row in results_df.iterrows():
            paper = {
                "id": row.get("id", ""),
                "title": row.get("title", ""),
                "abstract": row.get("abstract", ""),
                "similarity_score": row.get("similarity_score", 0.0),
                "rank": row.get("rank", 0),
                "arxiv_url": f"https://arxiv.org/abs/{row.get('id', '')}" if row.get("id") else None
            }
            papers.append(paper)
        
        search_time_ms = (pd.Timestamp.now() - start_time).total_seconds() * 1000
        
        logger.info(f"ArXiv search completed in {search_time_ms:.2f}ms, found {len(papers)} results")
        
        return ArxivSearchResponse(
            papers=papers,
            total_found=len(papers),
            search_time_ms=search_time_ms,
            query=request.query
        )
        
    except Exception as e:
        logger.error(f"Error during arXiv search: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@router.get("/status", response_model=ArxivStatusResponse)
async def get_arxiv_status():
    """Gets the current status of the arXiv search system."""
    return ArxivStatusResponse(
        status="ready" if arxiv_state.is_initialized else ("loading" if arxiv_state.is_loading else "not_initialized"),
        total_papers=arxiv_state.total_papers,
        index_ready=arxiv_state.is_initialized,
        last_updated=arxiv_state.last_updated
    )

@router.post("/initialize")
async def trigger_initialization(background_tasks: BackgroundTasks, force_reload: bool = False, force_rebuild_index: bool = False):
    """Manually triggers arXiv search initialization with optional cache bypass.
    
    Args:
        force_reload: If True, bypass metadata cache and reload from Kaggle
        force_rebuild_index: If True, bypass FAISS index cache and rebuild
    """
    if arxiv_state.is_initialized:
        return {
            "message": "ArXiv search already initialized",
            "total_papers": arxiv_state.total_papers,
            "last_updated": arxiv_state.last_updated
        }
    
    if arxiv_state.is_loading:
        return {"message": "ArXiv search initialization already in progress"}
    
    # Run initialization in background
    background_tasks.add_task(
        initialize_arxiv_search,
        force_reload=force_reload,
        force_rebuild_index=force_rebuild_index
    )
    
    cache_status = []
    if force_reload:
        cache_status.append("bypassing metadata cache")
    if force_rebuild_index:
        cache_status.append("bypassing FAISS index cache")
    
    status_msg = "ArXiv search initialization started"
    if cache_status:
        status_msg += f" ({', '.join(cache_status)})"
    
    return {"message": status_msg}

@router.post("/reinitialize")
async def force_reinitialize(background_tasks: BackgroundTasks):
    """Forces complete reinitialization, bypassing all caches."""
    
    # Reset state
    with arxiv_state.lock:
        arxiv_state.is_initialized = False
        arxiv_state.is_loading = False
        arxiv_state.faiss_index = None
        arxiv_state.metadata_df = None
        arxiv_state.sentence_model = None
        arxiv_state.total_papers = 0
        arxiv_state.last_updated = None
    
    # Run full reinitialization
    background_tasks.add_task(
        initialize_arxiv_search,
        force_reload=True,
        force_rebuild_index=True
    )
    
    return {"message": "Full reinitialization started (bypassing all caches)"}

@router.get("/status")
async def get_initialization_status():
    """Gets the current initialization status and cache information."""
    
    # Check cache file existence
    cache_dir = Path(__file__).parent / "arxiv_cache"
    metadata_cache = cache_dir / "arxiv_metadata.parquet"
    
    faiss_cache_dir = Path(__file__).parent / "faiss_index"
    index_cache = faiss_cache_dir / "arxiv_index.index"
    faiss_metadata_cache = faiss_cache_dir / "metadata.parquet"
    
    cache_info = {
        "metadata_cache_exists": metadata_cache.exists(),
        "metadata_cache_path": str(metadata_cache) if metadata_cache.exists() else None,
        "faiss_index_cache_exists": index_cache.exists(),
        "faiss_metadata_cache_exists": faiss_metadata_cache.exists(),
        "faiss_cache_dir": str(faiss_cache_dir)
    }
    
    # Add file sizes if they exist
    if metadata_cache.exists():
        cache_info["metadata_cache_size_mb"] = round(metadata_cache.stat().st_size / (1024*1024), 2)
    if index_cache.exists():
        cache_info["faiss_index_size_mb"] = round(index_cache.stat().st_size / (1024*1024), 2)
    
    return {
        "is_initialized": arxiv_state.is_initialized,
        "is_loading": arxiv_state.is_loading,
        "total_papers": arxiv_state.total_papers,
        "last_updated": arxiv_state.last_updated,
        "has_faiss_index": arxiv_state.faiss_index is not None,
        "has_metadata": arxiv_state.metadata_df is not None,
        "has_model": arxiv_state.sentence_model is not None,
        "cache_info": cache_info
    }

# Function to be called during FastAPI startup
async def startup_arxiv_search():
    """Startup function to initialize arXiv search system."""
    logger.info("Starting arXiv search initialization during app startup...")
    
    # Run initialization in background to avoid blocking app startup
    asyncio.create_task(initialize_arxiv_search())
    
    logger.info("ArXiv search initialization task created") 