import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
import logging
import pickle
from typing import Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def build_faiss_index(dataframe: pd.DataFrame, force_rebuild: bool = False) -> Tuple[faiss.IndexFlatIP, pd.DataFrame]:
    """Builds a FAISS similarity index from an arXiv metadata DataFrame with caching support.
    
    Args:
        dataframe: DataFrame with columns ['id', 'title', 'abstract']
        force_rebuild: If True, bypass cache and rebuild from scratch
        
    Returns:
        Tuple of (FAISS index, cleaned metadata DataFrame)
    """
    
    if dataframe.empty:
        logger.warning("Empty DataFrame provided")
        # Return empty index and DataFrame
        empty_index = faiss.IndexFlatIP(384)  # MiniLM embedding dimension
        return empty_index, dataframe.copy()
    
    # Validate required columns
    required_columns = ['title', 'abstract']
    missing_columns = [col for col in required_columns if col not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Setup cache paths
    cache_dir = Path(__file__).parent / "faiss_index"
    cache_dir.mkdir(exist_ok=True)
    
    index_file = cache_dir / "arxiv_index.index"
    metadata_file = cache_dir / "metadata.parquet"
    embeddings_file = cache_dir / "embeddings.npy"
    
    # Try to load from cache first (unless force_rebuild is True)
    if not force_rebuild and _cache_exists(index_file, metadata_file):
        try:
            logger.info("Loading FAISS index from cache...")
            cached_index, cached_metadata = _load_from_cache(index_file, metadata_file, embeddings_file)
            
            # Verify cache is compatible with current data
            if len(cached_metadata) == len(dataframe):
                logger.info(f"Successfully loaded FAISS index from cache with {cached_index.ntotal} vectors")
                return cached_index, cached_metadata
            else:
                logger.warning(f"Cache size mismatch: cached={len(cached_metadata)}, current={len(dataframe)}. Rebuilding...")
        except Exception as e:
            logger.warning(f"Failed to load cached FAISS index ({e}), rebuilding...")
    else:
        if force_rebuild:
            logger.info("Force rebuild requested, bypassing cache...")
        else:
            logger.info("FAISS index cache not found, building from scratch...")
    
    # Build index from scratch
    try:
        index, clean_df, embeddings = _build_faiss_index_from_scratch(dataframe)
        
        # Save to cache for future use
        try:
            _save_to_cache(index, clean_df, embeddings, index_file, metadata_file, embeddings_file)
            logger.info(f"Cached FAISS index with {index.ntotal} vectors to {cache_dir}")
        except Exception as e:
            logger.warning(f"Failed to save FAISS index cache: {e}")
        
        return index, clean_df
        
    except Exception as e:
        logger.error(f"Failed to build FAISS index: {e}")
        # If we have a cache as fallback, try to use it
        if _cache_exists(index_file, metadata_file):
            logger.warning("Attempting to use cached index as fallback...")
            try:
                return _load_from_cache(index_file, metadata_file, embeddings_file)
            except Exception as cache_e:
                logger.error(f"Cache fallback also failed: {cache_e}")
        raise

def _cache_exists(index_file: Path, metadata_file: Path) -> bool:
    """Check if all required cache files exist."""
    return index_file.exists() and metadata_file.exists()

def _load_from_cache(index_file: Path, metadata_file: Path, embeddings_file: Path) -> Tuple[faiss.IndexFlatIP, pd.DataFrame]:
    """Load FAISS index and metadata from cache files."""
    # Load FAISS index
    index = faiss.read_index(str(index_file))
    
    # Load metadata
    metadata_df = pd.read_parquet(metadata_file)
    
    # Optionally load embeddings (for debugging/verification)
    if embeddings_file.exists():
        try:
            embeddings = np.load(embeddings_file)
            logger.debug(f"Loaded embeddings with shape {embeddings.shape}")
        except Exception as e:
            logger.debug(f"Could not load embeddings from cache: {e}")
    
    return index, metadata_df

def _save_to_cache(index: faiss.IndexFlatIP, metadata_df: pd.DataFrame, embeddings: np.ndarray,
                   index_file: Path, metadata_file: Path, embeddings_file: Path):
    """Save FAISS index, metadata, and embeddings to cache files."""
    # Save FAISS index
    faiss.write_index(index, str(index_file))
    
    # Save metadata
    metadata_df.to_parquet(metadata_file, index=False)
    
    # Save embeddings
    np.save(embeddings_file, embeddings)

def _build_faiss_index_from_scratch(dataframe: pd.DataFrame) -> Tuple[faiss.IndexFlatIP, pd.DataFrame, np.ndarray]:
    """Build FAISS index from scratch and return index, metadata, and embeddings."""
    
    logger.info(f"Building FAISS index for {len(dataframe)} papers")
    
    # Clean the data and prepare text for embedding
    clean_df = dataframe.copy()
    clean_df['title'] = clean_df['title'].fillna('')
    clean_df['abstract'] = clean_df['abstract'].fillna('')
    
    # Concatenate title and abstract
    combined_texts = []
    for _, row in clean_df.iterrows():
        # Combine title and abstract with a separator
        combined_text = f"{row['title']} {row['abstract']}"
        combined_texts.append(combined_text.strip())
    
    logger.info("Loading SentenceTransformer model...")
    
    # Load the sentence transformer model
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    logger.info("Generating embeddings...")
    
    # Generate embeddings for all texts
    embeddings = model.encode(
        combined_texts,
        convert_to_numpy=True,
        show_progress_bar=True,
        batch_size=32
    )
    
    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)
    
    logger.info(f"Generated {embeddings.shape[0]} embeddings with dimension {embeddings.shape[1]}")
    
    # Create FAISS index (Inner Product for normalized vectors = cosine similarity)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    
    # Add embeddings to index
    index.add(embeddings.astype(np.float32))
    
    logger.info(f"FAISS index built successfully with {index.ntotal} vectors")
    
    # Return index, corresponding metadata, and embeddings
    return index, clean_df.reset_index(drop=True), embeddings


def search_similar_papers(index: faiss.IndexFlatIP, 
                         metadata_df: pd.DataFrame, 
                         query_text: str, 
                         model: SentenceTransformer = None, 
                         k: int = 10) -> pd.DataFrame:
    """Searches for similar papers using the FAISS index."""
    
    if model is None:
        logger.info("Loading SentenceTransformer model for search...")
        model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Generate query embedding
    query_embedding = model.encode([query_text], convert_to_numpy=True)
    faiss.normalize_L2(query_embedding)
    
    # Search for similar papers
    similarities, indices = index.search(query_embedding.astype(np.float32), k)
    
    # Prepare results
    results = []
    for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
        if idx < len(metadata_df):
            paper_info = metadata_df.iloc[idx].to_dict()
            paper_info['similarity_score'] = float(similarity)
            paper_info['rank'] = i + 1
            results.append(paper_info)
    
    return pd.DataFrame(results) 