import pandas as pd
import tempfile
import os


from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_arxiv_metadata(force_reload: bool = False) -> pd.DataFrame:
    """Downloads and parses the arXiv metadata file from Kaggle with caching support.
    
    Args:
        force_reload: If True, bypass cache and reload from Kaggle
        
    Returns:
        pd.DataFrame: Cleaned arXiv metadata with columns ['id', 'title', 'abstract']
    """
    
    # Setup cache paths
    cache_dir = Path(__file__).parent / "arxiv_cache"
    cache_file = cache_dir / "arxiv_metadata.parquet"
    
    # Create cache directory if it doesn't exist
    cache_dir.mkdir(exist_ok=True)
    
    # Try to load from cache first (unless force_reload is True)
    if not force_reload and cache_file.exists():
        try:
            logger.info("Loading arXiv metadata from cache...")
            cached_df = pd.read_parquet(cache_file)
            logger.info(f"Successfully loaded {len(cached_df)} records from cache at {cache_file}")
            return cached_df
        except Exception as e:
            logger.warning(f"Failed to load cached metadata ({e}), falling back to Kaggle download...")
    else:
        if force_reload:
            logger.info("Force reload requested, bypassing cache...")
        else:
            logger.info("Cache not found, downloading from Kaggle...")
    
    # Download and process from Kaggle
    try:
        df = _download_and_process_kaggle_data()
        
        # Save to cache for future use
        try:
            df.to_parquet(cache_file, index=False)
            logger.info(f"Cached {len(df)} records to {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to save metadata cache: {e}")
        
        return df
        
    except Exception as e:
        logger.error(f"Failed to load arXiv metadata: {e}")
        # If we have a cache file as fallback, try to use it
        if cache_file.exists():
            logger.warning("Attempting to use cached data as fallback...")
            try:
                return pd.read_parquet(cache_file)
            except Exception as cache_e:
                logger.error(f"Cache fallback also failed: {cache_e}")
        raise

def _download_and_process_kaggle_data() -> pd.DataFrame:
    """Helper function to download and process data from Kaggle."""
    
    # Initializes Kaggle API.
    api = KaggleApi()
    api.authenticate()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        logger.info("Downloading arXiv metadata from Kaggle...")
        
        # Download the dataset to temporary directory
        api.dataset_download_files(
            'Cornell-University/arxiv',
            path=temp_dir,
            unzip=True
        )
        
        # Find the JSON file in the downloaded files
        json_file_path = None
        for file in os.listdir(temp_dir):
            if file.endswith('.json') and 'arxiv-metadata' in file:
                json_file_path = os.path.join(temp_dir, file)
                break
        
        if not json_file_path:
            # If not found, try the expected filename directly
            json_file_path = os.path.join(temp_dir, 'arxiv-metadata-oai-snapshot.json')
        
        if not os.path.exists(json_file_path):
            raise FileNotFoundError(f"arXiv metadata JSON file not found in {temp_dir}")
        
        logger.info(f"Loading JSON file: {json_file_path}")
        
        # Load the JSON file with pandas
        df = pd.read_json(json_file_path, lines=True)
        
        logger.info(f"Loaded {len(df)} records from arXiv metadata")
        
        # Apply development limit if configured
        arxiv_limit_str = os.environ.get("ARXIV_LOAD_LIMIT")
        if not arxiv_limit_str:
            raise ValueError("ARXIV_LOAD_LIMIT environment variable is not set or is empty")

        arxiv_limit = int(arxiv_limit_str)

        if arxiv_limit > 0:
            df = df.head(arxiv_limit)
            logger.info(f"Limited to {len(df)} records for development (ARXIV_LOAD_LIMIT={arxiv_limit})")
        
        # Filter to only required columns
        required_columns = ['id', 'title', 'abstract']
        
        # Check if all required columns exist
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.warning(f"Missing columns: {missing_columns}")
            logger.info(f"Available columns: {df.columns.tolist()}")
        
        # Return DataFrame with only required columns (if they exist)
        available_columns = [col for col in required_columns if col in df.columns]
        result_df = df[available_columns].copy()
        
        # Cleans up any null values.
        result_df = result_df.dropna()
        
        logger.info(f"Returning {len(result_df)} cleaned records with columns: {result_df.columns.tolist()}")
        
        return result_df 