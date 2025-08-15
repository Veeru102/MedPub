"""
NLTK configuration utility to ensure consistent data path setup across all modules.
"""
import os
import nltk
import logging

logger = logging.getLogger(__name__)

def configure_nltk_data_path():
    """Configure NLTK data path from environment variable or default location."""
    nltk_data_path = os.environ.get("NLTK_DATA", "/opt/render/project/src/nltk_data")
    
    # Add to NLTK data path if not already present
    if nltk_data_path not in nltk.data.path:
        nltk.data.path.insert(0, nltk_data_path)  # Insert at beginning for priority
        logger.info(f"NLTK data path configured: {nltk_data_path}")
    
    # Also try common fallback paths
    fallback_paths = [
        os.path.join(os.getcwd(), "nltk_data"),
        "/tmp/nltk_data"
    ]
    
    for path in fallback_paths:
        if os.path.exists(path) and path not in nltk.data.path:
            nltk.data.path.append(path)
            logger.info(f"Added fallback NLTK data path: {path}")
    
    logger.info(f"Final NLTK data paths: {nltk.data.path}")
    return nltk_data_path

def ensure_nltk_data():
    """Ensure required NLTK data packages are available."""
    configure_nltk_data_path()
    
    required_packages = ['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'averaged_perceptron_tagger', 'omw-1.4']
    
    for package in required_packages:
        try:
            if package == 'punkt_tab':
                nltk.data.find('tokenizers/punkt_tab')
            elif package == 'punkt':
                nltk.data.find('tokenizers/punkt')
            elif package == 'stopwords':
                nltk.data.find('corpora/stopwords')
            elif package == 'wordnet':
                nltk.data.find('corpora/wordnet')
            elif package == 'averaged_perceptron_tagger':
                nltk.data.find('taggers/averaged_perceptron_tagger')
            elif package == 'omw-1.4':
                nltk.data.find('corpora/omw-1.4')
        except LookupError:
            logger.warning(f"NLTK package '{package}' not found in data paths")
            # Don't try to download during runtime to avoid issues
            continue
    
    return True