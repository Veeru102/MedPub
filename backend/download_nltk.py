#!/usr/bin/env python3
"""Downloads required NLTK data for the MedPub backend."""
# Run this script once after installing requirements.

import nltk
import sys
import os

def download_nltk_data(download_dir=None):
    """Downloads all required NLTK packages."""
    packages = [
        'punkt',
        'punkt_tab', 
        'stopwords',
        'wordnet',
        'averaged_perceptron_tagger',
        'omw-1.4'
    ]

    print("Downloading NLTK data packages...")
    
    for package in packages:
        try:
            print(f"Downloading {package}...")
            nltk.download(package, download_dir=download_dir, quiet=False)
        except Exception as e:
            print(f"Warning: Failed to download {package}: {e}")
    
    print("\nNLTK data download completed!")
    print("You can now run the backend server.")

if __name__ == "__main__":
    # Default to a directory relative to the script if not specified by env var
    nltk_data_path = os.environ.get("NLTK_DATA", os.path.join(os.getcwd(), "nltk_data"))
    os.makedirs(nltk_data_path, exist_ok=True)
    nltk.data.path.append(nltk_data_path) # Add to NLTK's data path
    download_nltk_data(download_dir=nltk_data_path) 