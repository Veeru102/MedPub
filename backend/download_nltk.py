#!/usr/bin/env python3
"""
Download required NLTK data for the MedPub backend
Run this script once after installing requirements
"""

import nltk
import sys
import os

def download_nltk_data():
    """Download all required NLTK packages"""
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
            nltk.download(package, quiet=False)
        except Exception as e:
            print(f"Warning: Failed to download {package}: {e}")
    
    print("\nNLTK data download completed!")
    print("You can now run the backend server.")

if __name__ == "__main__":
    download_nltk_data() 