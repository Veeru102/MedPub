#!/bin/bash
set -e

echo "Starting build process..."
echo "Current directory: $(pwd)"
echo "Listing files:"
ls -la

# Install Python dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create NLTK data directory
echo "Creating NLTK data directory..."
mkdir -p /opt/render/project/src/nltk_data

# Set NLTK data path and run download script
echo "Setting up NLTK data..."
export NLTK_DATA=/opt/render/project/src/nltk_data

# Check if we're in the backend directory or need to navigate to it
if [ -f "download_nltk.py" ]; then
    echo "Found download_nltk.py in current directory"
    python download_nltk.py
elif [ -f "backend/download_nltk.py" ]; then
    echo "Found download_nltk.py in backend directory"
    cd backend
    python download_nltk.py
else
    echo "Error: Could not find download_nltk.py"
    exit 1
fi

echo "Build completed successfully!"