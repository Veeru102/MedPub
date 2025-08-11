# Use a lightweight Python base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Run NLTK download during build
COPY download_nltk.py .
RUN python download_nltk.py

# Copy the backend application code
COPY . .

# Expose the port the application runs on
EXPOSE 8000 