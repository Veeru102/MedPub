## Overview

**MedPub** is a full-stack web application designed to facilitate medical literature review and knowledge extraction from PDF-based research papers. It leverages AI and machine learning to:

- Parse and segment medical PDFs using intelligent document chunking  
- Generate audience-specific summaries tailored for patients, clinicians, or researchers  
- Support interactive, multi-turn document chat through a Retrieval-Augmented Generation (RAG) pipeline  
- Retrieve semantically relevant content using OpenAI embeddings and FAISS-based vector search  
- Provide grounded answers with transparent source evidence from the original text  
- Recommend similar academic papers using sentence transformer models and semantic search  
- Enable multi-document selection and contextual reasoning with persistent chat memory  
- Ensure robust and reliable API interactions with built-in rate limiting and retry logic


### AI/ML Core: Retrieval-Augmented Generation (RAG) Pipeline

Central to MedPub's functionality is a RAG pipeline that processes unstructured PDF content into a searchable knowledge base for contextual responses.

*   **Document Ingestion & Chunking**: PDFs are parsed using `PyMuPDF` and segmented into semantically coherent chunks. Metadata such as page numbers and sections are preserved to enable accurate source attribution.
*   **Vector Embeddings**: Document chunks are transformed into high-dimensional vector embeddings via OpenAI's embedding models, capturing semantic relationships within the text.
*   **FAISS Vector Store**: Embeddings are indexed and stored in a [FAISS (Facebook AI Similarity Search)](https://github.com/facebookresearch/faiss) vector database. This facilitates low-latency similarity searches, enabling efficient retrieval of relevant document segments for a given query.
*   **Contextual Retrieval**: During user interactions, queries are embedded and matched against the FAISS index. A `similarity_score_threshold` is applied to filter retrieved chunks, ensuring high relevance and optimizing response latency.
*   **Conversational AI with LangChain**: `LangChain` orchestrates interactions between user queries, retrieved context, and Large Language Models (LLMs). This includes:
    *   **Multi-Turn Conversational Chain**: Utilizes `ConversationalRetrievalChain` with `ConversationSummaryBufferMemory` to maintain dialogue coherence across multiple turns.
    *   **Audience-Specific Summarization**: LLM prompts are dynamically adjusted to generate summaries tailored for different audiences (Patient, Clinician, Researcher).
    *   **Source Attribution**: Chat responses are augmented with concise, sentence- or paragraph-level source snippets from original PDFs, including cosine similarity-based confidence scores for verifiability.
    *   **LLM Integration Resilience**: Adaptive LLM model selection and robust error handling mechanisms (`retry_with_exponential_backoff`, `@rate_limit`) mitigate API rate limits and ensure consistent operation with external LLM providers (OpenAI).
*   **Semantic Search for Similar Papers**: An integrated `arxiv_search` module employs sentence transformer models and a dedicated FAISS index to identify and recommend semantically similar arXiv papers based on uploaded document content.

## Frontend Engineering

- React with TypeScript  
- Tailwind CSS  
- Light/dark theme toggle with system preference detection  
- Component-based structure (`Sidebar`, `PDFUpload`, `AudienceSelector`, `SummaryDisplay`, `Chat`, `SimilarPapersBox`)  
- Multi-document selection and contextual chat  
- Real-time upload progress and summarization status  

## Backend Infrastructure

- FastAPI with async endpoints  
- Modular service structure (`rag_engine.py`, `llm_services.py`, `arxiv_search.py`)  
- FAISS vector store with incremental update and local persistence  
- Async PDF processing, embedding, and LLM interaction  
- Rate limiting and exponential backoff for external APIs  
