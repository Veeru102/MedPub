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


## Setup Instructions

This project requires Python (for the backend) and Node.js (for the frontend).

### Prerequisites

*   **Python 3.9+**: Ensure Python is installed and configured in your system PATH.
*   **Node.js 18+**: Ensure Node.js and npm (or yarn) are installed.
*   **OpenAI API Key**: Obtain an API key from OpenAI.

### Local Development Setup

1.  **Backend Setup:**
    ```bash
    cd backend
    python -m venv venv
    source venv/bin/activate # On Windows: .\venv\Scripts\activate
    pip install -r requirements.txt
    # Create a .env file and add your OpenAI API key
    echo "OPENAI_API_KEY=\"your_openai_api_key_here\"" > .env
    uvicorn main:app --reload
    ```
2.  **Frontend Setup:**
    ```bash
    cd frontend
    npm install
    npm run dev
    ```

### Running the Application

Once both backend and frontend development servers are active, access the application via your web browser at the address provided by the Vite server (typically `http://localhost:5173`).

## Future Roadmap

*   **Advanced Citation Management**: Implement granular citation tracking and direct navigation to source content within PDFs.
*   **Multi-Document Synthesis**: Develop capabilities to synthesize findings across multiple papers, identifying consensus, contradictions, and thematic trends.
*   **User Authentication & Persistent Storage**: Integrate a robust authentication system and a dedicated database for persistent storage of user documents and chat histories.
*   **Scalable Vector Store**: Evaluate and integrate cloud-native or distributed vector store solutions (e.g., Pinecone, Weaviate, Milvus) for production-grade scalability.

## Challenges & Trade-offs

Development of MedPub involved addressing key engineering challenges and making deliberate design trade-offs:

*   **LLM API Resilience**: Implemented custom retry logic with exponential backoff and client-side rate limiting to manage OpenAI API quotas and ensure system stability under varying load conditions or transient network issues.
*   **Context Window Optimization**: Managed LLM token limits by implementing efficient document chunking strategies, dynamic retrieval `k` value adjustments, and `ConversationSummaryBufferMemory` for concise chat history summarization.
*   **UI Responsiveness for AI Workloads**: Engineered asynchronous backend operations and optimized frontend state management to prevent UI blocking during intensive AI tasks (embeddings, LLM calls), maintaining a fluid user experience.
*   **Accuracy vs. Latency in RAG**: Balanced the depth of retrieved context for answer accuracy against the need for low-latency responses, particularly for interactive chat.

### Prerequisites

*   **Python 3.9+**: Ensure Python is installed and accessible via your PATH.
*   **Node.js 18+**: Ensure Node.js and npm (or yarn) are installed.
*   **OpenAI API Key**: Obtain an API key from OpenAI and set it as an environment variable.
