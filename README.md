# MedPub: An AI-Powered Medical Research Assistant

MedPub is a full-stack web application designed to facilitate medical literature review and knowledge extraction from PDF documents. It leverages AI and machine learning to enable document ingestion, contextual summarization, and interactive multi-document querying, aiming to enhance research workflows and accelerate insight generation.

## Technical Architecture and Implementation

MedPub is architected as a full-stack AI application, demonstrating robust design patterns, efficient data processing, and responsive user interaction.

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

### Frontend Engineering

The user interface is developed with an emphasis on modularity, responsiveness, and an intuitive user experience.

*   **React with TypeScript**: Provides a type-safe, component-driven framework for building a scalable single-page application.
*   **Tailwind CSS**: A utility-first CSS framework enabling efficient and consistent styling, promoting rapid UI development and maintainability.
*   **Dynamic Theming System**: Implemented a light/dark mode theme with automatic system preference detection. This system utilizes React Context for state management and Tailwind's `dark:` variants for seamless visual transitions.
*   **Component-Based Architecture**: The UI is structured into discrete, reusable components (`Sidebar`, `PDFUpload`, `AudienceSelector`, `SummaryDisplay`, `Chat`, `SimilarPapersBox`), which enhances code organization, reusability, and development efficiency.
*   **Interactive Document Management**: Features support multi-document selection for unified chat contexts, real-time upload progress feedback, and instant summarization status updates.

### Backend Infrastructure & API Design

The backend is designed for high performance, scalability, and maintainability, serving as an API layer.

*   **FastAPI**: A modern Python web framework enabling the development of high-performance, asynchronous APIs. FastAPI automatically generates interactive API documentation (Swagger UI/ReDoc).
*   **Asynchronous Operations**: Leverages Python's `asyncio` and FastAPI's asynchronous capabilities for non-blocking I/O, essential for computationally intensive tasks such as PDF processing, embedding generation, and LLM inferences.
*   **Modular Service Design**: The backend is organized into distinct, focused services (`rag_engine.py`, `llm_services.py`, `arxiv_search.py`), promoting clear separation of concerns, testability, and parallel development.
*   **API Resilience**: Custom decorators (`@rate_limit`, `retry_with_exponential_backoff`) are implemented to manage external API calls, preventing rate limit breaches and enhancing system resilience.
*   **Vector Store Management**: The FAISS index is managed efficiently, supporting both initial construction and incremental updates. Local persistence of the index optimizes application startup times.

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
