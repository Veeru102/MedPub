# MedPub: An AI-Powered Medical Research Assistant

MedPub is an advanced web application designed to empower medical researchers and clinicians by streamlining the process of literature review and knowledge extraction from PDF-based research papers. Leveraging cutting-edge AI and machine learning, MedPub facilitates intelligent document ingestion, contextual summarization, and interactive, multi-document querying, significantly enhancing research efficiency and insight generation.

## Technical Highlights

MedPub is engineered as a robust, full-stack AI application, showcasing sophisticated architectural patterns, intelligent data processing, and seamless user interaction.

### AI/ML Core (Retrieval-Augmented Generation - RAG Pipeline)

At the heart of MedPub is a meticulously designed RAG pipeline that transforms raw PDF content into a searchable knowledge base and provides context-aware responses.

*   **Intelligent Document Ingestion**: PDFs are parsed using `PyMuPDF`, and their content is intelligently chunked into manageable segments. This process includes preserving metadata (e.g., page numbers, sections) for accurate source attribution.
*   **Vector Embeddings with OpenAI**: Document chunks are converted into high-dimensional vector embeddings using OpenAI's powerful embedding models. These numerical representations capture the semantic meaning of the text.
*   **FAISS Vector Store**: For efficient similarity search and retrieval, these embeddings are indexed and stored in a [FAISS (Facebook AI Similarity Search)](https://github.com/facebookresearch/faiss) vector database. This enables rapid identification of the most semantically relevant document chunks to a given query.
*   **Contextual Retrieval**: During a chat interaction, user questions are embedded and used to retrieve the most relevant document chunks from the FAISS index. A `similarity_score_threshold` is applied to ensure only highly pertinent information is considered, optimizing both accuracy and latency.
*   **Conversational AI with LangChain**: The system employs `LangChain` to orchestrate complex interactions between the user's query, retrieved context, and large language models (LLMs). This includes:
    *   **Multi-Turn Conversational Chain (`ConversationalRetrievalChain`)**: Supports dynamic, multi-turn dialogues with memory (`ConversationSummaryBufferMemory`) to maintain conversation context and coherence.
    *   **Audience-Specific Summarization**: Abstracts and full paper content can be summarized based on selected audience types (Patient, Clinician, Researcher) by dynamically adjusting LLM prompts and styles.
    *   **Source Evidence & Confidence Scoring**: Chat responses are augmented with precise, concise source snippets (sentence/paragraph-level) from the original PDFs, along with cosine similarity-based confidence scores, ensuring transparency and verifiability.
    *   **Robust LLM Integration**: Features adaptive LLM model selection (`gpt-4o-mini`, `gpt-3.5-turbo`, `gpt-4-turbo`, `gpt-4`) and includes sophisticated error handling mechanisms like `retry_with_exponential_backoff` and rate limiting (`@rate_limit`) to ensure reliable API interactions.
*   **Semantic Search for Similar Papers**: An integrated `arxiv_search` module leverages sentence transformer models and a dedicated FAISS index for arXiv papers. This allows MedPub to identify and recommend semantically similar research papers based on the content of the currently analyzed document.

### Frontend Architecture

The user interface is built with a focus on responsiveness, modularity, and a modern user experience.

*   **React with TypeScript**: Provides a robust and scalable foundation for the single-page application, ensuring type safety and maintainability.
*   **Tailwind CSS**: A utility-first CSS framework used for rapid and consistent UI development, facilitating a sleek, component-driven design.
*   **Theming System**: Implemented a dynamic light/dark mode theme toggle with automatic system preference detection, ensuring a visually appealing experience across user preferences. The theme system is managed via React Context and leverages Tailwind's `dark:` variants for seamless transitions.
*   **Component-Based Design**: The UI is logically decomposed into reusable components (`Sidebar`, `PDFUpload`, `AudienceSelector`, `SummaryDisplay`, `Chat`, `SimilarPapersBox`), promoting modularity, reusability, and easier maintenance.
*   **Interactive Document Management**: Features multi-document selection in the sidebar for contextual chat, real-time upload progress, and immediate feedback on summarization status.

### Backend Infrastructure & API Design

The backend is engineered for performance, scalability, and maintainability, providing a robust API layer for the frontend.

*   **FastAPI**: A modern, high-performance web framework for Python, ideal for building efficient and scalable asynchronous APIs. It provides automatic interactive API documentation (Swagger UI/ReDoc).
*   **Asynchronous Operations**: Extensive use of Python's `asyncio` and FastAPI's asynchronous capabilities ensures non-blocking I/O operations, critical for handling long-running tasks like PDF processing, embedding generation, and LLM calls.
*   **Modular Service Design**: The backend is structured into distinct services (`rag_engine.py`, `llm_services.py`, `arxiv_search.py`), promoting clear separation of concerns and independent development/testing.
*   **Rate Limiting & Error Handling**: Custom decorators (`@rate_limit`, `retry_with_exponential_backoff`) are implemented for external API calls (e.g., OpenAI, arXiv) to prevent rate limit breaches and ensure resilient operation. Comprehensive logging (`logging` module) provides insights into runtime behavior and errors.
*   **Vector Store Management**: The FAISS index is efficiently managed, supporting both initial creation from new documents and incremental updates, with local persistence for faster startup times.

## Architecture Diagram

```mermaid
graph TD
    A[User Interface - React/TypeScript] -->|API Calls (HTTP/JSON)| B(FastAPI Backend);

    B --> C{Document Ingestion & Processing};
    C --> D[PyMuPDF];
    C --> E[LangChain Document Loaders/Splitters];

    B --> F{Embedding Generation};
    F --> G[OpenAI Embeddings API];

    B --> H{Vector Store Management};
    H --> I[FAISS Index (Local Persistence)];

    B --> J{LLM Interactions};
    J --> K[OpenAI Chat/Completion API];
    J --> L[LangChain Chains (RAG, Conversational)];

    B --> M{External Services};
    M --> N[arXiv Search API];
    M --> O[Sentence Transformer Models];

    subgraph RAG Pipeline
        D --> F;
        E --> F;
        F --> H;
        H --> L;
        K --> L;
    end

    subgraph User Interaction Flow
        A --o Chat[Chat Component];
        A --o Summary[Summary Display];
        A --o SimilarPapers[Similar Papers Box];
        A --o Upload[PDF Upload];

        Upload --> B;
        Chat --> B;
        Summary --> B;
        SimilarPapers --> B;
    end

    style A fill:#fff,stroke:#333,stroke-width:2px,rx:8px,ry:8px;
    style B fill:#fff,stroke:#333,stroke-width:2px,rx:8px,ry:8px;
    style C fill:#f9f,stroke:#333,stroke-width:2px,rx:8px,ry:8px;
    style D fill:#cdf,stroke:#333,stroke-width:2px,rx:8px,ry:8px;
    style E fill:#cdf,stroke:#333,stroke-width:2px,rx:8px,ry:8px;
    style F fill:#fcf,stroke:#333,stroke-width:2px,rx:8px,ry:8px;
    style G fill:#dfd,stroke:#333,stroke-width:2px,rx:8px,ry:8px;
    style H fill:#f9f,stroke:#333,stroke-width:2px,rx:8px,ry:8px;
    style I fill:#cdf,stroke:#333,stroke-width:2px,rx:8px,ry:8px;
    style J fill:#fcf,stroke:#333,stroke-width:2px,rx:8px,ry:8px;
    style K fill:#dfd,stroke:#333,stroke-width:2px,rx:8px,ry:8px;
    style L fill:#cdf,stroke:#333,stroke-width:2px,rx:8px,ry:8px;
    style M fill:#f9f,stroke:#333,stroke-width:2px,rx:8px,ry:8px;
    style N fill:#dfd,stroke:#333,stroke-width:2px,rx:8px,ry:8px;
    style O fill:#dfd,stroke:#333,stroke-width:2px,rx:8px,ry:8px;

    classDef default fill:#f8f8f8,stroke:#333,stroke-width:2px;
    classDef mainNode fill:#e0f7fa,stroke:#00acc1,stroke-width:2px;

```

## Setup Instructions

This project requires both Python (for the backend) and Node.js (for the frontend).

### Prerequisites

*   **Python 3.9+**: Ensure Python is installed and accessible via your PATH.
*   **Node.js 18+**: Ensure Node.js and npm (or yarn) are installed.
*   **OpenAI API Key**: Obtain an API key from OpenAI and set it as an environment variable.
