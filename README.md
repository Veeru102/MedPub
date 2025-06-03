# MedCopilot

MedCopilot is an AI-powered literature review assistant that helps medical researchers analyze and summarize academic papers efficiently.

## Features

- 📄 PDF Upload and Processing
- 🔍 Smart Document Chunking
- 🤖 RAG-based Research Question Answering
- 📝 AI-Powered Paper Summarization
- 📚 Citation Management
- 💬 Interactive Chat Interface

## Tech Stack

### Frontend
- React with TypeScript
- Tailwind CSS + ShadCN UI
- Vite for build tooling

### Backend
- FastAPI (Python)
- WebSocket for real-time communication
- LangChain for document processing
- FAISS for vector storage
- PyMuPDF for PDF handling

### AI/ML
- OpenAI GPT-4/Claude 3
- LangChain for RAG implementation
- Custom summarization agents

## Project Structure

```
medcopilot/
├── frontend/           # React frontend application
├── backend/           # FastAPI backend server
├── docs/             # Documentation
└── scripts/          # Utility scripts
```

## Setup Instructions

### Prerequisites
- Python 3.9+
- Node.js 18+
- OpenAI API key

### Backend Setup
1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your OpenAI API key
   ```

4. Run the backend:
   ```bash
   uvicorn main:app --reload
   ```

### Frontend Setup
1. Install dependencies:
   ```bash
   cd frontend
   npm config get prefix
   npm install -g tailwindcss
   tailwindcss init -p
   npm install
   ```

2. Run the development server:
   ```bash
   npm run dev
   ```

## License

MIT License 