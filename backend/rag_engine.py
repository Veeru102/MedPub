from typing import List, Dict, Any
import os
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.documents import Document
import openai
from openai import AsyncOpenAI
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class RAGEngine:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
            
        # Initialize OpenAI client with minimal configuration
        self.openai_client = AsyncOpenAI(api_key=api_key)
        
        try:
            self.embeddings = OpenAIEmbeddings()
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI embeddings: {e}")
            raise
            
        self.vector_store = None
        self.qa_chain = None
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        # Define the path for the FAISS index file
        self.faiss_index_path = "faiss_index"

    def create_vector_store(self, chunks: List[str], metadata: Dict[str, Any] = None):
        """
        Create a FAISS vector store from document chunks
        """
        try:
            self.vector_store = FAISS.from_texts(
                chunks,
                self.embeddings,
                metadatas=[metadata] * len(chunks) if metadata else None
            )
            self.save_vector_store()
        except Exception as e:
            logger.error(f"Failed to create vector store: {e}")
            raise

    def create_vector_store_from_documents(self, documents: List[Document]):
        """
        Create a FAISS vector store from a list of Langchain Document objects
        """
        if not documents:
            raise ValueError("Cannot create vector store from empty documents list.")

        try:
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
            self.save_vector_store()
        except Exception as e:
            logger.error(f"Failed to create vector store from documents: {e}")
            raise

    def save_vector_store(self):
        """
        Save the current FAISS vector store to disk.
        """
        if self.vector_store:
            try:
                self.vector_store.save_local(self.faiss_index_path)
                logger.info(f"FAISS index saved to {self.faiss_index_path}")
            except Exception as e:
                logger.error(f"Failed to save FAISS index: {e}")
                raise

    def load_vector_store(self):
        """
        Load the FAISS vector store from disk.
        """
        if os.path.exists(self.faiss_index_path):
            try:
                self.vector_store = FAISS.load_local(
                    self.faiss_index_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                logger.info(f"FAISS index loaded from {self.faiss_index_path}")
            except Exception as e:
                logger.error(f"Error loading FAISS index: {e}")
                self.vector_store = None
        else:
            logger.info("No existing FAISS index found.")
            self.vector_store = None

    def setup_qa_chain(self):
        """
        Set up the QA chain with the vector store
        """
        if not self.vector_store:
            self.load_vector_store()
            if not self.vector_store:
                logger.warning("Vector store not available. RAG queries will not function.")
                self.qa_chain = None
                return

        try:
            if not self.qa_chain or self.qa_chain.memory != self.memory:
                self.qa_chain = ConversationalRetrievalChain.from_llm(
                    llm=ChatOpenAI(temperature=0),
                    retriever=self.vector_store.as_retriever(),
                    memory=self.memory,
                    return_source_documents=True,
                    output_key="answer"
                )
        except Exception as e:
            logger.error(f"Failed to setup QA chain: {e}")
            self.qa_chain = None
            raise

    async def query(self, question: str) -> Dict[str, Any]:
        """
        Query the RAG system with a question
        """
        if not self.qa_chain or self.qa_chain.retriever.vectorstore != self.vector_store:
            try:
                self.setup_qa_chain()
            except Exception as e:
                logger.error(f"Failed to setup QA chain during query: {e}")
                return {
                    "answer": "An error occurred while setting up the RAG system. Please try again.",
                    "sources": []
                }

        if not self.qa_chain:
            return {
                "answer": "The RAG system is not initialized. Please upload and process a document first.",
                "sources": []
            }

        try:
            result = await self.qa_chain.ainvoke({"question": question})
            return {
                "answer": result.get("answer", ""),
                "sources": [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata
                    }
                    for doc in result.get("source_documents", [])
                ]
            }
        except Exception as e:
            logger.error(f"Error in RAG query: {e}")
            raise

    async def summarize_paper(self, chunks: List[str]) -> Dict[str, Any]:
        """
        Generate a summary of the paper using GPT-4
        """
        if not chunks:
            return {
                "summary": "No content to summarize.",
                "status": "success"
            }

        try:
            full_text = "\n".join(chunks)
            
            max_prompt_length = 15000
            if len(full_text) > max_prompt_length:
                full_text = full_text[:max_prompt_length] + "\n... [Content truncated] ..."

            prompt = f"""You are a medical research assistant. Your task is to summarize the uploaded paper with emphasis on its:
- Key objectives
- Methodology
- Major findings
- Limitations
Also, infer the field of study and suggest 2-3 related papers if possible.

Paper content:
{full_text}
"""

            response = await self.openai_client.chat.completions.create(
                model="gpt-4o",  
                messages=[
                    {"role": "system", "content": "You are a medical research assistant specializing in paper analysis and summarization."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )

            return {
                "summary": response.choices[0].message.content,
                "status": "success"
            }

        except Exception as e:
            logger.error(f"Error in paper summarization: {e}")
            return {
                "summary": "Error generating summary",
                "status": "error",
                "error": str(e)
            } 