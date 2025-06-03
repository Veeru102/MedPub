from typing import List, Dict, Any
import os
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.documents import Document
import openai
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

class RAGEngine:
    def __init__(self):
        # Initialize OpenAI client without proxies
        self.openai_client = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
        )
        self.embeddings = OpenAIEmbeddings()
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
        self.vector_store = FAISS.from_texts(
            chunks,
            self.embeddings,
            metadatas=[metadata] * len(chunks) if metadata else None
        )
        self.save_vector_store()

    def create_vector_store_from_documents(self, documents: List[Document]):
        """
        Create a FAISS vector store from a list of Langchain Document objects
        """
        if not documents:
            raise ValueError("Cannot create vector store from empty documents list.")

        self.vector_store = FAISS.from_documents(documents, self.embeddings)
        self.save_vector_store()

    def save_vector_store(self):
        """
        Save the current FAISS vector store to disk.
        """
        if self.vector_store:
            self.vector_store.save_local(self.faiss_index_path)
            print(f"FAISS index saved to {self.faiss_index_path}")

    def load_vector_store(self):
        """
        Load the FAISS vector store from disk.
        """
        if os.path.exists(self.faiss_index_path):
            try:
                self.vector_store = FAISS.load_local(
                    self.faiss_index_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True  # Required for loading
                )
                print(f"FAISS index loaded from {self.faiss_index_path}")
            except Exception as e:
                print(f"Error loading FAISS index: {e}")
                self.vector_store = None # Ensure vector store is None if loading fails
        else:
            print("No existing FAISS index found.")

    def setup_qa_chain(self):
        """
        Set up the QA chain with the vector store
        """
        if not self.vector_store:
            # Attempt to load the vector store if not initialized
            self.load_vector_store()
            if not self.vector_store:
                # If still not initialized, it means no index file exists or loading failed
                # We can proceed without a vector store, but queries will not be grounded.
                # Or raise an error if RAG is strictly required.
                print("Warning: Vector store not available. RAG queries will not function.")
                self.qa_chain = None # Ensure qa_chain is None if vector store is not available
                return # Exit the function

        # Proceed with setting up the QA chain if vector store is available
        # Ensure only one memory instance is created per RAGEngine instance
        if not self.qa_chain or self.qa_chain.memory != self.memory:
            self.qa_chain = ConversationalRetrievalChain.from_llm(
                llm=ChatOpenAI(temperature=0),
                retriever=self.vector_store.as_retriever(),
                memory=self.memory,
                return_source_documents=True,
                output_key="answer"
            )

    async def query(self, question: str) -> Dict[str, Any]:
        """
        Query the RAG system with a question
        """
        # Ensure the QA chain is set up before querying. It will attempt to load the index.
        if not self.qa_chain or self.qa_chain.retriever.vectorstore != self.vector_store:
            self.setup_qa_chain()

        if not self.qa_chain:
            # If QA chain is still not initialized after setup_qa_chain,
            # it means the vector store was not available.
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
            print(f"Error in RAG query: {e}")
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
            print(f"Error in paper summarization: {e}")
            return {
                "summary": "Error generating summary",
                "status": "error",
                "error": str(e)
            } 