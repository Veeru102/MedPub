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
        self.openai_client = AsyncOpenAI()
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = None
        self.qa_chain = None
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )

    def create_vector_store(self, chunks: List[str], metadata: Dict[str, Any] = None):
        """
        Create a FAISS vector store from document chunks
        """
        self.vector_store = FAISS.from_texts(
            chunks,
            self.embeddings,
            metadatas=[metadata] * len(chunks) if metadata else None
        )

    def create_vector_store_from_documents(self, documents: List[Document]):
        """
        Create a FAISS vector store from a list of Langchain Document objects
        """
        if not documents:
            raise ValueError("Cannot create vector store from empty documents list.")

        self.vector_store = FAISS.from_documents(documents, self.embeddings)

    def setup_qa_chain(self):
        """
        Set up the QA chain with the vector store
        """
        if not self.vector_store:
            raise ValueError("Vector store not initialized")

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
        if not self.qa_chain:
            raise ValueError("QA chain not initialized")

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