from typing import List, Dict, Any, Optional
import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.memory import ConversationBufferMemory
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
import openai
from openai import AsyncOpenAI
from dotenv import load_dotenv
import logging
import asyncio
import time
from functools import wraps

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Optimized batch configuration
BATCH_SIZE = 16  # Increased from 3/5 to 16
RATE_LIMIT_CALLS_PER_MINUTE = 50

def rate_limit(calls_per_minute=RATE_LIMIT_CALLS_PER_MINUTE):
    """Rate limiting decorator"""
    def decorator(func):
        last_called = [0.0]
        
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Calculate time to wait
            elapsed = time.time() - last_called[0]
            time_between_calls = 60.0 / calls_per_minute
            
            if elapsed < time_between_calls:
                sleep_time = time_between_calls - elapsed
                logger.debug(f"Rate limiting: waiting {sleep_time:.2f} seconds")  # Reduced to debug level
                await asyncio.sleep(sleep_time)
            
            last_called[0] = time.time()
            return await func(*args, **kwargs)
        return wrapper
    return decorator

async def retry_with_exponential_backoff(
    func,
    max_retries: int = 3,  # Type hint added
    initial_delay: float = 1,
    backoff_factor: float = 2,
    *args,
    **kwargs
):
    """Retry function with exponential backoff"""
    # Ensure max_retries is an integer
    if not isinstance(max_retries, int):
        raise TypeError("max_retries must be an integer")
    
    start_time = time.time()
    for attempt in range(max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            if attempt == max_retries:
                raise e
            
            # Only backoff on rate limit errors
            if "429" in str(e) or "rate limit" in str(e).lower():
                delay = initial_delay * (backoff_factor ** attempt)
                elapsed = time.time() - start_time
                logger.warning(f"Rate limit hit after {elapsed:.2f}s, retrying in {delay}s (attempt {attempt + 1}/{max_retries + 1})")
                await asyncio.sleep(delay)
            else:
                raise e

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

    async def create_vector_store_from_documents_with_retry(self, documents: List[Document]):
        """
        Create a FAISS vector store from documents with optimized batching
        """
        if not documents:
            raise ValueError("Cannot create vector store from empty documents list.")

        try:
            logger.info(f"Creating vector store with {len(documents)} documents in batches of {BATCH_SIZE}")
            start_time = time.time()
            
            # Use retry logic for vector store creation
            self.vector_store = await retry_with_exponential_backoff(
                self._create_vector_store_async,
                max_retries=3,  # Explicit int
                documents=documents  # Named argument
            )
            
            elapsed = time.time() - start_time
            logger.info(f"Vector store created in {elapsed:.2f}s")
            self.save_vector_store()
            
        except Exception as e:
            logger.error(f"Failed to create vector store from documents: {e}")
            raise

    async def _create_vector_store_async(self, documents: List[Document]):
        """Helper method to create vector store asynchronously with optimized batching"""
        if len(documents) > BATCH_SIZE:
            logger.info(f"Processing {len(documents)} documents in {len(documents) // BATCH_SIZE + 1} batches")
            
            # Create initial vector store with first batch
            first_batch = documents[:BATCH_SIZE]
            batch_start = time.time()
            vector_store = FAISS.from_documents(first_batch, self.embeddings)
            batch_time = time.time() - batch_start
            logger.info(f"Batch 1/{len(documents) // BATCH_SIZE + 1} completed in {batch_time:.2f}s")
            
            # Add remaining documents in batches
            for i in range(BATCH_SIZE, len(documents), BATCH_SIZE):
                batch = documents[i:i + BATCH_SIZE]
                batch_start = time.time()
                
                try:
                    batch_store = FAISS.from_documents(batch, self.embeddings)
                    vector_store.merge_from(batch_store)
                    
                    batch_time = time.time() - batch_start
                    batch_num = i // BATCH_SIZE + 1
                    logger.info(f"Batch {batch_num}/{len(documents) // BATCH_SIZE + 1} completed in {batch_time:.2f}s")
                    
                except Exception as e:
                    if "429" in str(e):
                        logger.warning(f"Rate limit hit on batch {i // BATCH_SIZE + 1}, retrying...")
                        raise  # Let retry_with_exponential_backoff handle it
                    raise
            
            return vector_store
        else:
            return FAISS.from_documents(documents, self.embeddings)

    def create_vector_store_from_documents(self, documents: List[Document]):
        """
        Create a FAISS vector store from a list of Langchain Document objects
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                logger.warning("Already in event loop, using synchronous method with optimized batching")
                self._create_vector_store_sync_with_delays(documents)
            else:
                loop.run_until_complete(self.create_vector_store_from_documents_with_retry(documents))
        except RuntimeError:
            self._create_vector_store_sync_with_delays(documents)

    def _create_vector_store_sync_with_delays(self, documents: List[Document]):
        """Synchronous version with optimized batching"""
        if not documents:
            raise ValueError("Cannot create vector store from empty documents list.")

        try:
            logger.info(f"Processing {len(documents)} documents in batches of {BATCH_SIZE}")
            start_time = time.time()
            
            if len(documents) > BATCH_SIZE:
                # Create initial vector store with first batch
                first_batch = documents[:BATCH_SIZE]
                self.vector_store = FAISS.from_documents(first_batch, self.embeddings)
                
                # Add remaining documents in batches
                for i in range(BATCH_SIZE, len(documents), BATCH_SIZE):
                    batch = documents[i:i + BATCH_SIZE]
                    batch_start = time.time()
                    
                    batch_store = FAISS.from_documents(batch, self.embeddings)
                    self.vector_store.merge_from(batch_store)
                    
                    batch_time = time.time() - batch_start
                    batch_num = i // BATCH_SIZE + 1
                    logger.info(f"Batch {batch_num}/{len(documents) // BATCH_SIZE + 1} completed in {batch_time:.2f}s")
            else:
                self.vector_store = FAISS.from_documents(documents, self.embeddings)
                
            elapsed = time.time() - start_time
            logger.info(f"Vector store created in {elapsed:.2f}s")
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
                return

        try:
            # Create the language model with custom prompt
            system_prompt = "You are a helpful medical research assistant. Use the following context to answer the question. If you don't know the answer, say so."
            llm = ChatOpenAI(
                temperature=0.3,
                model="gpt-3.5-turbo"
            ).with_config({  # Configure prompt via llm instead
                "prompt": system_prompt
            })

            # Create the chain without prompt parameter
            self.qa_chain = create_retrieval_chain(
                llm=llm,
                retriever=self.vector_store.as_retriever()
            )
            
            logger.info("QA chain setup complete")
                    
        except Exception as e:
            logger.error(f"Error setting up QA chain: {e}")
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

    @rate_limit(calls_per_minute=30)  # Conservative rate limiting
    async def _make_chat_completion(self, model_name: str, messages: List[Dict], temperature: float = 0.3):
        """Make a rate-limited chat completion request"""
        return await retry_with_exponential_backoff(
            self.openai_client.chat.completions.create,
            model=model_name,
            messages=messages,
            temperature=temperature
        )

    async def summarize_paper(self, chunks: List[str]) -> Dict[str, Any]:
        """
        Generate a summary of the paper using GPT with rate limiting
        """
        if not chunks:
            return {
                "summary": "No content to summarize.",
                "status": "success"
            }

        try:
            full_text = "\n".join(chunks)
            
            max_prompt_length = 12000  # Reduced to be more conservative
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

            # Try different models with rate limiting
            models_to_try = ["gpt-4o-mini", "gpt-3.5-turbo", "gpt-4-turbo", "gpt-4"]
            
            response = None
            last_error = None
            
            for model_name in models_to_try:
                try:
                    logger.info(f"Attempting summarization with model: {model_name}")
                    
                    response = await self._make_chat_completion(
                        model_name=model_name,
                        messages=[
                            {"role": "system", "content": "You are a medical research assistant specializing in paper analysis and summarization."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.3
                    )
                    
                    logger.info(f"Successfully used model: {model_name}")
                    break
                    
                except Exception as e:
                    logger.warning(f"Failed to use model {model_name}: {e}")
                    last_error = e
                    
                    # Add extra delay if it's a rate limit error
                    if "429" in str(e):
                        logger.info("Rate limit detected, waiting extra time before next model...")
                        await asyncio.sleep(5)
                    
                    continue
            
            if response is None:
                raise Exception(f"All models failed. Last error: {last_error}")

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