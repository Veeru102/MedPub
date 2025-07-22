from typing import List, Dict, Any, Optional
import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain  
from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnableMap, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import openai
from openai import AsyncOpenAI
from dotenv import load_dotenv
import logging
import asyncio
import time
from functools import wraps
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Suppress tokenizer parallelism warnings.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configure logging.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# batch configuration
BATCH_SIZE = 16  
RATE_LIMIT_CALLS_PER_MINUTE = 50

def rate_limit(calls_per_minute=RATE_LIMIT_CALLS_PER_MINUTE):
    def decorator(func):
        last_called = [0.0]
        
        @wraps(func)
        async def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            time_between_calls = 60.0 / calls_per_minute
            
            if elapsed < time_between_calls:
                sleep_time = time_between_calls - elapsed
                logger.debug(f"Rate limiting: waiting {sleep_time:.2f} seconds")  
                await asyncio.sleep(sleep_time)
            
            last_called[0] = time.time()
            return await func(*args, **kwargs)
        return wrapper
    return decorator

async def retry_with_exponential_backoff(
    func,
    max_retries: int = 3,  
    initial_delay: float = 1,
    backoff_factor: float = 2,
    *args,
    **kwargs
):
    if not isinstance(max_retries, int):
        raise TypeError("max_retries must be an integer")
    
    start_time = time.time()
    for attempt in range(max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            if attempt == max_retries:
                raise e
            
            # Only backoff on rate limit errors.
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
            
        # Initialize OpenAI client.
        self.openai_client = AsyncOpenAI(api_key=api_key)
        
        try:
            self.embeddings = OpenAIEmbeddings()
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI embeddings: {e}")
            raise
            
        self.vector_store = None
        self.qa_chain = None
        self.chat_chain = None  
        
        # Memory for conversations.
        self.memory = ConversationSummaryBufferMemory(
            llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo"),
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
            max_token_limit=1000  
        )
        
        self.faiss_index_path = "faiss_index"

    def create_vector_store(self, chunks: List[str], metadata: Dict[str, Any] = None):
        """Create a FAISS vector store from document chunks"""
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
        """Create a FAISS vector store from documents with optimized batching"""
        if not documents:
            raise ValueError("Cannot create vector store from empty documents list.")

        try:
            logger.info(f"Creating vector store with {len(documents)} documents in batches of {BATCH_SIZE}")
            start_time = time.time()
            
            # Use retry logic for vector store creation
            self.vector_store = await retry_with_exponential_backoff(
                self._create_vector_store_async,
                max_retries=3,  
                documents=documents  
            )
            
            elapsed = time.time() - start_time
            logger.info(f"Vector store created in {elapsed:.2f}s")
            self.save_vector_store()
            
        except Exception as e:
            logger.error(f"Failed to create vector store from documents: {e}")
            raise

    async def _create_vector_store_async(self, documents: List[Document]):
        """Helper to create vector store asynchronously with optimized batching."""
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
                        raise  # Retry_with_exponential_backoff handles this.
                    raise
            
            return vector_store
        else:
            return FAISS.from_documents(documents, self.embeddings)

    def create_vector_store_from_documents(self, documents: List[Document]):
        """Creates a FAISS vector store from a list of Langchain Document objects."""
        if not documents:
            raise ValueError("Cannot create vector store from empty documents list.")

        try:
            logger.info(f"Processing {len(documents)} documents in batches of {BATCH_SIZE}")
            start_time = time.time()
            
            # Checks if a vector store already exists.
            if self.vector_store:
                logger.info("Updating existing vector store")
                # Processes documents in batches.
                for i in range(0, len(documents), BATCH_SIZE):
                    batch = documents[i:i + BATCH_SIZE]
                    batch_start = time.time()
                    
                    # Create temporary store for batch
                    batch_store = FAISS.from_documents(batch, self.embeddings)
                    
                    # Merge into existing store
                    self.vector_store.merge_from(batch_store)
                    
                    batch_time = time.time() - batch_start
                    batch_num = i // BATCH_SIZE + 1
                    logger.info(f"Batch {batch_num}/{len(documents) // BATCH_SIZE + 1} merged in {batch_time:.2f}s")
            else:
                # Create new vector store
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
            logger.info(f"Vector store {'updated' if self.vector_store else 'created'} in {elapsed:.2f}s")
            self.save_vector_store()
            
            # Reinitialize the chains with the updated vector store
            self.setup_qa_chain()
            self.setup_chat_chain()
            
        except Exception as e:
            logger.error(f"Failed to {'update' if self.vector_store else 'create'} vector store from documents: {e}")
            raise

    def save_vector_store(self):
        """Save the current FAISS vector store to disk"""
        if self.vector_store:
            try:
                self.vector_store.save_local(self.faiss_index_path)
                logger.info(f"FAISS index saved to {self.faiss_index_path}")
            except Exception as e:
                logger.error(f"Failed to save FAISS index: {e}")
                raise

    def load_vector_store(self):
        """Load the FAISS vector store from disk"""
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
        """Sets up the single-turn QA chain."""
        if not self.vector_store:
            self.load_vector_store()
            if not self.vector_store:
                return

        try:
            # Creates simple retrieval chain using LCEL for single-turn QA.
            llm = ChatOpenAI(temperature=0.3, model="gpt-3.5-turbo")
            retriever = self.vector_store.as_retriever(search_kwargs={"k": 4})
            
            # Creates QA prompt.
            qa_prompt = PromptTemplate(
                template="""You are a helpful medical research assistant. Use the following context to answer the question. If you don't know the answer based on the context, say so.

Context: {context}

Question: {question}

Answer:""",
                input_variables=["context", "question"]
            )
            
            # Builds chain using LCEL.
            def format_docs(docs):
                return "\n\n".join([d.page_content for d in docs])
            
            self.qa_chain = (
                RunnableMap({
                    "context": retriever | format_docs,
                    "question": RunnablePassthrough()
                })
                | qa_prompt
                | llm
                | StrOutputParser()
            )
            
            logger.info("QA chain setup complete")
                    
        except Exception as e:
            logger.error(f"Error setting up QA chain: {e}")
            raise

    def setup_chat_chain(self):
        """Sets up the conversational RAG chain for multi-turn chat."""
        if not self.vector_store:
            self.load_vector_store()
            if not self.vector_store:
                return

        try:
            # Creates the language model.
            llm = ChatOpenAI(
                temperature=0.3,
                model="gpt-3.5-turbo",
                streaming=True  
            )

            # Creates retriever with optimized search parameters.
            retriever = self.vector_store.as_retriever(
                search_type="similarity_score_threshold", 
                search_kwargs={
                    "k": 3,  
                    "score_threshold": 0.5  
                }
            )

            # Creates custom prompt for conversational RAG.
            chat_prompt = PromptTemplate(
                template="""You are a helpful medical research assistant engaged in a conversation. Use the following context from medical papers to answer the question, but also consider the conversation history.

Context from documents:
{context}

Current question: {question}

Instructions:
- Provide accurate, helpful responses based on the context
- If you don't know something based on the provided context, say so
- Maintain conversation flow by referencing previous exchanges when relevant
- Be conversational but professional
- When citing sources, be specific about which document and section the information comes from

Answer:""",
                input_variables=["context", "question"]
            )

            # Create the conversational retrieval chain
            self.chat_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=retriever,
                memory=self.memory,
                return_source_documents=True,
                verbose=True,
                combine_docs_chain_kwargs={"prompt": chat_prompt}
            )
            
            logger.info("Chat chain setup complete")
                    
        except Exception as e:
            logger.error(f"Error setting up chat chain: {e}")
            raise

    async def query(self, question: str) -> Dict[str, Any]:
        """Single-turn QA query."""
        if not self.qa_chain:
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
            
            # LCEL chain result is the answer string.
            return {
                "answer": result,
                "sources": [] 
            }
        except Exception as e:
            logger.error(f"Error in RAG query: {e}")
            raise

    async def chat(self, question: str, chat_history: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Multi-turn conversational chat with memory."""
        if not self.chat_chain:
            try:
                self.setup_chat_chain()
            except Exception as e:
                logger.error(f"Failed to setup chat chain during chat: {e}")
                return {
                    "answer": "An error occurred while setting up the chat system. Please try again.",
                    "sources": [],
                    "chat_history": []
                }

        if not self.chat_chain:
            return {
                "answer": "The chat system is not initialized. Please upload and process a document first.",
                "sources": [],
                "chat_history": []
            }

        try:
            # If external chat history is provided, it can be optionally injected.
            # Relies on internal memory
            
            result = await self.chat_chain.ainvoke({"question": question})
            
            # Extracts information from the result.
            answer = result.get("answer", "")
            source_documents = result.get("source_documents", [])
            
            # Calculates similarity scores for source documents.
            source_similarities = []
            for doc in source_documents:
                # Gets embeddings for the answer and document.
                answer_embedding = await self.embeddings.aembed_query(answer)
                doc_embedding = await self.embeddings.aembed_query(doc.page_content)
                
                # Calculates cosine similarity.
                similarity = float(cosine_similarity(
                    np.array(answer_embedding).reshape(1, -1),
                    np.array(doc_embedding).reshape(1, -1)
                )[0][0])
                
                source_similarities.append((doc, similarity))
            
            # Sort by similarity and get top sources
            source_similarities.sort(key=lambda x: x[1], reverse=True)
            top_sources = source_similarities[:3]  # Limits to top 3
            
            # Formats sources with similarity scores and metadata.
            sources = []
            for doc, similarity in top_sources:
                # Extracts a relevant snippet
                content = doc.page_content
                if len(content) > 200:
                    start = max(0, len(content)//2 - 100)
                    end = min(len(content), len(content)//2 + 100)
                    content = f"...{content[start:end]}..."
                
                sources.append({
                    "content": content,
                    "metadata": doc.metadata,
                    "similarity": similarity,
                    "document_title": doc.metadata.get("title", "Unknown"),
                    "section": doc.metadata.get("section", "Unknown"),
                    "page": doc.metadata.get("page", "Unknown")
                })
            
            # Gets current chat history from memory.
            chat_history = []
            if hasattr(self.memory, 'chat_memory') and hasattr(self.memory.chat_memory, 'messages'):
                messages = self.memory.chat_memory.messages
                for i in range(0, len(messages), 2):
                    if i + 1 < len(messages):
                        chat_history.append({
                            "human": messages[i].content,
                            "ai": messages[i + 1].content
                        })

            return {
                "answer": answer,
                "sources": sources,
                "chat_history": chat_history
            }
            
        except Exception as e:
            logger.error(f"Error in RAG chat: {e}")
            raise

    def clear_memory(self):
        """Clear conversation memory"""
        if self.memory:
            self.memory.clear()
            logger.info("Conversation memory cleared")

    @rate_limit(calls_per_minute=30)  
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
        Generates a summary of the paper using GPT.
        """
        if not chunks:
            return {
                "summary": "No content to summarize.",
                "status": "success"
            }

        try:
            full_text = "\n".join(chunks)
            
            max_prompt_length = 12000  
            if len(full_text) > max_prompt_length:
                full_text = full_text[:max_prompt_length] + "\n... [Content truncated] ..."

            prompt = f"""You are a medical research assistant. Your task is to summarize the uploaded paper with emphasis on its:
- Key objectives
- Methodology
- Major findings
- Limitations

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
                    
                    # Adds extra delay if it's a rate limit error.
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