"""
NaiveRAG implementation based on the demo/naive_rag.ipynb notebook.
"""

import os
import time
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

import pandas as pd
from tqdm import tqdm

try:
    from langchain_community.vectorstores import Chroma
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain import hub
    from langchain_core.output_parsers import StrOutputParser
    from langgraph.graph import END, StateGraph, START
    from typing_extensions import TypedDict
except ImportError as e:
    raise ImportError(
        "Required langchain dependencies not found. Please install: "
        "pip install langchain-community langchain-openai langchain "
        "langgraph langchain-text-splitters chromadb"
    ) from e

from .base import BaseRAG, RAGDocument
from ragpas.config import RAGConfig

logger = logging.getLogger(__name__)


class RAGState(TypedDict):
    """State class for LangGraph workflow."""
    query: str
    contexts: List[str]
    response: str


class NaiveRAG(BaseRAG):
    """Naive RAG implementation using Chroma vectorstore and LangGraph workflow.
    
    This implementation is based on the demo/naive_rag.ipynb notebook and provides
    a simple but effective RAG system with document management capabilities.
    """
    
    def __init__(self, config: RAGConfig):
        """Initialize NaiveRAG with configuration."""
        super().__init__(config)
        
        # Initialize components
        self.vectorstore = None
        self.embeddings = None
        self.llm = None
        self.text_splitter = None
        self.rag_workflow = None
        
    def initialize(self) -> None:
        """Initialize the RAG system components."""
        logger.info("Initializing NaiveRAG system...")
        
        # Initialize embeddings
        if self.config.embedding_model.find("doubao") != -1:
            self.embeddings = OpenAIEmbeddings(
                model=self.config.embedding_model,
                api_key=os.environ.get("EMBEDDING_API_KEY"),
                base_url=os.environ.get("EMBEDDING_API_URL"),
                dimensions=self.config.embedding_dimensions,
                check_embedding_ctx_length=False,
                openai_proxy=self.config.proxy
            )
        else:
            self.embeddings = OpenAIEmbeddings(
                model=self.config.embedding_model,
                api_key=os.environ.get("EMBEDDING_API_KEY"),
                base_url=os.environ.get("EMBEDDING_API_URL"),
                dimensions=self.config.embedding_dimensions,
                openai_proxy=self.config.proxy
            )
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=self.config.llm_model,
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url=os.environ.get("OPENAI_API_URL"),
            temperature=self.config.temperature,
            openai_proxy=self.config.proxy
        )
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )
        
        # Initialize vector store
        self.vectorstore = Chroma(
            collection_name=self.config.collection_name,
            persist_directory=self.config.database_path,
            embedding_function=self.embeddings,
        )
        
        # Initialize RAG workflow
        self._setup_rag_workflow()
        
        self._initialized = True
        logger.info("NaiveRAG system initialized successfully")
    
    def _setup_rag_workflow(self) -> None:
        """Setup the LangGraph workflow for RAG."""
        
        def retrieve(state: RAGState) -> Dict[str, Any]:
            """Retrieve relevant documents for the query."""
            query = state["query"]
            retrieved_docs = self.vectorstore.similarity_search(
                query=query, 
                k=self.config.top_k
            )
            contexts = [doc.page_content for doc in retrieved_docs]
            return {"query": query, "contexts": contexts}
        
        def generate(state: RAGState) -> Dict[str, Any]:
            """Generate response using retrieved contexts."""
            query = state["query"]
            contexts = state["contexts"]
            
            # Combine contexts
            combined_context = "\n\n".join(contexts)
            
            # Use RAG prompt from hub
            try:
                prompt = hub.pull("rlm/rag-prompt")
                rag_chain = prompt | self.llm | StrOutputParser()
                response = rag_chain.invoke({
                    "question": query, 
                    "context": combined_context
                })
            except Exception as e:
                logger.warning(f"Failed to use hub prompt, using fallback: {e}")
                # Fallback to simple prompt
                fallback_prompt = f"""Context:\n{combined_context}\n\nQuestion: {query}\n\nAnswer:"""
                response = self.llm.invoke(fallback_prompt).content
            
            return {
                "query": query, 
                "contexts": contexts, 
                "response": response
            }
        
        # Build workflow
        workflow = StateGraph(RAGState)
        workflow.add_node("retrieve", retrieve)
        workflow.add_node("generate", generate)
        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)
        
        self.rag_workflow = workflow.compile()
    
    def add_documents(self, documents: List[Union[str, RAGDocument]], 
                     metadata: Optional[List[Dict[str, Any]]] = None) -> List[str]:
        """Add documents to the vector database."""
        self._ensure_initialized()
        
        doc_ids = []
        texts_to_add = []
        metadatas_to_add = []
        
        for i, doc in enumerate(documents):
            if isinstance(doc, RAGDocument):
                content = doc.content
                doc_metadata = doc.metadata.copy()
            else:
                content = doc
                doc_metadata = metadata[i] if metadata and i < len(metadata) else {}
            
            # Split text into chunks
            chunks = self.text_splitter.split_text(content)
            
            for j, chunk in enumerate(chunks):
                chunk_id = str(uuid4())
                doc_ids.append(chunk_id)
                texts_to_add.append(chunk)
                
                chunk_metadata = doc_metadata.copy()
                chunk_metadata.update({
                    "id": chunk_id,
                    "chunk_index": j,
                    "total_chunks": len(chunks),
                    "original_doc_index": i
                })
                metadatas_to_add.append(chunk_metadata)
        
        # Add to vectorstore
        self.vectorstore.add_texts(
            texts=texts_to_add,
            metadatas=metadatas_to_add,
            ids=doc_ids
        )
        
        logger.info(f"Added {len(texts_to_add)} chunks from {len(documents)} documents")
        return doc_ids
    
    def update_document(self, document_id: str, 
                       content: Optional[str] = None,
                       metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Update an existing document in the database."""
        self._ensure_initialized()
        
        try:
            # Get existing document
            existing_doc = self.get_document(document_id)
            if not existing_doc:
                logger.warning(f"Document {document_id} not found for update")
                return False
            
            # Delete existing document
            self.delete_documents([document_id])
            
            # Add updated document
            new_content = content if content is not None else existing_doc.content
            new_metadata = metadata if metadata is not None else existing_doc.metadata
            new_metadata["id"] = document_id  # Preserve original ID
            
            self.add_documents([RAGDocument(new_content, new_metadata)])
            
            logger.info(f"Updated document {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update document {document_id}: {e}")
            return False
    
    def delete_documents(self, document_ids: List[str]) -> int:
        """Delete documents from the database."""
        self._ensure_initialized()
        
        try:
            self.vectorstore.delete(ids=document_ids)
            logger.info(f"Deleted {len(document_ids)} documents")
            return len(document_ids)
        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")
            return 0
    
    def retrieve_documents(self, query: str, 
                          top_k: Optional[int] = None,
                          filter_metadata: Optional[Dict[str, Any]] = None) -> List[RAGDocument]:
        """Retrieve relevant documents for a query."""
        self._ensure_initialized()
        
        k = top_k or self.config.top_k
        
        # Perform similarity search
        results = self.vectorstore.similarity_search(
            query=query,
            k=k,
            filter=filter_metadata
        )
        
        # Convert to RAGDocument objects
        documents = []
        for result in results:
            doc = RAGDocument(
                content=result.page_content,
                metadata=result.metadata
            )
            documents.append(doc)
        
        return documents
    
    def get_document(self, document_id: str) -> Optional[RAGDocument]:
        """Get a specific document by ID."""
        self._ensure_initialized()
        
        try:
            results = self.vectorstore.get(ids=[document_id])
            if results["documents"] and len(results["documents"]) > 0:
                return RAGDocument(
                    content=results["documents"][0],
                    metadata=results["metadatas"][0] if results["metadatas"] else {}
                )
        except Exception as e:
            logger.error(f"Failed to get document {document_id}: {e}")
        
        return None
    
    def list_documents(self, limit: Optional[int] = None,
                      filter_metadata: Optional[Dict[str, Any]] = None) -> List[RAGDocument]:
        """List documents in the database."""
        self._ensure_initialized()
        
        try:
            # Get all documents (Chroma doesn't support limit directly)
            results = self.vectorstore.get()
            
            documents = []
            for i, (content, metadata) in enumerate(zip(
                results["documents"], 
                results["metadatas"] or [{}] * len(results["documents"])
            )):
                # Apply metadata filter if specified
                if filter_metadata:
                    if not all(metadata.get(k) == v for k, v in filter_metadata.items()):
                        continue
                
                doc = RAGDocument(content=content, metadata=metadata)
                documents.append(doc)
                
                # Apply limit
                if limit and len(documents) >= limit:
                    break
            
            return documents
            
        except Exception as e:
            logger.error(f"Failed to list documents: {e}")
            return []
    
    def count_documents(self, filter_metadata: Optional[Dict[str, Any]] = None) -> int:
        """Count documents in the database."""
        self._ensure_initialized()
        
        if filter_metadata:
            # Need to list and filter for count
            filtered_docs = self.list_documents(filter_metadata=filter_metadata)
            return len(filtered_docs)
        else:
            # Get total count
            try:
                results = self.vectorstore.get()
                return len(results["documents"])
            except Exception as e:
                logger.error(f"Failed to count documents: {e}")
                return 0
    
    def generate_response(self, query: str, 
                         context_documents: Optional[List[RAGDocument]] = None,
                         custom_prompt: Optional[str] = None) -> str:
        """Generate a response for a query using retrieved context."""
        self._ensure_initialized()
        
        # Use the LangGraph workflow for response generation
        if context_documents is None:
            # Standard workflow: retrieve + generate
            state = RAGState(query=query)
            try:
                result = self.rag_workflow.invoke(state)
                return result["response"]
            except Exception as e:
                # Retry logic from original notebook
                logger.warning(f"RAG workflow failed, retrying: {e}")
                for attempt in range(5):
                    time.sleep(5)
                    try:
                        result = self.rag_workflow.invoke(state)
                        return result["response"]
                    except Exception:
                        if attempt == 4:  # Last attempt
                            raise Exception("Failed to generate response after 5 attempts")
                        continue
        else:
            # Use provided context documents
            contexts = [doc.content for doc in context_documents]
            state = RAGState(query=query, contexts=contexts)
            
            # Generate response directly
            combined_context = "\n\n".join(contexts)
            
            if custom_prompt:
                prompt_text = custom_prompt.format(
                    question=query, 
                    context=combined_context
                )
                response = self.llm.invoke(prompt_text).content
            else:
                try:
                    prompt = hub.pull("rlm/rag-prompt")
                    rag_chain = prompt | self.llm | StrOutputParser()
                    response = rag_chain.invoke({
                        "question": query, 
                        "context": combined_context
                    })
                except Exception as e:
                    logger.warning(f"Failed to use hub prompt: {e}")
                    fallback_prompt = f"""Context:\n{combined_context}\n\nQuestion: {query}\n\nAnswer:"""
                    response = self.llm.invoke(fallback_prompt).content
            
            return response
    
    def clear_database(self) -> bool:
        """Clear all documents from the database."""
        self._ensure_initialized()
        
        try:
            # Get all document IDs
            results = self.vectorstore.get()
            if results["ids"]:
                self.vectorstore.delete(ids=results["ids"])
            
            logger.info("Database cleared successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear database: {e}")
            return False
    
    def get_database_info(self) -> Dict[str, Any]:
        """Get information about the database."""
        self._ensure_initialized()
        
        try:
            results = self.vectorstore.get()
            return {
                "total_documents": len(results["documents"]),
                "collection_name": self.config.collection_name,
                "database_path": self.config.database_path,
                "embedding_model": self.config.embedding_model,
                "chunk_size": self.config.chunk_size,
                "chunk_overlap": self.config.chunk_overlap
            }
        except Exception as e:
            logger.error(f"Failed to get database info: {e}")
            return {}
    
    def process_csv_contexts(self, csv_file_path: str, 
                           context_column: str = "context") -> List[str]:
        """Process contexts from CSV file (utility method from original notebook)."""
        try:
            df = pd.read_csv(csv_file_path)
            contexts = df[context_column].tolist()
            
            # Add contexts to database
            doc_ids = self.batch_add_documents(contexts)
            logger.info(f"Processed {len(contexts)} contexts from {csv_file_path}")
            
            return doc_ids
            
        except Exception as e:
            logger.error(f"Failed to process CSV contexts: {e}")
            return []
    
    def batch_process_queries(self, queries: List[str], 
                             output_file: Optional[str] = None,
                             batch_size: int = 10) -> List[str]:
        """Batch process multiple queries (utility method from original notebook)."""
        self._ensure_initialized()
        
        responses = []
        
        for i, query in enumerate(tqdm(queries, desc="Processing queries")):
            try:
                response = self.generate_response(query)
                responses.append(response)
                
                # Save periodically if output file specified
                if output_file and (i % batch_size == 0 or i == len(queries) - 1):
                    df = pd.DataFrame({
                        "query": queries[:len(responses)],
                        "response": responses
                    })
                    df.to_csv(output_file, index=False)
                    logger.info(f"Saved progress: {len(responses)} queries processed")
                    
            except Exception as e:
                logger.error(f"Failed to process query {i}: {e}")
                responses.append(f"Error: {str(e)}")
        
        return responses
