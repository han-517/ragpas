"""
Abstract base class for RAG (Retrieval-Augmented Generation) systems.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING
from dataclasses import dataclass
import logging

if TYPE_CHECKING:
    from ragpas.config import RAGConfig

logger = logging.getLogger(__name__)


class RAGDocument:
    """Document wrapper for RAG systems."""
    
    def __init__(self, content: str, metadata: Optional[Dict[str, Any]] = None):
        self.content = content
        self.metadata = metadata or {}
        self.id = metadata.get("id") if metadata else None
    
    def __str__(self):
        return self.content
    
    def __repr__(self):
        return f"RAGDocument(content='{self.content[:50]}...', metadata={self.metadata})"


class BaseRAG(ABC):
    """Abstract base class for RAG (Retrieval-Augmented Generation) systems.
    
    This class defines the interface for RAG systems that can:
    1. Manage a document database (add, update, delete, retrieve)
    2. Generate responses based on retrieved context
    """
    
    def __init__(self, config: "RAGConfig"):
        """Initialize the RAG system with configuration.
        
        Args:
            config: RAG configuration object
        """
        self.config = config
        self._initialized = False
    
    @abstractmethod 
    def initialize(self) -> None:
        """Initialize the RAG system (load models, setup database, etc.)."""
        pass
    
    def _ensure_initialized(self) -> None:
        """Ensure the RAG system is initialized before operations."""
        if not self._initialized:
            self.initialize()
            self._initialized = True
    
    # Database operations
    @abstractmethod
    def add_documents(self, documents: List[Union[str, RAGDocument]], 
                     metadata: Optional[List[Dict[str, Any]]] = None) -> List[str]:
        """Add documents to the retrieval database.
        
        Args:
            documents: List of document contents or RAGDocument objects
            metadata: Optional list of metadata for each document
            
        Returns:
            List of document IDs that were added
        """
        pass
    
    @abstractmethod
    def update_document(self, document_id: str, 
                       content: Optional[str] = None,
                       metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Update an existing document in the database.
        
        Args:
            document_id: ID of the document to update
            content: New content for the document (if provided)
            metadata: New metadata for the document (if provided)
            
        Returns:
            True if update was successful, False otherwise
        """
        pass
    
    @abstractmethod
    def delete_documents(self, document_ids: List[str]) -> int:
        """Delete documents from the database.
        
        Args:
            document_ids: List of document IDs to delete
            
        Returns:
            Number of documents successfully deleted
        """
        pass
    
    @abstractmethod
    def retrieve_documents(self, query: str, 
                          top_k: Optional[int] = None,
                          filter_metadata: Optional[Dict[str, Any]] = None) -> List[RAGDocument]:
        """Retrieve relevant documents for a query.
        
        Args:
            query: Query string to search for
            top_k: Number of top documents to retrieve (uses config default if None)
            filter_metadata: Optional metadata filters
            
        Returns:
            List of relevant RAGDocument objects
        """
        pass
    
    @abstractmethod
    def get_document(self, document_id: str) -> Optional[RAGDocument]:
        """Get a specific document by ID.
        
        Args:
            document_id: ID of the document to retrieve
            
        Returns:
            RAGDocument if found, None otherwise
        """
        pass
    
    @abstractmethod
    def list_documents(self, limit: Optional[int] = None,
                      filter_metadata: Optional[Dict[str, Any]] = None) -> List[RAGDocument]:
        """List documents in the database.
        
        Args:
            limit: Maximum number of documents to return
            filter_metadata: Optional metadata filters
            
        Returns:
            List of RAGDocument objects
        """
        pass
    
    @abstractmethod
    def count_documents(self, filter_metadata: Optional[Dict[str, Any]] = None) -> int:
        """Count documents in the database.
        
        Args:
            filter_metadata: Optional metadata filters
            
        Returns:
            Number of documents matching the criteria
        """
        pass
    
    # Generation operations
    @abstractmethod
    def generate_response(self, query: str, 
                         context_documents: Optional[List[RAGDocument]] = None,
                         custom_prompt: Optional[str] = None) -> str:
        """Generate a response for a query using retrieved context.
        
        Args:
            query: User query
            context_documents: Pre-retrieved context documents (if None, will retrieve automatically)
            custom_prompt: Custom prompt template (if None, uses default)
            
        Returns:
            Generated response string
        """
        pass
    
    def query(self, query: str, 
              top_k: Optional[int] = None,
              filter_metadata: Optional[Dict[str, Any]] = None,
              custom_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Perform end-to-end RAG query (retrieve + generate).
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
            filter_metadata: Optional metadata filters for retrieval
            custom_prompt: Custom prompt template
            
        Returns:
            Dictionary containing query, retrieved documents, and response
        """
        self._ensure_initialized()
        
        # Retrieve relevant documents
        retrieved_docs = self.retrieve_documents(
            query=query, 
            top_k=top_k, 
            filter_metadata=filter_metadata
        )
        
        # Generate response
        response = self.generate_response(
            query=query,
            context_documents=retrieved_docs,
            custom_prompt=custom_prompt
        )
        
        return {
            "query": query,
            "retrieved_documents": retrieved_docs,
            "response": response,
            "num_retrieved": len(retrieved_docs)
        }
    
    # Utility operations
    @abstractmethod
    def clear_database(self) -> bool:
        """Clear all documents from the database.
        
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def get_database_info(self) -> Dict[str, Any]:
        """Get information about the database.
        
        Returns:
            Dictionary with database statistics and information
        """
        pass
    
    def batch_add_documents(self, documents: List[Union[str, RAGDocument]], 
                           batch_size: int = 100,
                           metadata: Optional[List[Dict[str, Any]]] = None) -> List[str]:
        """Add documents in batches for better performance.
        
        Args:
            documents: List of documents to add
            batch_size: Size of each batch
            metadata: Optional metadata for documents
            
        Returns:
            List of all document IDs that were added
        """
        self._ensure_initialized()
        
        all_doc_ids = []
        
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]
            batch_metadata = metadata[i:i + batch_size] if metadata else None
            
            batch_ids = self.add_documents(batch_docs, batch_metadata)
            all_doc_ids.extend(batch_ids)
            
            logger.info(f"Added batch {i//batch_size + 1}, "
                       f"documents {i+1}-{min(i+batch_size, len(documents))}")
        
        return all_doc_ids
