"""
Simple Vector Store - SQLite-independent alternative for ChromaDB
Uses in-memory storage to avoid SQLite compatibility issues
"""

import numpy as np
from typing import List, Dict, Any
from langchain_core.documents import Document

class SimpleRetriever:
    """Simple retriever that implements the LangChain retriever interface."""
    
    def __init__(self, store):
        self.store = store
    
    def get_relevant_documents(self, query):
        """Get relevant documents for a query."""
        return self.store.similarity_search(query, k=5)
    
    def invoke(self, query):
        """Invoke the retriever (alias for get_relevant_documents)."""
        return self.get_relevant_documents(query)

class SimpleVectorStore:
    """
    A simple, lightweight vector store that doesn't require SQLite.
    Stores embeddings in memory for fast retrieval.
    """
    
    def __init__(self, embedding_model, documents: List[Document] = None):
        self.embedding_model = embedding_model
        self.documents = documents or []
        self.embeddings = []
        self.metadata = []
        
        if documents:
            self._create_embeddings()
    
    def _create_embeddings(self):
        """Create embeddings for all documents."""
        if not self.documents:
            return
        
        print("Creating embeddings for documents...")
        texts = [doc.page_content for doc in self.documents if doc.page_content]
        
        try:
            self.embeddings = self.embedding_model.embed_documents(texts)
            self.metadata = [doc.metadata for doc in self.documents if doc.page_content]
            print(f"Created {len(self.embeddings)} embeddings successfully")
        except Exception as e:
            print(f"Error creating embeddings: {e}")
            # Fallback to simple random embeddings
            self._create_fallback_embeddings(len(texts))
    
    def _create_fallback_embeddings(self, num_docs: int):
        """Create simple fallback embeddings if the main model fails."""
        print("Using fallback embeddings...")
        self.embeddings = []
        for i in range(num_docs):
            # Create deterministic random embeddings
            np.random.seed(i)
            embedding = np.random.rand(384).tolist()  # 384 dimensions
            self.embeddings.append(embedding)
        self.metadata = [doc.metadata for doc in self.documents if doc.page_content]
    
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """Search for similar documents using cosine similarity."""
        if not self.embeddings:
            return []
        
        try:
            # Get query embedding
            query_embedding = self.embedding_model.embed_query(query)
        except Exception as e:
            print(f"Error embedding query: {e}")
            # Use simple fallback
            np.random.seed(hash(query) % 1000)
            query_embedding = np.random.rand(384).tolist()
        
        # Calculate similarities
        similarities = []
        for i, doc_embedding in enumerate(self.embeddings):
            similarity = self._cosine_similarity(query_embedding, doc_embedding)
            similarities.append((similarity, i))
        
        # Sort by similarity and return top k
        similarities.sort(reverse=True)
        top_k_indices = [idx for _, idx in similarities[:k]]
        
        results = []
        for idx in top_k_indices:
            if idx < len(self.documents):
                results.append(self.documents[idx])
        
        return results
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            vec1 = np.array(vec1)
            vec2 = np.array(vec2)
            
            # Normalize vectors
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            # Calculate cosine similarity
            similarity = np.dot(vec1, vec2) / (norm1 * norm2)
            return float(similarity)
        except Exception:
            return 0.0
    
    def add_documents(self, documents: List[Document]):
        """Add new documents to the vector store."""
        self.documents.extend(documents)
        self._create_embeddings()
    
    def get_document_count(self) -> int:
        """Get the total number of documents."""
        return len(self.documents)
    
    def as_retriever(self, **kwargs):
        """Create a retriever interface compatible with LangChain."""
        return SimpleRetriever(self)

def create_simple_vector_store(docs: List[Document], embedding_model):
    """
    Create a simple vector store that doesn't require SQLite.
    
    Args:
        docs: List of Document objects
        embedding_model: The embedding model to use
        
    Returns:
        A SimpleVectorStore instance
    """
    if not docs:
        raise ValueError("No documents provided")
    
    print("Creating simple vector store (SQLite-independent)...")
    vectorstore = SimpleVectorStore(embedding_model, docs)
    print("Simple vector store created successfully!")
    
    return vectorstore
