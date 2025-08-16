from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import List
from langchain_core.documents import Document
import numpy as np

def create_vector_store(docs: List[Document]):
    """
    Creates a Chroma vector store from a list of documents.
    Uses local embeddings to avoid network issues.

    Args:
        docs: A list of Document objects (chunks).

    Returns:
        A Chroma vector store instance.
    """
    print("Creating vector store... This may take a moment.")
    
    if not docs:
        raise ValueError("No documents provided to create vector store")
    
    print(f"Processing {len(docs)} documents...")
    
    # Extract text content from documents
    texts = []
    for doc in docs:
        if hasattr(doc, 'page_content') and doc.page_content:
            texts.append(doc.page_content)
        elif isinstance(doc, str):
            texts.append(doc)
        else:
            print(f"Warning: Document has no page_content: {type(doc)}")
    
    if not texts:
        raise ValueError("No text content found in documents")
    
    print(f"Extracted {len(texts)} text chunks")
    
    try:
        # Use a lightweight model that downloads at runtime
        print("Attempting to use Hugging Face embeddings...")
        # Use a smaller, faster model that's under 100MB
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            cache_folder=None,  # Don't cache locally
            local_files_only=False  # Allow downloading at runtime
        )
        print("Using Hugging Face embeddings with runtime loading.")
    except Exception as e:
        print(f"Hugging Face model failed: {e}")
        print("Falling back to local embeddings...")
        
        # Fallback: Create simple local embeddings
        embedding_model = LocalEmbeddings()
    
    try:
        # Test the embedding model
        print("Testing embedding model...")
        test_embedding = embedding_model.embed_query("test")
        print(f"Test embedding generated: {len(test_embedding)} dimensions")
        
        # Create the vector store from the documents and embedding model
        print("Creating Chroma vector store...")
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embedding_model
        )
        
        print("Vector store created successfully.")
        return vectorstore
        
    except Exception as e:
        print(f"Error creating vector store: {e}")
        print("Attempting to debug embedding issue...")
        
        # Debug: Check what's happening with embeddings
        try:
            test_embeddings = embedding_model.embed_documents(texts[:1])
            print(f"Debug: Generated {len(test_embeddings)} test embeddings")
            if test_embeddings:
                print(f"Debug: First embedding has {len(test_embeddings[0])} dimensions")
        except Exception as debug_e:
            print(f"Debug: Embedding test failed: {debug_e}")
        
        raise Exception(f"Failed to create vector store: {e}")

class LocalEmbeddings:
    """
    Vercel-optimized lightweight embedding model.
    Creates simple vector representations without heavy dependencies.
    """
    
    def __init__(self, dimensions=384):
        self.dimensions = dimensions
        print(f"Using local embeddings with {dimensions} dimensions")
    
    def embed_documents(self, texts):
        """Create simple embeddings for documents."""
        if not texts:
            return []
        
        embeddings = []
        for text in texts:
            if not text or not text.strip():
                # Handle empty text by creating a random vector
                embedding = self._create_random_embedding()
            else:
                # Create a simple hash-based embedding
                embedding = self._text_to_embedding(text)
            embeddings.append(embedding)
        
        print(f"Generated {len(embeddings)} local embeddings")
        return embeddings
    
    def embed_query(self, text):
        """Create embedding for a query."""
        if not text or not text.strip():
            return self._create_random_embedding()
        return self._text_to_embedding(text)
    
    def _text_to_embedding(self, text):
        """Convert text to a simple embedding vector."""
        try:
            import hashlib
            
            # Create a hash of the text
            text_hash = hashlib.md5(text.encode()).hexdigest()
            
            # Convert hash to numbers (0-255)
            numbers = [int(text_hash[i:i+2], 16) for i in range(0, len(text_hash), 2)]
            
            # Pad or truncate to desired dimensions
            if len(numbers) < self.dimensions:
                # Extend with pattern-based numbers
                for i in range(len(numbers), self.dimensions):
                    numbers.append((numbers[i % len(numbers)] + i) % 256)
            else:
                numbers = numbers[:self.dimensions]
            
            # Convert to float and normalize
            vector = [float(num) / 255.0 for num in numbers]
            
            # Ensure we have exactly the right dimensions
            assert len(vector) == self.dimensions, f"Vector length {len(vector)} != {self.dimensions}"
            
            return vector
            
        except Exception as e:
            print(f"Error in text embedding: {e}")
            return self._create_random_embedding()
    
    def _create_random_embedding(self):
        """Create a random embedding when text processing fails."""
        import random
        
        # Create a deterministic random vector
        random.seed(42)  # Fixed seed for consistency
        vector = [random.uniform(-1.0, 1.0) for _ in range(self.dimensions)]
        
        # Normalize to unit vector
        magnitude = sum(x*x for x in vector) ** 0.5
        if magnitude > 0:
            vector = [x / magnitude for x in vector]
        
        return vector