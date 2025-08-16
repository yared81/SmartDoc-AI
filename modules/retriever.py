from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.vectorstores import Chroma
from typing import Union, Any

def create_retriever(
    vectorstore: Any,  # Changed from Chroma to Any to handle different types
    search_k: int = 15,  # Increased for more context
    reranker_top_n: int = 5  # Increased for better coverage
):
    """
    Creates an enhanced retriever with improved context retrieval and optional reranking.
    Designed to provide more comprehensive context for better AI responses.
    Handles both ChromaDB and fallback vector stores.

    Args:
        vectorstore: The vector store instance (ChromaDB or fallback).
        search_k: The number of documents to retrieve initially (increased for more context).
        reranker_top_n: The number of documents to return after reranking (increased for coverage).

    Returns:
        A retriever instance with enhanced context retrieval.
    """
    # 1. Initialize the base retriever with more context
    try:
        base_retriever = vectorstore.as_retriever(
            search_kwargs={
                "k": search_k,
                "fetch_k": search_k * 2,  # Fetch more for better selection
                "score_threshold": 0.5  # Only return relevant documents
            }
        )
    except Exception as e:
        print(f"Error creating base retriever: {e}")
        # Fallback: create a simple retriever
        base_retriever = vectorstore.as_retriever()
    
    # Check if this is a fallback retriever (SimpleRetriever)
    if hasattr(base_retriever, '__class__') and 'SimpleRetriever' in str(base_retriever.__class__):
        print("Using fallback retriever - skipping advanced reranking")
        return base_retriever
    
    try:
        # 2. Try to initialize the Cross-Encoder model for reranking
        print("Loading advanced reranker for better context selection...")
        cross_encoder_model = HuggingFaceCrossEncoder(model_name='cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        # 3. Initialize the LangChain reranker with better parameters
        reranker = CrossEncoderReranker(
            model=cross_encoder_model, 
            top_n=reranker_top_n,
            threshold=0.5  # Only keep highly relevant documents
        )
        
        # 4. Create the enhanced compression retriever
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=reranker, 
            base_retriever=base_retriever
        )
        
        print("Advanced reranker loaded successfully - using enhanced context retrieval.")
        return compression_retriever
        
    except Exception as e:
        print(f"Advanced reranker failed to load: {e}")
        print("Using enhanced basic retrieval with more context.")
        
        # Enhanced fallback: Return basic retriever with better parameters
        return base_retriever