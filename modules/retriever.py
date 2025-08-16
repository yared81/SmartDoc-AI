from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.vectorstores import Chroma

def create_retriever(
    vectorstore: Chroma, 
    search_k: int = 15,  # Increased for more context
    reranker_top_n: int = 5  # Increased for better coverage
):
    """
    Creates an enhanced retriever with improved context retrieval and optional reranking.
    Designed to provide more comprehensive context for better AI responses.

    Args:
        vectorstore: The Chroma vector store instance.
        search_k: The number of documents to retrieve initially (increased for more context).
        reranker_top_n: The number of documents to return after reranking (increased for coverage).

    Returns:
        A retriever instance with enhanced context retrieval.
    """
    # 1. Initialize the base retriever with more context
    base_retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": search_k,
            "fetch_k": search_k * 2,  # Fetch more for better selection
            "score_threshold": 0.5  # Only return relevant documents
        }
    )
    
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