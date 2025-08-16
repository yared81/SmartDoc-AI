from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List

def chunk_data(data: List[Document], chunk_size: int = 1000, chunk_overlap: int = 100) -> List[Document]:
    """
    Splits a list of documents into smaller chunks.
    
    Args:
        data: A list of Document objects to be split.
        chunk_size: The maximum size of each chunk (in characters).
        chunk_overlap: The number of characters to overlap between chunks.
        
    Returns:
        A list of chunked Document objects.
    """
    print(f"Chunker: Processing {len(data)} documents")
    
    # Check input documents
    for i, doc in enumerate(data):
        if not hasattr(doc, 'page_content'):
            print(f"Chunker Warning: Document {i} has no page_content attribute")
            continue
        if not doc.page_content or not doc.page_content.strip():
            print(f"Chunker Warning: Document {i} has empty page_content")
            continue
        print(f"Chunker: Document {i} has {len(doc.page_content)} characters")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    chunks = text_splitter.split_documents(data)
    print(f"Chunker: Created {len(chunks)} chunks")
    
    # Check output chunks
    for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
        if hasattr(chunk, 'page_content'):
            print(f"Chunker: Chunk {i} has {len(chunk.page_content)} characters")
        else:
            print(f"Chunker Warning: Chunk {i} has no page_content attribute")
    
    return chunks