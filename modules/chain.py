# helpers/chain.py

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
# Import ChatGroq instead of HuggingFaceHub
from langchain_groq import ChatGroq
from langchain.retrievers import ContextualCompressionRetriever
from typing import Any

def _format_docs(docs: list) -> str:
    """
    Formats the retrieved documents into a single string.
    """
    return "\n\n".join(doc.page_content for doc in docs)

def create_rag_chain(retriever: Any):
    """
    Creates the full RAG chain for question answering using Groq.
    Handles both advanced retrievers and fallback retrievers.

    Args:
        retriever: The retriever instance to fetch relevant documents.

    Returns:
        A runnable RAG chain.
    """
    # 1. Initialize the LLM from Groq
    # We'll use Llama 3, which is very fast on Groq.
    # The API key is automatically read from the GROQ_API_KEY environment variable.
    try:
        llm = ChatGroq(
            temperature=0, 
            model_name="llama3-8b-8192"
        )
    except Exception as e:
        if "groq_api_key" in str(e).lower() or "api_key" in str(e).lower():
            raise Exception(
                "GROQ_API_KEY not found! Please add it to Streamlit Cloud secrets:\n"
                "1. Go to your app's Settings → Secrets\n"
                "2. Add: GROQ_API_KEY = 'your_actual_api_key_here'\n"
                "3. Wait 1-2 minutes for it to propagate"
            )
        else:
            raise Exception(f"Error initializing Groq LLM: {e}")

    # 2. Create the enhanced prompt template for better context understanding
    prompt_template = """
    You are an expert AI assistant with deep understanding of documents and context. Your task is to provide comprehensive, accurate answers based on the provided context.

    **IMPORTANT INSTRUCTIONS:**
    1. **Context Analysis**: Carefully analyze the provided context to understand the full scope
    2. **Comprehensive Answers**: Provide detailed, well-structured answers that cover all relevant aspects
    3. **Source Attribution**: Reference specific parts of the context when possible
    4. **Logical Flow**: Organize your answer in a logical, easy-to-follow structure
    5. **Professional Tone**: Use clear, professional language suitable for business contexts
    6. **Actionable Insights**: When possible, provide actionable insights or recommendations

    **Context:**
    {context}

    **Question:**
    {question}

    **Answer Guidelines:**
    - Start with a direct answer to the question
    - Provide supporting details from the context
    - Use bullet points or numbered lists for clarity when appropriate
    - If the context doesn't contain enough information, clearly state what's missing
    - End with a brief summary or key takeaway

    **Your Answer:**
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)

    # 3. Create the RAG chain using LCEL
    rag_chain = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain