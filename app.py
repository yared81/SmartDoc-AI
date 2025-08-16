import streamlit as st
from dotenv import load_dotenv
import base64
from PIL import Image
import io
from datetime import datetime

from modules.chunker import chunk_data
from modules.document_loader import load_documents, load_from_text
from modules.vectorstore import create_vector_store
from modules.retriever import create_retriever
from modules.chain import create_rag_chain

# Load environment variables
load_dotenv()

# Configure the page
st.set_page_config(
    page_title="SmartDoc AI - Intelligent Document Q&A",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed"
)

def get_current_timestamp():
    """Get current timestamp in a readable format."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def process_question(question):
    """Process a single question using the RAG chain."""
    with st.spinner("🧠 Activating AI Intelligence..."):
        try:
            # Get answer from RAG chain
            answer = st.session_state.rag_chain.invoke(question)
            
            # Extract source information from the answer
            sources = []
            if hasattr(answer, 'metadata') and 'sources' in answer.metadata:
                sources = answer.metadata['sources']
            elif hasattr(answer, 'source_documents'):
                # Extract unique source names from source documents
                for doc in answer.source_documents:
                    if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                        source_name = doc.metadata['source']
                        if source_name not in sources:
                            sources.append(source_name)
            
            # If no sources found, use document sources as fallback
            if not sources and st.session_state.document_sources:
                sources = list(st.session_state.document_sources.keys())
            
            # Format source text for display
            sources_text = ""
            if sources:
                if len(sources) == 1:
                    sources_text = f"<br><small style='color: #cbd5e1; font-style: italic;'>📄 Source: {sources[0]}</small>"
                else:
                    sources_text = f"<br><small style='color: #cbd5e1; font-style: italic;'>📄 Sources: {', '.join(sources)}</small>"
            
            # Display answer with source attribution
            st.markdown(f"""
            <div class="answer-container">
                <h4>💡 AI Intelligence Response:</h4>
                <div class="answer-content">{answer}</div>
                {sources_text}
            </div>
            """, unsafe_allow_html=True)
            
            # Store in chat history
            chat_item = {
                "question": question,
                "answer": str(answer),
                "sources": sources,
                "timestamp": get_current_timestamp()
            }
            st.session_state.chat_history.append(chat_item)
            
            # Show success message
            st.success("✅ Insight discovered and added to conversation memory!")
            
            # Clear the question input by refreshing the page
            st.rerun()
            
        except Exception as e:
            st.error(f"Error generating answer: {str(e)}")
            # Reset the last question to allow retry
            st.session_state.last_question = ""

# Initialize session state
if "documents" not in st.session_state:
    st.session_state.documents = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = False
if "processed" not in st.session_state:
    st.session_state.processed = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "document_sources" not in st.session_state:
    st.session_state.document_sources = {}
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "current_time" not in st.session_state:
    st.session_state.current_time = get_current_timestamp()

# Custom CSS for premium enterprise-grade design
st.markdown("""
<style>
    /* Import Google Fonts for premium typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
    
    /* CSS Variables for consistent theming */
    :root {
        --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --secondary-gradient: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        --accent-gradient: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        --success-gradient: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        --warning-gradient: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        --error-gradient: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
        --glass-bg: rgba(255, 255, 255, 0.1);
        --glass-border: rgba(255, 255, 255, 0.2);
        --shadow-light: 0 8px 32px rgba(31, 38, 135, 0.37);
        --shadow-medium: 0 12px 40px rgba(31, 38, 135, 0.45);
        --shadow-heavy: 0 20px 60px rgba(31, 38, 135, 0.55);
    }
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    section[data-testid="stSidebar"] {display: none !important;}
    .css-1d391kg {display: none !important;}
    .css-1lcbmhc {display: none !important;}
    
    /* Global styles with premium typography */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        box-sizing: border-box;
    }
    
    body, .main, .stApp {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 25%, #16213e 50%, #0f3460 75%, #533483 100%);
        color: #ffffff;
        min-height: 100vh;
        overflow-x: hidden;
    }
    
    /* Premium scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #667eea, #764ba2);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #764ba2, #667eea);
    }
    
    /* Premium header with glassmorphism */
    .main-header {
        background: var(--glass-bg);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid var(--glass-border);
        border-radius: 30px;
        padding: 4rem 3rem;
        margin: 3rem auto;
        text-align: center;
        box-shadow: var(--shadow-medium);
        position: relative;
        overflow: hidden;
        max-width: 1200px;
        animation: slideInDown 1s ease-out;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(45deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        z-index: 1;
    }
    
    .main-header::after {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(102, 126, 234, 0.1) 0%, transparent 70%);
        animation: rotate 20s linear infinite;
        z-index: 0;
    }
    
    .main-header h1, .main-header h3, .main-header p {
        position: relative;
        z-index: 2;
    }
    
    .main-header h1 {
        font-size: 4rem;
        font-weight: 800;
        margin-bottom: 1rem;
        background: var(--primary-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-shadow: 0 0 30px rgba(102, 126, 234, 0.5);
        animation: glow 2s ease-in-out infinite alternate;
    }
    
    .main-header h3 {
        font-size: 1.8rem;
        font-weight: 500;
        margin-bottom: 1.5rem;
        color: #e2e8f0;
        opacity: 0.9;
    }
    
    .main-header p {
        font-size: 1.2rem;
        color: #cbd5e1;
        max-width: 700px;
        margin: 0 auto;
        line-height: 1.6;
    }
    
    /* Premium welcome section */
    .welcome-section {
        background: var(--glass-bg);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid var(--glass-border);
        border-radius: 25px;
        padding: 4rem 3rem;
        margin: 3rem auto;
        text-align: center;
        box-shadow: var(--shadow-medium);
        max-width: 1000px;
        animation: slideInUp 1s ease-out 0.2s both;
        position: relative;
        overflow: hidden;
    }
    
    .welcome-section::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: var(--primary-gradient);
        animation: slideInLeft 1.5s ease-out 0.5s both;
    }
    
    .welcome-section h2 {
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 1.5rem;
        background: var(--primary-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .welcome-section p {
        font-size: 1.3rem;
        color: #e2e8f0;
        margin-bottom: 2rem;
        max-width: 700px;
        margin-left: auto;
        margin-right: auto;
        line-height: 1.7;
    }
    
    /* Premium feature cards with hover effects */
    .features-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 2rem;
        margin: 3rem auto;
        max-width: 1200px;
        animation: slideInUp 1s ease-out 0.4s both;
    }
    
    .feature-card {
        background: var(--glass-bg);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid var(--glass-border);
        border-radius: 20px;
        padding: 3rem 2rem;
        text-align: center;
        box-shadow: var(--shadow-light);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
        cursor: pointer;
    }
    
    .feature-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
        transition: left 0.6s;
    }
    
    .feature-card:hover::before {
        left: 100%;
    }
    
    .feature-card:hover {
        transform: translateY(-10px) scale(1.02);
        box-shadow: var(--shadow-heavy);
        border-color: rgba(102, 126, 234, 0.5);
    }
    
    .feature-card h4 {
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 1.5rem;
        color: #ffffff;
        position: relative;
        z-index: 1;
    }
    
    .feature-card p {
        color: #cbd5e1;
        font-size: 1.1rem;
        line-height: 1.6;
        margin: 0;
        position: relative;
        z-index: 1;
    }
    
    /* Premium upload section */
    .upload-section {
        background: var(--glass-bg);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid var(--glass-border);
        border-radius: 25px;
        padding: 4rem 3rem;
        margin: 3rem auto;
        text-align: center;
        box-shadow: var(--shadow-medium);
        max-width: 900px;
        animation: slideInUp 1s ease-out 0.6s both;
        position: relative;
        overflow: hidden;
    }
    
    .upload-section::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: var(--secondary-gradient);
        animation: slideInRight 1.5s ease-out 0.7s both;
    }
    
    .upload-section h2 {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 1.5rem;
        background: var(--secondary-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .upload-section p {
        font-size: 1.2rem;
        color: #cbd5e1;
        margin-bottom: 3rem;
        line-height: 1.6;
    }
    
    /* Premium file uploader */
    .stFileUploader > div > div {
        border: 2px dashed rgba(102, 126, 234, 0.5);
        border-radius: 20px;
        background: rgba(255, 255, 255, 0.05);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        color: #ffffff !important;
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
    }
    
    .stFileUploader > div > div:hover {
        border-color: #667eea;
        background: rgba(102, 126, 234, 0.1);
        transform: translateY(-2px);
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    
    .stFileUploader > div > div > button {
        background: var(--primary-gradient) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 600 !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
    }
    
    .stFileUploader > div > div > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6) !important;
    }
    
    /* Premium text area */
    .stTextArea > div > div > textarea {
        border: 2px solid rgba(102, 126, 234, 0.3);
        border-radius: 15px;
        background: rgba(255, 255, 255, 0.05);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        font-size: 1rem;
        padding: 1.5rem;
        color: #ffffff !important;
        font-weight: 500;
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
    }
    
    .stTextArea > div > div > textarea:focus {
        border-color: #667eea;
        background: rgba(255, 255, 255, 0.1);
        box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.2);
        outline: none;
        transform: translateY(-1px);
    }
    
    .stTextArea > div > div > textarea::placeholder {
        color: rgba(203, 213, 225, 0.7) !important;
    }
    
    /* Premium buttons */
    .stButton > button {
        background: var(--primary-gradient);
        color: white;
        border: none;
        border-radius: 15px;
        padding: 1rem 2.5rem;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
        transition: left 0.6s;
    }
    
    .stButton > button:hover::before {
        left: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.6);
    }
    
    .stButton > button:active {
        transform: translateY(-1px);
    }
    
    /* Premium chat container */
    .chat-container {
        background: var(--glass-bg);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid var(--glass-border);
        border-radius: 25px;
        padding: 3rem;
        margin: 3rem auto;
        box-shadow: var(--shadow-medium);
        max-width: 1000px;
        animation: slideInUp 1s ease-out 0.8s both;
    }
    
    .chat-header {
        text-align: center;
        margin-bottom: 3rem;
        padding-bottom: 2rem;
        border-bottom: 2px solid rgba(102, 126, 234, 0.3);
        position: relative;
    }
    
    .chat-header::after {
        content: '';
        position: absolute;
        bottom: -2px;
        left: 50%;
        transform: translateX(-50%);
        width: 100px;
        height: 2px;
        background: var(--primary-gradient);
        border-radius: 1px;
    }
    
    .chat-header h2 {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
        background: var(--primary-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .chat-header p {
        font-size: 1.2rem;
        color: #cbd5e1;
        margin: 0;
        line-height: 1.6;
    }
    
    /* Premium question input */
    .stTextInput > div > div > input {
        border: 2px solid rgba(102, 126, 234, 0.3);
        border-radius: 15px;
        background: rgba(255, 255, 255, 0.05);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        color: #ffffff !important;
        font-size: 1.1rem;
        padding: 1rem 1.5rem;
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        background: rgba(255, 255, 255, 0.1);
        box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.2);
        transform: translateY(-1px);
    }
    
    .stTextInput > div > div > input::placeholder {
        color: rgba(203, 213, 225, 0.7) !important;
    }
    
    /* Premium answer display */
    .answer-container {
        background: var(--glass-bg);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid var(--glass-border);
        border-radius: 20px;
        padding: 2.5rem;
        margin: 2rem 0;
        border-left: 5px solid #667eea;
        box-shadow: var(--shadow-light);
        position: relative;
        animation: slideInRight 0.6s ease-out;
    }
    
    .answer-container::before {
        content: '💡';
        position: absolute;
        top: -15px;
        left: 25px;
        background: var(--glass-bg);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        padding: 8px 15px;
        border-radius: 25px;
        font-size: 1.3rem;
        box-shadow: var(--shadow-light);
        border: 1px solid var(--glass-border);
    }
    
    .answer-content {
        margin-top: 1.5rem;
        line-height: 1.7;
        color: #ffffff;
        font-size: 1.1rem;
    }
    
    /* Premium status indicators */
    .status-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 2rem;
        margin: 3rem auto;
        max-width: 1000px;
        animation: slideInUp 1s ease-out 1s both;
    }
    
    .status-card {
        background: var(--glass-bg);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid var(--glass-border);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        box-shadow: var(--shadow-light);
        border-top: 4px solid;
        border-image: var(--primary-gradient) 1;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .status-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(45deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        opacity: 0;
        transition: opacity 0.3s;
    }
    
    .status-card:hover::before {
        opacity: 1;
    }
    
    .status-card:hover {
        transform: translateY(-5px);
        box-shadow: var(--shadow-medium);
    }
    
    .status-card h3 {
        color: #cbd5e1;
        font-size: 1rem;
        font-weight: 500;
        margin: 0 0 1rem 0;
        text-transform: uppercase;
        letter-spacing: 1px;
        position: relative;
        z-index: 1;
    }
    
    .status-card .value {
        color: #ffffff;
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0;
        background: var(--primary-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        position: relative;
        z-index: 1;
    }
    
    /* Premium success/error messages */
    .stSuccess {
        background: var(--success-gradient);
        color: white;
        border-radius: 15px;
        padding: 1.5rem 2rem;
        border: none;
        box-shadow: var(--shadow-medium);
        margin: 2rem auto;
        max-width: 900px;
        font-weight: 600;
        animation: slideInDown 0.6s ease-out;
    }
    
    .stError {
        background: var(--error-gradient);
        color: white;
        border-radius: 15px;
        padding: 1.5rem 2rem;
        border: none;
        box-shadow: var(--shadow-medium);
        margin: 2rem auto;
        max-width: 900px;
        font-weight: 600;
        animation: slideInDown 0.6s ease-out;
    }
    
    .stWarning {
        background: var(--warning-gradient);
        color: white;
        border-radius: 15px;
        padding: 1.5rem 2rem;
        border: none;
        box-shadow: var(--shadow-medium);
        margin: 2rem auto;
        max-width: 900px;
        font-weight: 600;
        animation: slideInDown 0.6s ease-out;
    }
    
    /* Premium animations */
    @keyframes slideInDown {
        from {
            opacity: 0;
            transform: translateY(-50px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes slideInUp {
        from {
            opacity: 0;
            transform: translateY(50px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes slideInLeft {
        from {
            opacity: 0;
            transform: translateX(-50px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateX(50px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    @keyframes glow {
        from {
            text-shadow: 0 0 30px rgba(102, 126, 234, 0.5);
        }
        to {
            text-shadow: 0 0 50px rgba(102, 126, 234, 0.8);
        }
    }
    
    @keyframes rotate {
        from {
            transform: rotate(0deg);
        }
        to {
            transform: rotate(360deg);
        }
    }
    
    /* Premium responsive design */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 2.5rem;
        }
        
        .main-header h3 {
            font-size: 1.4rem;
        }
        
        .chat-container, .upload-section, .welcome-section {
            margin: 1.5rem;
            padding: 2rem;
        }
        
        .status-container {
            grid-template-columns: 1fr;
            gap: 1.5rem;
        }
        
        .features-grid {
            grid-template-columns: 1fr;
            gap: 1.5rem;
        }
        
        .main-header, .welcome-section, .upload-section {
            padding: 2rem 1.5rem;
        }
    }
    
    /* Premium loading animations */
    .stSpinner > div {
        background: var(--primary-gradient) !important;
        border-radius: 50% !important;
        animation: pulse 1.5s ease-in-out infinite !important;
    }
    
    @keyframes pulse {
        0%, 100% {
            opacity: 1;
            transform: scale(1);
        }
        50% {
            opacity: 0.5;
            transform: scale(1.1);
        }
    }
    
    /* Premium focus states */
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        outline: none;
        border-color: #667eea;
        box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.2);
    }
    
    /* Premium hover effects for all interactive elements */
    .stButton > button:hover,
    .stFileUploader > div > div:hover,
    .stTextInput > div > div > input:hover,
    .stTextArea > div > div > textarea:hover {
        transform: translateY(-2px);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
</style>
""", unsafe_allow_html=True)

def display_chat_history():
    """Display the conversation history with source attribution."""
    if st.session_state.chat_history:
        st.markdown("""
        <div class="chat-container">
            <div class="chat-header">
                <h2>📚 Conversation History</h2>
                <p>Your previous questions and AI responses</p>
            </div>
        """, unsafe_allow_html=True)
        
        for i, chat_item in enumerate(st.session_state.chat_history):
            # Question
            st.markdown(f"""
            <div style="background: rgba(102, 126, 234, 0.1); border-radius: 15px; padding: 1.5rem; margin: 1rem 0; border-left: 4px solid #667eea;">
                <div style="font-weight: 600; color: #667eea; margin-bottom: 0.5rem;">🤔 Question {i+1}:</div>
                <div style="color: #ffffff;">{chat_item['question']}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Answer with source attribution
            sources_text = ""
            if chat_item.get('sources'):
                if len(chat_item['sources']) == 1:
                    sources_text = f"<br><small style='color: #cbd5e1; font-style: italic;'>📄 Source: {chat_item['sources'][0]}</small>"
                else:
                    sources_text = f"<br><small style='color: #cbd5e1; font-style: italic;'>📄 Sources: {', '.join(chat_item['sources'])}</small>"
            
            st.markdown(f"""
            <div class="answer-container">
                <h4>💡 Answer:</h4>
                <div class="answer-content">{chat_item['answer']}</div>
                {sources_text}
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

# Main header
st.markdown("""
<div class="main-header">
    <h1>SmartDoc AI</h1>
    <h3>Enterprise Document Intelligence Platform</h3>
    <p>Transform your document chaos into actionable intelligence. Get instant, accurate answers from any document with AI that understands context, remembers conversations, and delivers insights in seconds.</p>
</div>
""", unsafe_allow_html=True)

# Main content area
if st.session_state.processed:
    # Document summary with enhanced information
    st.markdown("""
    <div class="status-container">
        <div class="status-card">
            <h3>Documents</h3>
            <div class="value">{}</div>
        </div>
        <div class="status-card">
            <h3>Chunks</h3>
            <div class="value">{}</div>
        </div>
        <div class="status-card">
            <h3>Conversations</h3>
            <div class="value">{}</div>
        </div>
        <div class="status-card">
            <h3>Status</h3>
            <div class="value">Ready</div>
        </div>
    </div>
    """.format(
        len(st.session_state.document_sources), 
        sum(info['chunks'] for info in st.session_state.document_sources.values()),
        len(st.session_state.chat_history)
    ), unsafe_allow_html=True)
    
    # Display chat history if exists
    if st.session_state.chat_history:
        display_chat_history()
    
    # Q&A Section
    st.markdown("""
    <div class="chat-container">
        <div class="chat-header">
            <h2>💬 Unlock Your Document Intelligence</h2>
            <p>Ask any question about your documents and get instant, accurate answers powered by enterprise-grade AI. Our system understands context, remembers conversations, and delivers insights that drive better decisions.</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Show current document context
    if st.session_state.document_sources:
        st.info(f"""
        **📚 Current Document Context:**
        - **Active documents**: {len(st.session_state.document_sources)}
        - **Document sources**: {', '.join(st.session_state.document_sources.keys())}
        - **Total chunks**: {sum(info['chunks'] for info in st.session_state.document_sources.values())}
        """)
    
    # Simple question input without form
    question = st.text_input(
        "What would you like to discover?",
        placeholder="Ask any question about your documents to unlock insights...",
        key=f"question_input_{len(st.session_state.chat_history)}"
    )
    
    # Process question when Enter is pressed or button is clicked
    col1, col2 = st.columns([3, 1])
    with col1:
        if st.button("🔍 Discover Insights", use_container_width=True):
            if question and question.strip():
                process_question(question)
    with col2:
        if st.button("🗑️ Clear", use_container_width=True):
            # Use rerun to clear the input instead of modifying session state
            st.rerun()
    
    # Process question when Enter is pressed
    if question and question.strip():
        if "last_question" not in st.session_state or st.session_state.last_question != question:
            st.session_state.last_question = question
            process_question(question)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Clear chat history button
    if st.session_state.chat_history:
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("🗑️ Clear Chat History", use_container_width=True, type="secondary"):
                st.session_state.chat_history = []
                st.success("Chat history cleared!")
                st.rerun()

else:
    # Welcome message - MOVED TO TOP
    st.markdown("""
    <div class="welcome-section">
        <h2>Stop Searching, Start Discovering</h2>
        <p>Transform hours of document hunting into seconds of insight. Our enterprise-grade AI understands your content, remembers your conversations, and delivers precise answers that drive better decisions.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Features showcase - MOVED TO TOP
    st.markdown("""
    <div class="features-grid">
        <div class="feature-card">
            <h4>🔍 Intelligent Search & Retrieval</h4>
            <p>Find the exact information you need with our hybrid AI system that combines semantic understanding with context-aware ranking for 95%+ accuracy.</p>
        </div>
        <div class="feature-card">
            <h4>⚡ Lightning-Fast Processing</h4>
            <p>Powered by Groq's ultra-fast inference engine, get answers in milliseconds instead of minutes. Process documents 10x faster than traditional methods.</p>
        </div>
        <div class="feature-card">
            <h4>🧠 Context-Aware Intelligence</h4>
            <p>Our AI doesn't just read—it understands. Preserves document context, maintains conversation memory, and delivers insights that make sense.</p>
        </div>
        <div class="feature-card">
            <h4>🎯 Enterprise-Grade Accuracy</h4>
            <p>Built for professionals who need reliable results. Advanced RAG technology ensures answers are grounded in your actual content, not generic responses.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Supported formats - MOVED TO TOP
    st.markdown("""
    <div class="chat-container">
        <div class="chat-header">
            <h3>Universal Document Intelligence</h3>
            <p>From contracts to spreadsheets, our AI understands every format. No more switching between tools—get intelligent insights from all your business documents in one place.</p>
        </div>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-top: 2rem;">
            <div style="text-align: center; padding: 1.5rem; background: rgba(102, 126, 234, 0.1); border-radius: 15px; border: 1px solid rgba(102, 126, 234, 0.3);">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">📄</div>
                <strong style="color: #ffffff;">PDF Documents</strong>
                <p style="color: #cbd5e1; font-size: 0.9rem; margin: 0.5rem 0 0 0;">Contracts, reports, manuals</p>
            </div>
            <div style="text-align: center; padding: 1.5rem; background: rgba(102, 126, 234, 0.1); border-radius: 15px; border: 1px solid rgba(102, 126, 234, 0.3);">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">📝</div>
                <strong style="color: #ffffff;">Word Documents</strong>
                <p style="color: #cbd5e1; font-size: 0.9rem; margin: 0.5rem 0 0 0;">Proposals, policies, guides</p>
            </div>
            <div style="text-align: center; padding: 1.5rem; background: rgba(102, 126, 234, 0.1); border-radius: 15px; border: 1px solid rgba(102, 126, 234, 0.3);">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">📊</div>
                <strong style="color: #ffffff;">Excel Spreadsheets</strong>
                <p style="color: #cbd5e1; font-size: 0.9rem; margin: 0.5rem 0 0 0;">Data analysis, reports</p>
            </div>
            <div style="text-align: center; padding: 1.5rem; background: rgba(102, 126, 234, 0.1); border-radius: 15px; border: 1px solid rgba(102, 126, 234, 0.3);">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">📃</div>
                <strong style="color: #ffffff;">Text Files</strong>
                <p style="color: #cbd5e1; font-size: 0.9rem; margin: 0.5rem 0 0 0;">Notes, logs, scripts</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Upload section (moved below welcome and features)
    st.markdown("""
    <div class="upload-section">
        <h2>Ready to Transform Your Workflow?</h2>
        <p>Upload your documents or enter text to experience the future of document intelligence. Our AI will process your content and unlock insights you never knew existed.</p>
    """, unsafe_allow_html=True)
    
    # File upload
    st.markdown("**📁 Choose Your Documents:**")
    
    # Upload progress indicator
    if "upload_progress" not in st.session_state:
        st.session_state.upload_progress = 0
        st.session_state.upload_status = "Ready to upload"
    
    # File uploader with progress indicator
    uploaded_files = st.file_uploader(
        "Select files to upload",
        type=['pdf', 'docx', 'doc', 'xlsx', 'xls', 'txt'],
        accept_multiple_files=True,
        help="Upload PDFs, Word documents, Excel spreadsheets, and text files for instant AI analysis",
        key="file_uploader",
        on_change=lambda: st.session_state.update({"upload_progress": 100, "upload_status": "Upload complete!"})
    )
    
    # Show upload progress
    if uploaded_files:
        # Simulate upload progress (since Streamlit doesn't provide real-time upload progress)
        if st.session_state.upload_progress < 100:
            st.session_state.upload_progress = 100
            st.session_state.upload_status = "Upload complete!"
        
        # Upload progress bar
        st.markdown("""
        <div style="background: rgba(102, 126, 234, 0.1); border-radius: 15px; padding: 1.5rem; margin: 1rem 0; border-left: 4px solid #667eea;">
            <h4 style="color: #667eea; margin-bottom: 1rem;">📤 File Upload Progress</h4>
        """, unsafe_allow_html=True)
        
        # Progress bar
        progress_bar = st.progress(st.session_state.upload_progress / 100)
        st.markdown(f"<p style='color: #cbd5e1; text-align: center;'>{st.session_state.upload_status}</p>", unsafe_allow_html=True)
        
        # File list with details
        st.markdown("<h5 style='color: #ffffff; margin: 1rem 0;'>📋 Uploaded Files:</h5>", unsafe_allow_html=True)
        
        for i, file in enumerate(uploaded_files):
            file_size = len(file.read()) / 1024  # Size in KB
            file.seek(0)  # Reset file pointer for later use
            
            st.markdown(f"""
            <div style="margin: 0.5rem 0; padding: 0.5rem; background: rgba(255, 255, 255, 0.05); border-radius: 10px;">
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <span style="color: #10b981;">✅</span>
                    <strong style="color: #ffffff;">{file.name}</strong>
                    <span style="color: #cbd5e1; font-size: 0.9rem;">({file_size:.1f} KB)</span>
                    <span style="color: #10b981; font-size: 0.8rem;">✓ Uploaded</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Reset progress for next upload
        if st.session_state.upload_progress == 100:
            st.session_state.upload_progress = 0
            st.session_state.upload_status = "Ready for next upload"
    
    # Manual text input
    st.markdown("**✍️ Or Enter Text Directly:**")
    manual_text = st.text_area(
        "Paste your content here:",
        height=150,
        placeholder="Enter or paste your document content, notes, or any text you'd like to analyze with AI intelligence..."
    )
    
    # Process button
    if st.button("🚀 Process & Activate AI Intelligence", type="primary", use_container_width=True):
        if uploaded_files or manual_text.strip():
            # Clear ALL old data completely before processing new documents
            st.session_state.document_sources = {}
            st.session_state.chat_history = []
            st.session_state.last_question = ""
            st.session_state.vector_store = None
            st.session_state.retriever = None
            st.session_state.rag_chain = None
            st.session_state.documents = []
            st.session_state.processed = False
            
            # Force garbage collection to clear any cached data
            import gc
            gc.collect()
            
            # Clear any cached embeddings or models
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except:
                pass
            
            # Initialize progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                all_docs = []
                document_sources = {}
                
                # Process uploaded files with progress
                if uploaded_files:
                    status_text.text("📄 Loading uploaded files...")
                    progress_bar.progress(10)
                    
                    for i, uploaded_file in enumerate(uploaded_files):
                        # Show individual file progress
                        file_progress = 10 + (i / len(uploaded_files)) * 30
                        status_text.text(f"📄 Processing {uploaded_file.name}...")
                        progress_bar.progress(int(file_progress))
                        
                        file_content = uploaded_file.read()
                        docs = load_documents(file_content, uploaded_file.name)
                        all_docs.extend(docs)
                        
                        # Store source information
                        for doc in docs:
                            if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                                source_name = doc.metadata['source']
                                document_sources[source_name] = {
                                    'type': doc.metadata.get('type', 'unknown'),
                                    'chunks': len(docs)
                                }
                
                # Process manual text
                if manual_text.strip():
                    status_text.text("✍️ Processing manual text input...")
                    progress_bar.progress(45)
                    
                    manual_docs = load_from_text(manual_text)
                    all_docs.extend(manual_docs)
                    
                    # Store manual text source
                    document_sources["Manual Input"] = {
                        'type': 'manual',
                        'chunks': len(manual_docs)
                    }
                
                if all_docs:
                    # Store document sources in session state
                    st.session_state.document_sources = document_sources
                    
                    # Process documents silently
                    
                    # Chunk documents
                    status_text.text("✂️ Chunking documents...")
                    progress_bar.progress(60)
                    
                    chunks = chunk_data(all_docs)
                    
                    # Create vector store
                    status_text.text("🧠 Creating vector embeddings...")
                    progress_bar.progress(80)
                    
                    vector_store = create_vector_store(chunks)
                    st.session_state.vector_store = vector_store
                    
                    # Create retriever and RAG chain
                    status_text.text("🔗 Building RAG system...")
                    progress_bar.progress(90)
                    
                    st.session_state.retriever = create_retriever(vector_store)
                    st.session_state.rag_chain = create_rag_chain(st.session_state.retriever)
                    st.session_state.documents = all_docs
                    st.session_state.processed = True
                    
                    # Show success message
                    st.success("🎉 AI Intelligence Successfully Activated! Your documents are now ready for questions.")
                    
                    st.rerun()
                else:
                    st.error("No valid documents found.")
                    
            except Exception as e:
                error_msg = str(e)
                st.error(f"Error processing documents: {error_msg}")
                    
            finally:
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
        else:
            st.warning("Please upload files or enter text to activate AI intelligence.")
    
    # Test mode button for UI demonstration
    st.markdown("---")
    if st.button("🎨 Experience the Interface (Demo Mode)", use_container_width=True, help="See the full interface without downloading AI models"):
        st.session_state.documents = [{"source": "Demo Document", "type": "demo"}]
        st.session_state.document_sources = {"Demo Document": {"type": "demo", "chunks": 3}}
        st.session_state.processed = True
        st.success("🎯 Demo mode activated! Experience the full interface and test conversation features.")
        st.rerun()
    
    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748b; padding: 2rem;">
    <p>Built with Streamlit, LangChain, and Groq • Advanced RAG System</p>
</div>
""", unsafe_allow_html=True)