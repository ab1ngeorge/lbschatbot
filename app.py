# app.py - Enhanced LBSCEK RAG Chatbot
import streamlit as st
import os
import logging
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import json
import hashlib

from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.callbacks import StreamlitCallbackHandler

# --------------- LOGGING SETUP ---------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------------- PAGE CONFIG ---------------
st.set_page_config(
    page_title="LBSCEK AI Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------- CUSTOM CSS ---------------
st.markdown("""
<style>
    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Chat messages */
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.75rem;
        margin-bottom: 0.5rem;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        margin-bottom: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2rem;
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    
    /* Quick question buttons */
    .quick-btn {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 2rem;
        color: white;
        margin: 0.25rem;
        cursor: pointer;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .quick-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* Source cards */
    .source-card {
        background: #f8f9fa;
        border-left: 4px solid #667eea;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0 0.5rem 0.5rem 0;
    }
    
    /* Stats cards */
    .stat-card {
        background: white;
        padding: 1rem;
        border-radius: 0.75rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        padding-top: 1rem;
    }
    
    /* Success/Error messages */
    .success-msg {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        padding: 0.75rem;
        border-radius: 0.5rem;
        color: #155724;
    }
    
    .error-msg {
        background: #f8d7da;
        border: 1px solid #f5c6cb;
        padding: 0.75rem;
        border-radius: 0.5rem;
        color: #721c24;
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --------------- CONSTANTS ---------------
VECTORSTORE_PATH = "lbscek_vectorstore"
DEFAULT_MODEL = "gpt-4o-mini"
AVAILABLE_MODELS = ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]

# --------------- LBSCEK PAGES ---------------
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_lbs_links() -> List[str]:
    """Comprehensive list of LBSCEK pages"""
    return [
        # Main pages
        "https://lbscek.ac.in/",
        "https://lbscek.ac.in/about-us/",
        "https://lbscek.ac.in/vision-mission/",
        
        # Academics
        "https://lbscek.ac.in/departments/",
        "https://lbscek.ac.in/faculty/",
        "https://lbscek.ac.in/academic-calendar/",
        "https://lbscek.ac.in/programmes/",
        
        # Departments (if available)
        "https://lbscek.ac.in/department-of-computer-science/",
        "https://lbscek.ac.in/department-of-electronics/",
        "https://lbscek.ac.in/department-of-electrical/",
        "https://lbscek.ac.in/department-of-mechanical/",
        "https://lbscek.ac.in/department-of-civil/",
        
        # Admissions & Placements
        "https://lbscek.ac.in/admission/",
        "https://lbscek.ac.in/placements/",
        "https://lbscek.ac.in/training-placement-cell/",
        
        # Facilities & Infrastructure
        "https://lbscek.ac.in/facilities/",
        "https://lbscek.ac.in/library/",
        "https://lbscek.ac.in/hostel/",
        "https://lbscek.ac.in/laboratory/",
        
        # Student Life
        "https://lbscek.ac.in/student-activities/",
        "https://lbscek.ac.in/clubs/",
        "https://lbscek.ac.in/events/",
        
        # Others
        "https://lbscek.ac.in/research/",
        "https://lbscek.ac.in/contact-us/",
        "https://lbscek.ac.in/gallery/",
    ]

# --------------- QUICK QUESTIONS ---------------
QUICK_QUESTIONS = [
    "📚 What courses are offered?",
    "🎯 How to apply for admission?",
    "💼 What are placement statistics?",
    "👨‍🏫 Tell me about faculty",
    "🏛️ What facilities are available?",
    "📍 Where is the college located?",
]

# --------------- CUSTOM PROMPT ---------------
CUSTOM_PROMPT = PromptTemplate(
    input_variables=["context", "question", "chat_history"],
    template="""You are the official AI assistant for LBS College of Engineering Kasaragod (LBSCEK). 
You provide accurate, helpful information about the college based ONLY on the official website content.

Guidelines:
- Be friendly, professional, and informative
- Only use information from the provided context
- If you don't find specific information, say so honestly
- Provide detailed answers when information is available
- Format responses clearly with bullet points or numbered lists when appropriate
- Include relevant links or references when available
- If asked about something not related to LBSCEK, politely redirect to college topics

Chat History:
{chat_history}

Context from LBSCEK Website:
{context}

Question: {question}

Helpful Answer:"""
)

# --------------- SESSION STATE INITIALIZATION ---------------
def init_session_state():
    """Initialize all session state variables"""
    defaults = {
        "messages": [],
        "api_key_set": False,
        "api_key": "",
        "vectorstore": None,
        "qa_chain": None,
        "chat_history": [],
        "total_questions": 0,
        "model": DEFAULT_MODEL,
        "temperature": 0.1,
        "show_sources": True,
        "vectorstore_loaded": False,
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# --------------- HELPER FUNCTIONS ---------------
def validate_api_key(api_key: str) -> bool:
    """Validate OpenAI API key format"""
    if not api_key:
        return False
    return api_key.startswith("sk-") and len(api_key) > 20

def get_cache_key(urls: List[str]) -> str:
    """Generate cache key for vectorstore"""
    return hashlib.md5("".join(sorted(urls)).encode()).hexdigest()

def load_and_process_documents(urls: List[str]) -> List:
    """Load documents from URLs with error handling"""
    all_docs = []
    failed_urls = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, url in enumerate(urls):
        try:
            status_text.text(f"Loading: {url}")
            loader = WebBaseLoader([url])
            loader.requests_kwargs = {'timeout': 10}
            docs = loader.load()
            all_docs.extend(docs)
            logger.info(f"Successfully loaded: {url}")
        except Exception as e:
            logger.warning(f"Failed to load {url}: {str(e)}")
            failed_urls.append(url)
        
        progress_bar.progress((i + 1) / len(urls))
    
    progress_bar.empty()
    status_text.empty()
    
    if failed_urls:
        st.warning(f"⚠️ Could not load {len(failed_urls)} pages. Using available content.")
        with st.expander("Failed URLs"):
            for url in failed_urls:
                st.text(url)
    
    return all_docs

@st.cache_resource(show_spinner=False)
def build_vectorstore(_api_key: str) -> Optional[FAISS]:
    """Build or load FAISS vectorstore"""
    try:
        os.environ["OPENAI_API_KEY"] = _api_key
        urls = get_lbs_links()
        
        # Check for cached vectorstore
        cache_key = get_cache_key(urls)
        cache_path = f"{VECTORSTORE_PATH}_{cache_key}"
        
        embeddings = OpenAIEmbeddings()
        
        # Try to load existing vectorstore
        if os.path.exists(cache_path):
            try:
                vectorstore = FAISS.load_local(
                    cache_path, 
                    embeddings,
                    allow_dangerous_deserialization=True
                )
                logger.info("Loaded cached vectorstore")
                return vectorstore
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        
        # Build new vectorstore
        with st.status("🔄 Building knowledge base...", expanded=True) as status:
            st.write("📥 Loading LBSCEK website pages...")
            docs = load_and_process_documents(urls)
            
            if not docs:
                st.error("❌ No documents could be loaded!")
                return None
            
            st.write(f"✅ Loaded {len(docs)} pages")
            
            st.write("✂️ Splitting documents...")
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            split_docs = splitter.split_documents(docs)
            st.write(f"✅ Created {len(split_docs)} text chunks")
            
            st.write("🧠 Creating embeddings...")
            vectorstore = FAISS.from_documents(split_docs, embeddings)
            
            # Save to cache
            try:
                vectorstore.save_local(cache_path)
                st.write("💾 Saved to cache")
            except Exception as e:
                logger.warning(f"Failed to save cache: {e}")
            
            status.update(label="✅ Knowledge base ready!", state="complete")
        
        return vectorstore
        
    except Exception as e:
        logger.error(f"Error building vectorstore: {e}")
        st.error(f"❌ Error building knowledge base: {str(e)}")
        return None

def get_qa_chain(vectorstore: FAISS, model: str, temperature: float):
    """Create conversational QA chain"""
    try:
        llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            streaming=True,
        )
        
        memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
            k=5  # Remember last 5 exchanges
        )
        
        retriever = vectorstore.as_retriever(
            search_type="mmr",  # Maximum Marginal Relevance
            search_kwargs={"k": 4, "fetch_k": 8}
        )
        
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": CUSTOM_PROMPT},
            verbose=False,
        )
        
        return qa_chain
        
    except Exception as e:
        logger.error(f"Error creating QA chain: {e}")
        raise

def format_sources(sources: List) -> str:
    """Format source documents for display"""
    if not sources:
        return ""
    
    formatted = []
    seen_urls = set()
    
    for doc in sources:
        url = doc.metadata.get("source", "Unknown")
        if url not in seen_urls:
            seen_urls.add(url)
            content = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            formatted.append(f"**Source:** [{url}]({url})\n> {content}")
    
    return "\n\n".join(formatted)

def export_chat_history() -> str:
    """Export chat history as JSON"""
    export_data = {
        "exported_at": datetime.now().isoformat(),
        "total_messages": len(st.session_state.messages),
        "messages": st.session_state.messages
    }
    return json.dumps(export_data, indent=2)

# --------------- SIDEBAR ---------------
def render_sidebar():
    """Render sidebar with settings and info"""
    with st.sidebar:
        st.image("https://via.placeholder.com/250x80?text=LBSCEK", use_container_width=True)
        
        st.markdown("---")
        
        # API Key Section
        st.header("🔑 API Configuration")
        
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=st.session_state.api_key,
            help="Get your key from https://platform.openai.com/api-keys",
            placeholder="sk-..."
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Set Key", use_container_width=True):
                if validate_api_key(api_key):
                    st.session_state.api_key = api_key
                    st.session_state.api_key_set = True
                    os.environ["OPENAI_API_KEY"] = api_key
                    st.success("✅ API Key set!")
                    st.rerun()
                else:
                    st.error("❌ Invalid API key format")
        
        with col2:
            if st.button("🔄 Reset", use_container_width=True):
                st.session_state.api_key = ""
                st.session_state.api_key_set = False
                st.session_state.vectorstore_loaded = False
                st.rerun()
        
        if st.session_state.api_key_set:
            st.success("✅ API Key configured")
        
        st.markdown("---")
        
        # Model Settings
        st.header("⚙️ Settings")
        
        st.session_state.model = st.selectbox(
            "Model",
            AVAILABLE_MODELS,
            index=AVAILABLE_MODELS.index(st.session_state.model),
            help="GPT-4o-mini is faster and cheaper, GPT-4o is more capable"
        )
        
        st.session_state.temperature = st.slider(
            "Creativity",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.temperature,
            step=0.1,
            help="Lower = more focused, Higher = more creative"
        )
        
        st.session_state.show_sources = st.checkbox(
            "Show sources",
            value=st.session_state.show_sources,
            help="Display source documents for answers"
        )
        
        st.markdown("---")
        
        # Chat Stats
        st.header("📊 Session Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Questions", st.session_state.total_questions)
        with col2:
            st.metric("Messages", len(st.session_state.messages))
        
        st.markdown("---")
        
        # Actions
        st.header("🛠️ Actions")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.session_state.chat_history = []
                st.session_state.total_questions = 0
                st.rerun()
        
        with col2:
            if st.button("🔄 Rebuild KB", use_container_width=True):
                st.cache_resource.clear()
                st.session_state.vectorstore_loaded = False
                st.rerun()
        
        # Export chat
        if st.session_state.messages:
            st.download_button(
                "📥 Export Chat",
                export_chat_history(),
                file_name=f"lbscek_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
        
        st.markdown("---")
        
        # Help
        with st.expander("ℹ️ Help"):
            st.markdown("""
            **How to use:**
            1. Enter your OpenAI API key
            2. Wait for knowledge base to load
            3. Ask questions about LBSCEK
            
            **Tips:**
            - Be specific in your questions
            - Use quick questions for common topics
            - Check sources for detailed info
            
            **Supported Topics:**
            - Admissions & Programs
            - Departments & Faculty
            - Placements & Statistics
            - Facilities & Infrastructure
            - Contact Information
            """)
        
        st.markdown("---")
        st.caption("Made with ❤️ for LBSCEK")

# --------------- MAIN CHAT INTERFACE ---------------
def render_header():
    """Render main header"""
    st.markdown("""
    <div class="main-header">
        <h1>🎓 LBS College of Engineering Kasaragod</h1>
        <p>AI-Powered Assistant • Ask anything about LBSCEK</p>
    </div>
    """, unsafe_allow_html=True)

def render_quick_questions():
    """Render quick question buttons"""
    st.markdown("**Quick Questions:**")
    cols = st.columns(3)
    for i, question in enumerate(QUICK_QUESTIONS):
        with cols[i % 3]:
            if st.button(question, key=f"quick_{i}", use_container_width=True):
                return question.split(" ", 1)[1]  # Remove emoji
    return None

def render_chat_messages():
    """Render chat message history"""
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar="🎓" if message["role"] == "assistant" else "👤"):
            st.markdown(message["content"])
            
            # Show sources if available
            if message["role"] == "assistant" and "sources" in message and st.session_state.show_sources:
                with st.expander("📄 View Sources", expanded=False):
                    st.markdown(message["sources"])

def process_question(question: str, qa_chain) -> Tuple[str, str]:
    """Process a question and return answer with sources"""
    try:
        result = qa_chain({"question": question})
        answer = result.get("answer", "I couldn't find an answer to that question.")
        sources = format_sources(result.get("source_documents", []))
        return answer, sources
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        raise

def main():
    """Main application logic"""
    render_sidebar()
    render_header()
    
    # Check API key
    if not st.session_state.api_key_set:
        st.info("👋 Welcome! Please set your OpenAI API key in the sidebar to get started.")
        
        with st.expander("🔐 How to get an API key"):
            st.markdown("""
            1. Go to [OpenAI Platform](https://platform.openai.com)
            2. Sign up or log in
            3. Navigate to API Keys section
            4. Create a new secret key
            5. Copy and paste it in the sidebar
            
            **Note:** You need to have credits in your OpenAI account.
            """)
        return
    
    # Build vectorstore
    if not st.session_state.vectorstore_loaded:
        vectorstore = build_vectorstore(st.session_state.api_key)
        if vectorstore:
            st.session_state.vectorstore = vectorstore
            st.session_state.vectorstore_loaded = True
        else:
            st.error("Failed to build knowledge base. Please check your API key and try again.")
            return
    
    # Quick questions
    quick_question = render_quick_questions()
    
    st.markdown("---")
    
    # Display welcome message if no messages
    if not st.session_state.messages:
        st.session_state.messages.append({
            "role": "assistant",
            "content": """👋 **Welcome to LBSCEK AI Assistant!**

I'm here to help you with information about LBS College of Engineering Kasaragod. You can ask me about:

• 📚 **Academic Programs** - Courses, departments, curriculum
• 🎯 **Admissions** - Eligibility, process, requirements
• 💼 **Placements** - Statistics, companies, training
• 👨‍🏫 **Faculty** - Departments, qualifications
• 🏛️ **Facilities** - Infrastructure, labs, library
• 📍 **Contact** - Location, phone, email

Feel free to ask anything about LBSCEK!""",
            "sources": ""
        })
    
    # Render chat messages
    render_chat_messages()
    
    # Handle quick question or chat input
    user_input = quick_question or st.chat_input("Ask about LBSCEK...")
    
    if user_input:
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })
        
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)
        
        # Generate response
        with st.chat_message("assistant", avatar="🎓"):
            try:
                # Create QA chain
                qa_chain = get_qa_chain(
                    st.session_state.vectorstore,
                    st.session_state.model,
                    st.session_state.temperature
                )
                
                # Process with streaming
                with st.spinner("🔍 Searching LBSCEK knowledge base..."):
                    answer, sources = process_question(user_input, qa_chain)
                
                st.markdown(answer)
                
                # Show sources
                if sources and st.session_state.show_sources:
                    with st.expander("📄 View Sources", expanded=False):
                        st.markdown(sources)
                
                # Save to session state
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })
                
                st.session_state.total_questions += 1
                
            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                st.error(error_msg)
                
                if "api" in str(e).lower() or "key" in str(e).lower():
                    st.info("💡 Please check your OpenAI API key and ensure you have credits.")
                elif "rate" in str(e).lower():
                    st.info("💡 Rate limit reached. Please wait a moment and try again.")
                else:
                    st.info("💡 Try refreshing the page or rebuilding the knowledge base.")
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "sources": ""
                })
        
        st.rerun()

# --------------- RUN APPLICATION ---------------
if __name__ == "__main__":
    main()
