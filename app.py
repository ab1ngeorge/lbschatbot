# app.py - LBSCEK RAG Chatbot (Complete Fixed Version)
import streamlit as st
import os
import sys
import logging
from datetime import datetime
from typing import List, Dict, Optional
import json
import requests

# --------------- LOGGING ---------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------------- PAGE CONFIG (MUST BE FIRST ST COMMAND) ---------------
st.set_page_config(
    page_title="LBSCEK AI Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------- DEPENDENCY CHECK ---------------
def check_dependencies():
    """Check and display missing dependencies"""
    missing = []
    
    try:
        import langchain
    except ImportError:
        missing.append("langchain")
    
    try:
        import langchain_community
    except ImportError:
        missing.append("langchain-community")
    
    try:
        import langchain_openai
    except ImportError:
        missing.append("langchain-openai")
    
    try:
        import faiss
    except ImportError:
        missing.append("faiss-cpu")
    
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        missing.append("beautifulsoup4")
    
    try:
        import tiktoken
    except ImportError:
        missing.append("tiktoken")
    
    return missing

# Check dependencies first
missing_deps = check_dependencies()

if missing_deps:
    st.error("❌ Missing Dependencies Detected!")
    st.markdown(f"""
    ### Missing packages: `{', '.join(missing_deps)}`
    
    **Please create/update your `requirements.txt` file with:**
    
    ```
    streamlit==1.31.0
    langchain==0.1.9
    langchain-community==0.0.24
    langchain-openai==0.0.8
    langchain-text-splitters==0.0.1
    openai==1.12.0
    faiss-cpu==1.7.4
    tiktoken==0.6.0
    beautifulsoup4==4.12.3
    lxml==5.1.0
    requests==2.31.0
    urllib3==2.2.0
    ```
    
    **Steps to fix:**
    1. Create `requirements.txt` in your repo root
    2. Copy the above content
    3. Commit and push to GitHub
    4. Reboot the app (Manage app → Reboot)
    """)
    st.stop()

# --------------- IMPORTS (After dependency check) ---------------
try:
    from langchain_community.document_loaders import WebBaseLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    from langchain.chains import ConversationalRetrievalChain
    from langchain.memory import ConversationBufferWindowMemory
    from langchain.schema import Document
    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    LANGCHAIN_AVAILABLE = False
    IMPORT_ERROR = str(e)

if not LANGCHAIN_AVAILABLE:
    st.error(f"❌ Import Error: {IMPORT_ERROR}")
    st.markdown("""
    ### Please ensure requirements.txt is correct and reboot the app.
    
    Go to **Manage app** → **Reboot app**
    """)
    st.stop()

# --------------- CUSTOM CSS ---------------
st.markdown("""
<style>
    /* Main container */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Header */
    .main-header {
        background: linear-gradient(135deg, #1a365d 0%, #2563eb 100%);
        padding: 2rem;
        border-radius: 1rem;
        color: white;
        margin-bottom: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.2rem;
        font-weight: 700;
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
        font-size: 1.1rem;
    }
    
    /* Chat container */
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.75rem;
    }
    
    /* Buttons */
    .stButton > button {
        border-radius: 0.5rem;
        font-weight: 500;
        transition: all 0.2s;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    
    /* Quick question buttons */
    div[data-testid="column"] .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        font-size: 0.85rem;
    }
    
    /* Info boxes */
    .info-box {
        background: #f0f9ff;
        border-left: 4px solid #2563eb;
        padding: 1rem;
        border-radius: 0 0.5rem 0.5rem 0;
        margin: 1rem 0;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
</style>
""", unsafe_allow_html=True)

# --------------- CONSTANTS ---------------
LBSCEK_URLS = [
    "https://lbscek.ac.in/",
    "https://lbscek.ac.in/about-us/",
    "https://lbscek.ac.in/departments/",
    "https://lbscek.ac.in/faculty/",
    "https://lbscek.ac.in/admission/",
    "https://lbscek.ac.in/placements/",
    "https://lbscek.ac.in/facilities/",
    "https://lbscek.ac.in/contact-us/",
    "https://lbscek.ac.in/programmes/",
    "https://lbscek.ac.in/hostel/",
    "https://lbscek.ac.in/library/",
    "https://lbscek.ac.in/vision-mission/",
]

QUICK_QUESTIONS = [
    ("📚", "What courses are offered at LBSCEK?"),
    ("🎯", "How to apply for admission?"),
    ("💼", "Tell me about placements"),
    ("👨‍🏫", "Who are the faculty members?"),
    ("🏛️", "What facilities are available?"),
    ("📍", "Where is the college located?"),
]

WELCOME_MESSAGE = """👋 **Welcome to LBSCEK AI Assistant!**

I'm here to help you with information about **LBS College of Engineering Kasaragod**.

**I can answer questions about:**
- 📚 Academic programs and courses
- 🎯 Admission process and eligibility
- 💼 Placement statistics and companies
- 👨‍🏫 Faculty and departments
- 🏛️ Facilities and infrastructure
- 📍 Contact information and location

**Try asking:** "What are the departments at LBSCEK?" or click a quick question below!
"""

# --------------- SESSION STATE ---------------
def initialize_session_state():
    """Initialize all session state variables"""
    defaults = {
        "messages": [],
        "api_key": "",
        "api_key_valid": False,
        "vectorstore": None,
        "vectorstore_built": False,
        "qa_chain": None,
        "total_questions": 0,
        "error_count": 0,
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

initialize_session_state()

# --------------- UTILITY FUNCTIONS ---------------
def is_valid_api_key(api_key: str) -> bool:
    """Validate OpenAI API key format"""
    if not api_key:
        return False
    api_key = api_key.strip()
    return api_key.startswith("sk-") and len(api_key) > 30

def fetch_url_content(url: str, timeout: int = 15) -> Optional[str]:
    """Fetch content from URL with error handling"""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        return response.text
    except Exception as e:
        logger.warning(f"Failed to fetch {url}: {e}")
        return None

def load_lbscek_documents() -> List[Document]:
    """Load documents from LBSCEK website"""
    documents = []
    failed_urls = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, url in enumerate(LBSCEK_URLS):
        status_text.text(f"📥 Loading: {url}")
        
        try:
            loader = WebBaseLoader(
                web_paths=[url],
                bs_kwargs={
                    "parse_only": None,
                    "features": "lxml"
                }
            )
            loader.requests_per_second = 1
            docs = loader.load()
            
            # Add metadata
            for doc in docs:
                doc.metadata["source"] = url
                doc.metadata["loaded_at"] = datetime.now().isoformat()
            
            documents.extend(docs)
            logger.info(f"Loaded: {url}")
            
        except Exception as e:
            logger.warning(f"Failed to load {url}: {e}")
            failed_urls.append(url)
        
        progress_bar.progress((i + 1) / len(LBSCEK_URLS))
    
    progress_bar.empty()
    status_text.empty()
    
    if failed_urls:
        st.warning(f"⚠️ Could not load {len(failed_urls)} pages. Using available content.")
        with st.expander("View failed URLs"):
            for url in failed_urls:
                st.text(url)
    
    return documents

def split_documents(documents: List[Document]) -> List[Document]:
    """Split documents into chunks"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return splitter.split_documents(documents)

@st.cache_resource(show_spinner=False)
def create_vectorstore(_api_key: str) -> Optional[FAISS]:
    """Create FAISS vectorstore from documents"""
    try:
        os.environ["OPENAI_API_KEY"] = _api_key
        
        with st.status("🔄 Building Knowledge Base...", expanded=True) as status:
            # Step 1: Load documents
            st.write("📥 **Step 1/3:** Loading LBSCEK website pages...")
            documents = load_lbscek_documents()
            
            if not documents:
                st.error("❌ Failed to load any documents!")
                return None
            
            st.write(f"✅ Loaded **{len(documents)}** pages")
            
            # Step 2: Split documents
            st.write("✂️ **Step 2/3:** Splitting into chunks...")
            chunks = split_documents(documents)
            st.write(f"✅ Created **{len(chunks)}** text chunks")
            
            # Step 3: Create embeddings
            st.write("🧠 **Step 3/3:** Creating embeddings (this may take a minute)...")
            embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small",
                openai_api_key=_api_key
            )
            
            vectorstore = FAISS.from_documents(
                documents=chunks,
                embedding=embeddings
            )
            
            status.update(label="✅ Knowledge Base Ready!", state="complete", expanded=False)
        
        return vectorstore
        
    except Exception as e:
        logger.error(f"Vectorstore creation failed: {e}")
        st.error(f"❌ Error creating knowledge base: {str(e)}")
        return None

def create_qa_chain(vectorstore: FAISS, api_key: str):
    """Create the QA chain"""
    try:
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.1,
            openai_api_key=api_key,
            max_tokens=1000,
        )
        
        memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
            k=5  # Remember last 5 exchanges
        )
        
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )
        
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=True,
            verbose=False,
        )
        
        return qa_chain
        
    except Exception as e:
        logger.error(f"QA chain creation failed: {e}")
        raise

def format_source_documents(sources: List[Document]) -> str:
    """Format source documents for display"""
    if not sources:
        return "_No sources found_"
    
    seen_urls = set()
    formatted_sources = []
    
    for i, doc in enumerate(sources, 1):
        url = doc.metadata.get("source", "Unknown")
        
        if url in seen_urls:
            continue
        seen_urls.add(url)
        
        content_preview = doc.page_content[:200].replace("\n", " ").strip()
        if len(doc.page_content) > 200:
            content_preview += "..."
        
        formatted_sources.append(
            f"**Source {i}:** [{url}]({url})\n"
            f"> {content_preview}"
        )
    
    return "\n\n".join(formatted_sources)

def process_user_question(question: str) -> Dict:
    """Process a user question and return response"""
    try:
        if st.session_state.qa_chain is None:
            st.session_state.qa_chain = create_qa_chain(
                st.session_state.vectorstore,
                st.session_state.api_key
            )
        
        result = st.session_state.qa_chain.invoke({"question": question})
        
        return {
            "success": True,
            "answer": result.get("answer", "I couldn't find an answer to that question."),
            "sources": result.get("source_documents", [])
        }
        
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        return {
            "success": False,
            "answer": f"Sorry, I encountered an error: {str(e)}",
            "sources": []
        }

# --------------- UI COMPONENTS ---------------
def render_header():
    """Render the main header"""
    st.markdown("""
    <div class="main-header">
        <h1>🎓 LBS College of Engineering Kasaragod</h1>
        <p>AI-Powered Assistant • Your Guide to LBSCEK</p>
    </div>
    """, unsafe_allow_html=True)

def render_sidebar():
    """Render the sidebar"""
    with st.sidebar:
        st.markdown("## 🎓 LBSCEK Chatbot")
        st.markdown("---")
        
        # API Key Section
        st.markdown("### 🔑 OpenAI API Key")
        
        api_key_input = st.text_input(
            "Enter your API key",
            type="password",
            value=st.session_state.api_key,
            placeholder="sk-...",
            help="Get your API key from https://platform.openai.com/api-keys",
            label_visibility="collapsed"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("✅ Connect", use_container_width=True, type="primary"):
                if is_valid_api_key(api_key_input):
                    st.session_state.api_key = api_key_input.strip()
                    st.session_state.api_key_valid = True
                    os.environ["OPENAI_API_KEY"] = st.session_state.api_key
                    st.success("Connected!")
                    st.rerun()
                else:
                    st.error("Invalid API key format!")
        
        with col2:
            if st.button("🔄 Reset", use_container_width=True):
                # Clear all session state
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.cache_resource.clear()
                st.rerun()
        
        # Connection status
        if st.session_state.api_key_valid:
            st.success("✅ API Connected")
        else:
            st.warning("⚠️ API Key Required")
        
        st.markdown("---")
        
        # Statistics
        st.markdown("### 📊 Session Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Questions", st.session_state.total_questions)
        with col2:
            st.metric("Messages", len(st.session_state.messages))
        
        st.markdown("---")
        
        # Actions
        st.markdown("### 🛠️ Actions")
        
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.session_state.total_questions = 0
            st.session_state.qa_chain = None
            st.rerun()
        
        if st.button("🔄 Rebuild Knowledge Base", use_container_width=True):
            st.session_state.vectorstore = None
            st.session_state.vectorstore_built = False
            st.session_state.qa_chain = None
            st.cache_resource.clear()
            st.rerun()
        
        # Export chat
        if st.session_state.messages:
            export_data = {
                "exported_at": datetime.now().isoformat(),
                "total_messages": len(st.session_state.messages),
                "messages": st.session_state.messages
            }
            
            st.download_button(
                "📥 Export Chat History",
                data=json.dumps(export_data, indent=2),
                file_name=f"lbscek_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
        
        st.markdown("---")
        
        # Help section
        with st.expander("ℹ️ Help & Info"):
            st.markdown("""
            **How to use:**
            1. Enter your OpenAI API key
            2. Click "Connect"
            3. Wait for knowledge base to build
            4. Start asking questions!
            
            **Get API Key:**
            1. Go to [OpenAI Platform](https://platform.openai.com)
            2. Sign up or log in
            3. Navigate to API Keys
            4. Create a new key
            
            **Topics I know about:**
            - Admissions & Programs
            - Departments & Faculty
            - Placements
            - Facilities
            - Contact Info
            
            **Tips:**
            - Be specific in your questions
            - Use quick questions for common topics
            - Check sources for detailed info
            """)
        
        st.markdown("---")
        st.caption("Made with ❤️ for LBSCEK")

def render_quick_questions() -> Optional[str]:
    """Render quick question buttons and return selected question"""
    st.markdown("**💡 Quick Questions:**")
    
    cols = st.columns(3)
    selected_question = None
    
    for i, (emoji, question) in enumerate(QUICK_QUESTIONS):
        with cols[i % 3]:
            button_label = f"{emoji} {question[:25]}..." if len(question) > 25 else f"{emoji} {question}"
            if st.button(button_label, key=f"quick_{i}", use_container_width=True):
                selected_question = question
    
    return selected_question

def render_chat_messages():
    """Render all chat messages"""
    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]
        avatar = "🎓" if role == "assistant" else "👤"
        
        with st.chat_message(role, avatar=avatar):
            st.markdown(content)
            
            # Show sources if available
            if role == "assistant" and message.get("sources"):
                with st.expander("📄 View Sources", expanded=False):
                    st.markdown(message["sources"])

def render_api_key_prompt():
    """Render the API key prompt for new users"""
    st.info("👋 Welcome! Please enter your OpenAI API key in the sidebar to get started.")
    
    with st.expander("🔐 How to get an OpenAI API Key", expanded=True):
        st.markdown("""
        ### Steps to get your API key:
        
        1. **Visit OpenAI Platform**
           - Go to [platform.openai.com](https://platform.openai.com)
        
        2. **Sign Up or Log In**
           - Create an account or sign in to existing one
        
        3. **Navigate to API Keys**
           - Click on your profile → "View API keys"
           - Or go directly to [API Keys page](https://platform.openai.com/api-keys)
        
        4. **Create New Key**
           - Click "Create new secret key"
           - Give it a name (e.g., "LBSCEK Chatbot")
           - Copy the key immediately (you won't see it again!)
        
        5. **Add Credits (if needed)**
           - New accounts get free credits
           - Add payment method if credits are exhausted
        
        ### ⚠️ Important:
        - Keep your API key secret
        - Don't share it publicly
        - Each query costs a small amount
        """)
    
    # Show a demo question
    st.markdown("---")
    st.markdown("### 🎯 Example Questions You Can Ask:")
    
    demo_cols = st.columns(2)
    with demo_cols[0]:
        st.markdown("""
        - What courses are offered at LBSCEK?
        - How do I apply for admission?
        - What is the placement record?
        """)
    with demo_cols[1]:
        st.markdown("""
        - Tell me about the faculty
        - What facilities are available?
        - Where is the college located?
        """)

# --------------- MAIN APPLICATION ---------------
def main():
    """Main application entry point"""
    
    # Render sidebar
    render_sidebar()
    
    # Render header
    render_header()
    
    # Check if API key is set
    if not st.session_state.api_key_valid:
        render_api_key_prompt()
        return
    
    # Build vectorstore if needed
    if not st.session_state.vectorstore_built:
        vectorstore = create_vectorstore(st.session_state.api_key)
        
        if vectorstore is None:
            st.error("❌ Failed to build knowledge base. Please check your API key and try again.")
            
            if st.button("🔄 Retry"):
                st.cache_resource.clear()
                st.rerun()
            return
        
        st.session_state.vectorstore = vectorstore
        st.session_state.vectorstore_built = True
        st.rerun()
    
    # Quick questions
    selected_quick_question = render_quick_questions()
    
    st.markdown("---")
    
    # Initialize with welcome message
    if not st.session_state.messages:
        st.session_state.messages.append({
            "role": "assistant",
            "content": WELCOME_MESSAGE,
            "sources": ""
        })
    
    # Render chat messages
    render_chat_messages()
    
    # Chat input
    user_input = selected_quick_question or st.chat_input("Ask me anything about LBSCEK...")
    
    if user_input:
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })
        
        # Display user message
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)
        
        # Generate and display response
        with st.chat_message("assistant", avatar="🎓"):
            with st.spinner("🔍 Searching LBSCEK knowledge base..."):
                result = process_user_question(user_input)
            
            # Display answer
            st.markdown(result["answer"])
            
            # Format and display sources
            sources_formatted = format_source_documents(result["sources"])
            
            if result["sources"]:
                with st.expander("📄 View Sources", expanded=False):
                    st.markdown(sources_formatted)
            
            # Save to session state
            st.session_state.messages.append({
                "role": "assistant",
                "content": result["answer"],
                "sources": sources_formatted
            })
            
            # Update stats
            st.session_state.total_questions += 1
            
            if not result["success"]:
                st.session_state.error_count += 1
                
                if "api" in result["answer"].lower() or "key" in result["answer"].lower():
                    st.info("💡 **Tip:** Make sure your OpenAI API key is valid and has credits.")
        
        # Rerun to update UI
        st.rerun()

# --------------- RUN APPLICATION ---------------
if __name__ == "__main__":
    main()
