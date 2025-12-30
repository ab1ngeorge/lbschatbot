# app.py - LBSCEK RAG Chatbot (Fixed)
import streamlit as st
import os
import logging
from datetime import datetime
from typing import List
import json

# --------------- PAGE CONFIG (MUST BE FIRST!) ---------------
st.set_page_config(
    page_title="LBSCEK AI Assistant",
    page_icon="🎓",
    layout="wide"
)

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------------- DEPENDENCY CHECK ---------------
REQUIRED_PACKAGES = """
streamlit>=1.31.0
langchain>=0.2.0
langchain-community>=0.2.0
langchain-openai>=0.1.0
langchain-text-splitters>=0.2.0
openai>=1.12.0
faiss-cpu>=1.9.0
tiktoken>=0.6.0
beautifulsoup4>=4.12.0
lxml>=5.0.0
requests>=2.31.0
aiohttp>=3.9.0
"""

def check_dependencies():
    """Check all required dependencies"""
    missing = []
    
    try:
        import langchain_community
    except ImportError:
        missing.append("langchain-community")
    
    try:
        import langchain_openai
    except ImportError:
        missing.append("langchain-openai")
    
    try:
        import langchain_text_splitters
    except ImportError:
        missing.append("langchain-text-splitters")
    
    try:
        import faiss
    except ImportError:
        missing.append("faiss-cpu")
    
    try:
        import bs4
    except ImportError:
        missing.append("beautifulsoup4")
    
    try:
        import tiktoken
    except ImportError:
        missing.append("tiktoken")
    
    return missing

missing_packages = check_dependencies()

if missing_packages:
    st.error(f"❌ Missing packages: {', '.join(missing_packages)}")
    st.markdown("### Fix: Update your `requirements.txt`:")
    st.code(REQUIRED_PACKAGES, language="text")
    st.info("After updating, go to **Manage app** → **Reboot app**")
    st.stop()

# --------------- IMPORTS ---------------
try:
    from langchain_community.document_loaders import WebBaseLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    from langchain.chains import ConversationalRetrievalChain
    from langchain.memory import ConversationBufferWindowMemory
    from langchain.schema import Document
except ImportError as e:
    st.error(f"❌ Import Error: {e}")
    st.code(REQUIRED_PACKAGES, language="text")
    st.stop()

# --------------- CSS ---------------
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a365d 0%, #2563eb 100%);
        padding: 2rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .main-header h1 { margin: 0; font-size: 2rem; }
    .main-header p { margin: 0.5rem 0 0 0; opacity: 0.9; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stButton > button {
        border-radius: 0.5rem;
    }
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
]

QUICK_QUESTIONS = [
    "📚 What courses are offered?",
    "🎯 How to apply for admission?",
    "💼 Tell me about placements",
    "👨‍🏫 Faculty information",
    "🏛️ What facilities available?",
    "📍 College location?",
]

WELCOME_MESSAGE = """👋 **Welcome to LBSCEK AI Assistant!**

I can help you with:
- 📚 Courses & Programs
- 🎯 Admissions
- 💼 Placements
- 👨‍🏫 Faculty
- 🏛️ Facilities
- 📍 Contact Info

Ask me anything about LBSCEK!"""

# --------------- SESSION STATE ---------------
def init_session_state():
    defaults = {
        "messages": [],
        "api_key": "",
        "api_key_valid": False,
        "vectorstore": None,
        "vectorstore_built": False,
        "qa_chain": None,
        "total_questions": 0,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session_state()

# --------------- HELPER FUNCTIONS ---------------
def is_valid_api_key(key: str) -> bool:
    """Validate API key format"""
    if not key:
        return False
    key = key.strip()
    # OpenAI keys start with sk- and are usually 51+ chars
    # Also allow sk-proj- format
    return (key.startswith("sk-") and len(key) > 30)

def load_documents_safe() -> List[Document]:
    """Load LBSCEK website pages with error handling"""
    docs = []
    failed = []
    
    progress = st.progress(0, text="Loading pages...")
    
    for i, url in enumerate(LBSCEK_URLS):
        try:
            loader = WebBaseLoader(
                web_paths=[url],
                bs_kwargs={"features": "lxml"}
            )
            loaded = loader.load()
            for doc in loaded:
                doc.metadata["source"] = url
            docs.extend(loaded)
            logger.info(f"✓ Loaded: {url}")
        except Exception as e:
            logger.warning(f"✗ Failed {url}: {e}")
            failed.append(url)
        
        progress.progress((i + 1) / len(LBSCEK_URLS), text=f"Loading: {url}")
    
    progress.empty()
    
    if failed:
        st.warning(f"⚠️ Could not load {len(failed)} pages: {', '.join(failed)}")
    
    if not docs:
        st.error("❌ Failed to load any pages!")
    
    return docs

@st.cache_resource(show_spinner=False)
def build_vectorstore(_api_key: str):
    """Build FAISS vectorstore from LBSCEK pages"""
    try:
        os.environ["OPENAI_API_KEY"] = _api_key
        
        with st.status("🔄 Building Knowledge Base...", expanded=True) as status:
            # Load documents
            st.write("📥 Loading LBSCEK website pages...")
            docs = load_documents_safe()
            
            if not docs:
                return None
            
            st.write(f"✅ Loaded {len(docs)} pages")
            
            # Split documents
            st.write("✂️ Splitting into chunks...")
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            chunks = splitter.split_documents(docs)
            st.write(f"✅ Created {len(chunks)} chunks")
            
            # Create embeddings
            st.write("🧠 Creating embeddings (this may take a minute)...")
            embeddings = OpenAIEmbeddings(
                openai_api_key=_api_key,
                model="text-embedding-3-small"
            )
            vectorstore = FAISS.from_documents(chunks, embeddings)
            
            status.update(label="✅ Knowledge Base Ready!", state="complete", expanded=False)
        
        return vectorstore
        
    except Exception as e:
        logger.error(f"Vectorstore error: {e}")
        st.error(f"❌ Error building knowledge base: {str(e)}")
        
        if "api" in str(e).lower() or "key" in str(e).lower() or "auth" in str(e).lower():
            st.warning("💡 Check your OpenAI API key - it may be invalid or have no credits")
        
        return None

def create_qa_chain(vectorstore, api_key: str):
    """Create conversational QA chain"""
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.1,
        openai_api_key=api_key,
        max_tokens=1000
    )
    
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
        k=5
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
        verbose=False
    )
    
    return qa_chain

def format_sources(sources) -> str:
    """Format source documents"""
    if not sources:
        return ""
    
    seen = set()
    result = []
    
    for doc in sources:
        url = doc.metadata.get("source", "Unknown")
        if url not in seen:
            seen.add(url)
            preview = doc.page_content[:150].replace("\n", " ").strip()
            result.append(f"🔗 **{url}**\n> {preview}...")
    
    return "\n\n".join(result)

def process_question(question: str) -> dict:
    """Process user question"""
    try:
        # Create QA chain if needed
        if st.session_state.qa_chain is None:
            st.session_state.qa_chain = create_qa_chain(
                st.session_state.vectorstore,
                st.session_state.api_key
            )
        
        # Get answer
        result = st.session_state.qa_chain.invoke({"question": question})
        
        return {
            "success": True,
            "answer": result.get("answer", "Sorry, I couldn't find an answer."),
            "sources": result.get("source_documents", [])
        }
        
    except Exception as e:
        logger.error(f"QA error: {e}")
        return {
            "success": False,
            "answer": f"❌ Error: {str(e)}",
            "sources": []
        }

# --------------- SIDEBAR ---------------
with st.sidebar:
    st.header("🎓 LBSCEK Chatbot")
    st.markdown("---")
    
    # API Key Section
    st.subheader("🔑 API Key")
    api_input = st.text_input(
        "OpenAI API Key",
        type="password",
        value=st.session_state.api_key,
        placeholder="sk-...",
        help="Get from https://platform.openai.com/api-keys"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("✅ Connect", use_container_width=True, type="primary"):
            if is_valid_api_key(api_input):
                st.session_state.api_key = api_input.strip()
                st.session_state.api_key_valid = True
                os.environ["OPENAI_API_KEY"] = st.session_state.api_key
                st.success("Connected!")
                st.rerun()
            else:
                st.error("Invalid key format!")
    
    with col2:
        if st.button("🔄 Reset All", use_container_width=True):
            # Clear everything
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.cache_resource.clear()
            st.rerun()
    
    # Status
    if st.session_state.api_key_valid:
        st.success("✅ API Connected")
    else:
        st.warning("⚠️ Enter API Key")
    
    st.markdown("---")
    
    # Stats
    st.subheader("📊 Stats")
    col1, col2 = st.columns(2)
    col1.metric("Questions", st.session_state.total_questions)
    col2.metric("Messages", len(st.session_state.messages))
    
    st.markdown("---")
    
    # Actions
    st.subheader("🛠️ Actions")
    
    if st.button("🗑️ Clear Chat", use_container_width=True):
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
    
    # Export
    if st.session_state.messages:
        export_data = {
            "exported": datetime.now().isoformat(),
            "messages": st.session_state.messages
        }
        st.download_button(
            "📥 Export Chat",
            json.dumps(export_data, indent=2),
            f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )
    
    st.markdown("---")
    
    # Help
    with st.expander("ℹ️ Help"):
        st.markdown("""
        **How to use:**
        1. Get API key from [OpenAI](https://platform.openai.com/api-keys)
        2. Enter key and click **Connect**
        3. Wait for knowledge base to build
        4. Start asking questions!
        
        **Tips:**
        - Be specific in your questions
        - Check sources for details
        - Use quick questions for common topics
        """)

# --------------- MAIN CONTENT ---------------
# Header
st.markdown("""
<div class="main-header">
    <h1>🎓 LBS College of Engineering Kasaragod</h1>
    <p>AI Assistant • Powered by Official Website</p>
</div>
""", unsafe_allow_html=True)

# Check API key
if not st.session_state.api_key_valid:
    st.info("👋 Enter your OpenAI API key in the sidebar to start!")
    
    with st.expander("🔐 How to get an API key", expanded=True):
        st.markdown("""
        ### Steps:
        1. Go to [platform.openai.com](https://platform.openai.com)
        2. Sign up or log in
        3. Navigate to **API Keys** section
        4. Click **Create new secret key**
        5. Copy the key and paste it in the sidebar
        
        ⚠️ **Note:** You need credits in your OpenAI account
        """)
    
    st.stop()

# Build vectorstore if needed
if not st.session_state.vectorstore_built:
    vs = build_vectorstore(st.session_state.api_key)
    
    if vs is None:
        st.error("❌ Failed to build knowledge base!")
        if st.button("🔄 Retry"):
            st.cache_resource.clear()
            st.rerun()
        st.stop()
    
    st.session_state.vectorstore = vs
    st.session_state.vectorstore_built = True
    st.rerun()

# Quick Questions
st.markdown("**💡 Quick Questions:**")
cols = st.columns(3)
selected_quick_q = None

for i, q in enumerate(QUICK_QUESTIONS):
    with cols[i % 3]:
        if st.button(q, key=f"quick_{i}", use_container_width=True):
            selected_quick_q = q.split(" ", 1)[1]  # Remove emoji

st.markdown("---")

# Initialize welcome message
if not st.session_state.messages:
    st.session_state.messages.append({
        "role": "assistant",
        "content": WELCOME_MESSAGE,
        "sources": ""
    })

# Display chat messages
for msg in st.session_state.messages:
    avatar = "🎓" if msg["role"] == "assistant" else "👤"
    
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        
        if msg.get("sources"):
            with st.expander("📄 Sources"):
                st.markdown(msg["sources"])

# Chat input
user_input = selected_quick_q or st.chat_input("Ask about LBSCEK...")

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
        with st.spinner("🔍 Searching LBSCEK knowledge base..."):
            result = process_question(user_input)
        
        # Display answer
        st.markdown(result["answer"])
        
        # Display sources
        sources_text = format_sources(result["sources"])
        if sources_text:
            with st.expander("📄 Sources"):
                st.markdown(sources_text)
        
        # Save to session
        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "sources": sources_text
        })
        
        if result["success"]:
            st.session_state.total_questions += 1
        else:
            if "api" in result["answer"].lower() or "key" in result["answer"].lower():
                st.info("💡 Check your OpenAI API key and credits")
    
    st.rerun()
