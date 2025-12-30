# app.py - LBSCEK RAG Chatbot (Fixed Version)
import streamlit as st
import os
import logging
from datetime import datetime
from typing import List, Optional, Tuple
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------------- PAGE CONFIG (Must be first!) ---------------
st.set_page_config(
    page_title="LBSCEK AI Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------- IMPORTS WITH ERROR HANDLING ---------------
try:
    from langchain_community.document_loaders import WebBaseLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    from langchain.chains import ConversationalRetrievalChain
    from langchain.memory import ConversationBufferWindowMemory
    IMPORTS_OK = True
except ImportError as e:
    IMPORTS_OK = False
    IMPORT_ERROR = str(e)

# --------------- CUSTOM CSS ---------------
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        max-width: 1200px;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2rem;
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    
    .stButton button {
        border-radius: 0.5rem;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --------------- CHECK IMPORTS ---------------
if not IMPORTS_OK:
    st.error(f"❌ Import Error: {IMPORT_ERROR}")
    st.markdown("""
    ### 🔧 Fix Instructions:
    
    Make sure your `requirements.txt` contains:
    ```
    streamlit>=1.28.0
    langchain>=0.2.0
    langchain-community>=0.2.0
    langchain-openai>=0.1.0
    langchain-text-splitters>=0.2.0
    faiss-cpu>=1.7.4
    openai>=1.0.0
    tiktoken>=0.5.0
    beautifulsoup4>=4.12.0
    lxml>=4.9.0
    requests>=2.31.0
    ```
    
    Then redeploy your app.
    """)
    st.stop()

# --------------- CONSTANTS ---------------
QUICK_QUESTIONS = [
    "📚 What courses are offered?",
    "🎯 How to apply for admission?",
    "💼 What are placement statistics?",
    "👨‍🏫 Tell me about faculty",
    "🏛️ What facilities are available?",
    "📍 Where is the college located?",
]

# --------------- LBSCEK PAGES ---------------
def get_lbs_links() -> List[str]:
    """List of LBSCEK pages to scrape"""
    return [
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
    ]

# --------------- SESSION STATE ---------------
def init_session_state():
    """Initialize session state variables"""
    defaults = {
        "messages": [],
        "api_key_set": False,
        "api_key": "",
        "vectorstore": None,
        "vectorstore_loaded": False,
        "total_questions": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# --------------- HELPER FUNCTIONS ---------------
def validate_api_key(api_key: str) -> bool:
    """Validate OpenAI API key format"""
    return api_key and api_key.startswith("sk-") and len(api_key) > 20

def load_documents(urls: List[str]) -> List:
    """Load documents from URLs with progress"""
    all_docs = []
    failed = []
    
    progress = st.progress(0)
    status = st.empty()
    
    for i, url in enumerate(urls):
        try:
            status.text(f"📥 Loading: {url}")
            loader = WebBaseLoader(
                web_paths=[url],
                bs_kwargs={"parse_only": None}
            )
            docs = loader.load()
            all_docs.extend(docs)
        except Exception as e:
            logger.warning(f"Failed: {url} - {e}")
            failed.append(url)
        progress.progress((i + 1) / len(urls))
    
    progress.empty()
    status.empty()
    
    if failed:
        st.warning(f"⚠️ Could not load {len(failed)} pages")
    
    return all_docs

@st.cache_resource(show_spinner=False)
def build_vectorstore(api_key: str) -> Optional[FAISS]:
    """Build FAISS vectorstore from LBSCEK pages"""
    try:
        os.environ["OPENAI_API_KEY"] = api_key
        
        with st.status("🔄 Building knowledge base...", expanded=True) as status:
            # Load documents
            st.write("📥 Loading LBSCEK website...")
            urls = get_lbs_links()
            docs = load_documents(urls)
            
            if not docs:
                st.error("❌ No documents loaded!")
                return None
            
            st.write(f"✅ Loaded {len(docs)} pages")
            
            # Split documents
            st.write("✂️ Processing text...")
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
            )
            chunks = splitter.split_documents(docs)
            st.write(f"✅ Created {len(chunks)} chunks")
            
            # Create embeddings
            st.write("🧠 Creating embeddings...")
            embeddings = OpenAIEmbeddings()
            vectorstore = FAISS.from_documents(chunks, embeddings)
            
            status.update(label="✅ Knowledge base ready!", state="complete")
        
        return vectorstore
        
    except Exception as e:
        logger.error(f"Vectorstore error: {e}")
        st.error(f"❌ Error: {str(e)}")
        return None

def get_qa_chain(vectorstore: FAISS):
    """Create QA chain"""
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.1,
    )
    
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
        k=5
    )
    
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )
    
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
    )
    
    return qa_chain

def format_sources(sources: List) -> str:
    """Format sources for display"""
    seen = set()
    formatted = []
    for doc in sources:
        url = doc.metadata.get("source", "Unknown")
        if url not in seen:
            seen.add(url)
            preview = doc.page_content[:150] + "..."
            formatted.append(f"🔗 **{url}**\n> {preview}")
    return "\n\n".join(formatted) if formatted else "No sources found"

# --------------- SIDEBAR ---------------
def render_sidebar():
    """Render sidebar"""
    with st.sidebar:
        st.header("🎓 LBSCEK Chatbot")
        st.markdown("---")
        
        # API Key
        st.subheader("🔑 API Key")
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=st.session_state.api_key,
            placeholder="sk-...",
            help="Get from https://platform.openai.com/api-keys"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Set Key", use_container_width=True):
                if validate_api_key(api_key):
                    st.session_state.api_key = api_key
                    st.session_state.api_key_set = True
                    os.environ["OPENAI_API_KEY"] = api_key
                    st.success("Key set!")
                    st.rerun()
                else:
                    st.error("Invalid key!")
        
        with col2:
            if st.button("🔄 Reset", use_container_width=True):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.cache_resource.clear()
                st.rerun()
        
        if st.session_state.api_key_set:
            st.success("✅ API Key configured")
        
        st.markdown("---")
        
        # Stats
        st.subheader("📊 Stats")
        col1, col2 = st.columns(2)
        col1.metric("Questions", st.session_state.total_questions)
        col2.metric("Messages", len(st.session_state.messages))
        
        st.markdown("---")
        
        # Clear chat
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.total_questions = 0
            st.rerun()
        
        # Rebuild
        if st.button("🔄 Rebuild Knowledge Base", use_container_width=True):
            st.cache_resource.clear()
            st.session_state.vectorstore_loaded = False
            st.rerun()
        
        st.markdown("---")
        
        # Export
        if st.session_state.messages:
            export = json.dumps({
                "exported": datetime.now().isoformat(),
                "messages": st.session_state.messages
            }, indent=2)
            st.download_button(
                "📥 Export Chat",
                export,
                f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                use_container_width=True
            )
        
        st.markdown("---")
        
        # Help
        with st.expander("ℹ️ Help"):
            st.markdown("""
            **How to use:**
            1. Enter OpenAI API key
            2. Click "Set Key"
            3. Wait for knowledge base
            4. Ask questions!
            
            **Topics:**
            - Admissions
            - Departments
            - Placements
            - Faculty
            - Facilities
            """)

# --------------- MAIN ---------------
def main():
    """Main app"""
    render_sidebar()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎓 LBS College of Engineering Kasaragod</h1>
        <p>AI Assistant • Powered by Official Website Data</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check API key
    if not st.session_state.api_key_set:
        st.info("👋 Welcome! Enter your OpenAI API key in the sidebar to start.")
        
        with st.expander("🔐 How to get an API key"):
            st.markdown("""
            1. Go to [OpenAI Platform](https://platform.openai.com)
            2. Sign up or log in
            3. Go to API Keys
            4. Create new key
            5. Copy and paste in sidebar
            """)
        st.stop()
    
    # Build vectorstore
    if not st.session_state.vectorstore_loaded:
        vectorstore = build_vectorstore(st.session_state.api_key)
        if vectorstore:
            st.session_state.vectorstore = vectorstore
            st.session_state.vectorstore_loaded = True
        else:
            st.error("Failed to build knowledge base. Check API key.")
            st.stop()
    
    # Quick questions
    st.markdown("**💡 Quick Questions:**")
    cols = st.columns(3)
    quick_q = None
    for i, q in enumerate(QUICK_QUESTIONS):
        with cols[i % 3]:
            if st.button(q, key=f"q_{i}", use_container_width=True):
                quick_q = q.split(" ", 1)[1]
    
    st.markdown("---")
    
    # Welcome message
    if not st.session_state.messages:
        st.session_state.messages.append({
            "role": "assistant",
            "content": """👋 **Welcome to LBSCEK AI Assistant!**

Ask me about:
- 📚 Academic programs & courses
- 🎯 Admission process
- 💼 Placements & companies
- 👨‍🏫 Faculty information
- 🏛️ Facilities & infrastructure
- 📍 Contact details

How can I help you today?"""
        })
    
    # Display messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="🎓" if msg["role"] == "assistant" else "👤"):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📄 Sources"):
                    st.markdown(msg["sources"])
    
    # Chat input
    user_input = quick_q or st.chat_input("Ask about LBSCEK...")
    
    if user_input:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)
        
        # Generate response
        with st.chat_message("assistant", avatar="🎓"):
            try:
                with st.spinner("🔍 Searching..."):
                    qa_chain = get_qa_chain(st.session_state.vectorstore)
                    result = qa_chain({"question": user_input})
                    answer = result.get("answer", "Sorry, I couldn't find an answer.")
                    sources = format_sources(result.get("source_documents", []))
                
                st.markdown(answer)
                
                with st.expander("📄 Sources"):
                    st.markdown(sources)
                
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
                    st.info("💡 Check your API key and credits")
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": error_msg
                })
        
        st.rerun()

if __name__ == "__main__":
    main()
