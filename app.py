# app.py - LBSCEK RAG Chatbot (Fixed for LangChain 0.2.x)
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

# --------------- REQUIREMENTS INFO ---------------
REQUIREMENTS_TEXT = """streamlit==1.31.0
langchain==0.2.16
langchain-core==0.2.40
langchain-community==0.2.16
langchain-openai==0.1.25
langchain-text-splitters==0.2.4
openai==1.51.0
faiss-cpu==1.9.0.post1
tiktoken==0.7.0
beautifulsoup4==4.12.3
lxml==5.3.0
requests==2.32.3
aiohttp==3.10.8"""

# --------------- DEPENDENCY CHECK ---------------
def check_and_import():
    """Check dependencies and import modules"""
    errors = []
    
    # Check each package
    packages_to_check = [
        ("langchain_community", "langchain-community"),
        ("langchain_openai", "langchain-openai"),
        ("langchain_text_splitters", "langchain-text-splitters"),
        ("langchain_core", "langchain-core"),
        ("faiss", "faiss-cpu"),
        ("bs4", "beautifulsoup4"),
        ("tiktoken", "tiktoken"),
    ]
    
    for module_name, package_name in packages_to_check:
        try:
            __import__(module_name)
        except ImportError:
            errors.append(package_name)
    
    return errors

missing = check_and_import()

if missing:
    st.error(f"❌ Missing packages: {', '.join(missing)}")
    st.markdown("### Update your `requirements.txt` with:")
    st.code(REQUIREMENTS_TEXT, language="text")
    st.info("Then go to **Manage app** → **Reboot app**")
    st.stop()

# --------------- IMPORTS ---------------
try:
    from langchain_community.document_loaders import WebBaseLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.messages import HumanMessage, AIMessage
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough
except ImportError as e:
    st.error(f"❌ Import Error: {e}")
    st.markdown("### Update your `requirements.txt` with:")
    st.code(REQUIREMENTS_TEXT, language="text")
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
    .stButton > button { border-radius: 0.5rem; }
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

SYSTEM_PROMPT = """You are an AI assistant for LBS College of Engineering Kasaragod (LBSCEK).
Answer questions based ONLY on the provided context from the official LBSCEK website.
Be helpful, accurate, and concise. If you don't find the answer in the context, say so honestly.

Context:
{context}

Chat History:
{chat_history}
"""

# --------------- SESSION STATE ---------------
def init_session():
    defaults = {
        "messages": [],
        "chat_history": [],
        "api_key": "",
        "api_key_valid": False,
        "vectorstore": None,
        "vectorstore_built": False,
        "total_questions": 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()

# --------------- HELPER FUNCTIONS ---------------
def is_valid_api_key(key: str) -> bool:
    if not key:
        return False
    key = key.strip()
    return key.startswith("sk-") and len(key) > 30

def load_documents() -> List:
    """Load LBSCEK website pages"""
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
        except Exception as e:
            logger.warning(f"Failed {url}: {e}")
            failed.append(url)
        
        progress.progress((i + 1) / len(LBSCEK_URLS))
    
    progress.empty()
    
    if failed:
        st.warning(f"⚠️ Could not load {len(failed)} pages")
    
    return docs

@st.cache_resource(show_spinner=False)
def build_vectorstore(_api_key: str):
    """Build FAISS vectorstore"""
    try:
        os.environ["OPENAI_API_KEY"] = _api_key
        
        with st.status("🔄 Building Knowledge Base...", expanded=True) as status:
            st.write("📥 Loading LBSCEK website...")
            docs = load_documents()
            
            if not docs:
                st.error("❌ No documents loaded!")
                return None
            
            st.write(f"✅ Loaded {len(docs)} pages")
            
            st.write("✂️ Splitting text...")
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            chunks = splitter.split_documents(docs)
            st.write(f"✅ Created {len(chunks)} chunks")
            
            st.write("🧠 Creating embeddings...")
            embeddings = OpenAIEmbeddings(
                openai_api_key=_api_key,
                model="text-embedding-3-small"
            )
            vectorstore = FAISS.from_documents(chunks, embeddings)
            
            status.update(label="✅ Knowledge Base Ready!", state="complete", expanded=False)
        
        return vectorstore
        
    except Exception as e:
        logger.error(f"Error: {e}")
        st.error(f"❌ Error: {e}")
        return None

def format_docs(docs):
    """Format documents for context"""
    return "\n\n".join(doc.page_content for doc in docs)

def format_chat_history(history: List) -> str:
    """Format chat history"""
    if not history:
        return "No previous conversation."
    
    formatted = []
    for msg in history[-6:]:  # Last 3 exchanges
        role = "Human" if msg["role"] == "user" else "Assistant"
        formatted.append(f"{role}: {msg['content'][:200]}")
    
    return "\n".join(formatted)

def get_answer(question: str, vectorstore, api_key: str) -> dict:
    """Get answer using RAG"""
    try:
        # Retrieve relevant documents
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        docs = retriever.invoke(question)
        
        # Format context
        context = format_docs(docs)
        chat_history = format_chat_history(st.session_state.chat_history)
        
        # Create prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "{question}")
        ])
        
        # Create LLM
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.1,
            openai_api_key=api_key,
            max_tokens=1000
        )
        
        # Create chain
        chain = prompt | llm | StrOutputParser()
        
        # Get answer
        answer = chain.invoke({
            "context": context,
            "chat_history": chat_history,
            "question": question
        })
        
        return {
            "success": True,
            "answer": answer,
            "sources": docs
        }
        
    except Exception as e:
        logger.error(f"Error: {e}")
        return {
            "success": False,
            "answer": f"❌ Error: {str(e)}",
            "sources": []
        }

def format_sources(sources) -> str:
    """Format sources for display"""
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

# --------------- SIDEBAR ---------------
with st.sidebar:
    st.header("🎓 LBSCEK Chatbot")
    st.markdown("---")
    
    # API Key
    st.subheader("🔑 API Key")
    api_input = st.text_input(
        "OpenAI API Key",
        type="password",
        value=st.session_state.api_key,
        placeholder="sk-...",
        help="Get from platform.openai.com/api-keys"
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
                st.error("Invalid key!")
    
    with col2:
        if st.button("🔄 Reset", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.cache_resource.clear()
            st.rerun()
    
    if st.session_state.api_key_valid:
        st.success("✅ API Connected")
    else:
        st.warning("⚠️ Enter API Key")
    
    st.markdown("---")
    
    # Stats
    st.subheader("📊 Stats")
    c1, c2 = st.columns(2)
    c1.metric("Questions", st.session_state.total_questions)
    c2.metric("Messages", len(st.session_state.messages))
    
    st.markdown("---")
    
    # Actions
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.session_state.total_questions = 0
        st.rerun()
    
    if st.button("🔄 Rebuild KB", use_container_width=True):
        st.session_state.vectorstore = None
        st.session_state.vectorstore_built = False
        st.cache_resource.clear()
        st.rerun()
    
    if st.session_state.messages:
        st.download_button(
            "📥 Export Chat",
            json.dumps({"messages": st.session_state.messages}, indent=2),
            f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            use_container_width=True
        )
    
    st.markdown("---")
    
    with st.expander("ℹ️ Help"):
        st.markdown("""
        **How to use:**
        1. Get API key from [OpenAI](https://platform.openai.com/api-keys)
        2. Enter key and click **Connect**
        3. Wait for knowledge base
        4. Ask questions!
        """)

# --------------- MAIN ---------------
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
        1. Go to [platform.openai.com](https://platform.openai.com)
        2. Sign up or log in
        3. Go to **API Keys** section
        4. Click **Create new secret key**
        5. Copy and paste in sidebar
        
        ⚠️ You need credits in your OpenAI account
        """)
    st.stop()

# Build vectorstore
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
quick_q = None

for i, q in enumerate(QUICK_QUESTIONS):
    with cols[i % 3]:
        if st.button(q, key=f"q{i}", use_container_width=True):
            quick_q = q.split(" ", 1)[1]

st.markdown("---")

# Welcome message
if not st.session_state.messages:
    st.session_state.messages.append({
        "role": "assistant",
        "content": WELCOME_MESSAGE,
        "sources": ""
    })

# Display messages
for msg in st.session_state.messages:
    avatar = "🎓" if msg["role"] == "assistant" else "👤"
    
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        
        if msg.get("sources"):
            with st.expander("📄 Sources"):
                st.markdown(msg["sources"])

# Chat input
user_input = quick_q or st.chat_input("Ask about LBSCEK...")

if user_input:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)
    
    # Get response
    with st.chat_message("assistant", avatar="🎓"):
        with st.spinner("🔍 Searching..."):
            result = get_answer(
                user_input,
                st.session_state.vectorstore,
                st.session_state.api_key
            )
        
        st.markdown(result["answer"])
        
        sources_text = format_sources(result["sources"])
        if sources_text:
            with st.expander("📄 Sources"):
                st.markdown(sources_text)
        
        # Save
        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "sources": sources_text
        })
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": result["answer"]
        })
        
        if result["success"]:
            st.session_state.total_questions += 1
        else:
            if "api" in result["answer"].lower() or "key" in result["answer"].lower():
                st.info("💡 Check your API key and credits")
    
    st.rerun()
