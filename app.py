# app.py - LBSCEK RAG Chatbot (NO BeautifulSoup needed)
import streamlit as st
import os
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

# --------------- CONFIG ---------------
LBS_BASE_URL = "https://lbscek.ac.in/"
os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY_HERE"  # Replace with your key

# --------------- HARDCODED LBSCEK PAGES (NO SCRAPING) ---------------
@st.cache_data
def get_lbs_links():
    """Hardcoded list of important LBSCEK pages"""
    return [
        "https://lbscek.ac.in/",
        "https://lbscek.ac.in/departments/",
        "https://lbscek.ac.in/faculty/",
        "https://lbscek.ac.in/about-us/",
        "https://lbscek.ac.in/admission/",
        "https://lbscek.ac.in/placements/",
        "https://lbscek.ac.in/facilities/",
        "https://lbscek.ac.in/contact-us/",
    ]

@st.cache_resource(show_spinner="Loading LBSCEK website...")
def build_vectorstore():
    """Load LBSCEK pages, split, and create FAISS index"""
    urls = get_lbs_links()
    st.info(f"Loading {len(urls)} pages from lbscek.ac.in...")
    
    loader = WebBaseLoader(urls)
    docs = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    split_docs = splitter.split_documents(docs)
    
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    return vectorstore

@st.cache_resource(show_spinner="Setting up chatbot...")
def get_qa_chain():
    """Create the RAG QA chain"""
    vectorstore = build_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
    )
    return qa_chain

# --------------- STREAMLIT UI ---------------
st.set_page_config(page_title="LBSCEK Chatbot", page_icon="🎓", layout="wide")
st.title("🎓 LBS College of Engineering Chatbot")
st.markdown("**Ask anything about LBSCEK Kasaragod** - answers from official website only")

# Sidebar for API key (safer than hardcoding)
if "api_key_set" not in st.session_state:
    st.session_state.api_key_set = False

with st.sidebar:
    st.header("🔑 API Key")
    api_key = st.text_input("OpenAI API Key", type="password", 
                           help="Get from https://platform.openai.com/account/api-keys")
    if st.button("Set API Key") and api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        st.session_state.api_key_set = True
        st.success("✅ API Key set!")
        st.rerun()

# Check API key
if not st.session_state.api_key_set:
    st.warning("⚠️ Please set your OpenAI API key in the sidebar first!")
    st.stop()

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! Ask me anything about LBS College of Engineering Kasaragod. I know about departments, faculty, admissions, placements, etc."}
    ]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask about LBSCEK..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Searching LBSCEK website..."):
            try:
                qa_chain = get_qa_chain()
                result = qa_chain({"query": prompt})
                answer = result["result"]
                sources = result.get("source_documents", [])
                
                st.markdown(answer)
                
                # Show sources
                with st.expander(f"📄 Sources ({len(sources)} pages)", expanded=False):
                    for i, doc in enumerate(sources):
                        st.markdown(f"**Page {i+1}:** [{doc.metadata.get('source', 'Unknown')}]")
                        st.caption(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
                        
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.info("💡 Make sure your OpenAI API key is valid and you have credits.")

    # Save assistant response
    st.session_state.messages.append({"role": "assistant", "content": answer})

# Clear chat button
if st.sidebar.button("🗑️ Clear Chat"):
    st.session_state.messages = []
    st.rerun()
