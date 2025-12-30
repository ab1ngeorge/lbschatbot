# app.py
import streamlit as st
import os
import requests
from bs4 import BeautifulSoup

from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

# --------------- CONFIG ---------------
LBS_BASE_URL = "https://lbscek.ac.in/"
# Set your key as environment variable or directly here (not recommended in production)
os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

# --------------- HELPER: SCRAPE ALL MAIN LINKS (OPTIONAL SIMPLE CRAWL) ---------------
def get_lbs_links(base_url=LBS_BASE_URL, max_pages=10):
    """
    Very simple crawler: collects internal links from the home page only.
    For a bigger project, expand this to follow links recursively with checks.
    """
    urls = set([base_url])
    try:
        resp = requests.get(base_url, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href.startswith("http") and "lbscek.ac.in" in href:
                urls.add(href)
            elif href.startswith("/"):
                urls.add(base_url.rstrip("/") + href)
    except Exception:
        pass
    # Limit pages for speed
    return list(urls)[:max_pages]

@st.cache_resource(show_spinner=True)
def build_vectorstore():
    # 1. Load website pages as documents
    urls = get_lbs_links()
    loader = WebBaseLoader(urls)
    docs = loader.load()

    # 2. Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""],
    )
    split_docs = splitter.split_documents(docs)

    # 3. Create embeddings + FAISS index
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    return vectorstore

@st.cache_resource(show_spinner=True)
def get_qa_chain():
    vectorstore = build_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
    )
    return qa_chain

# --------------- STREAMLIT UI ---------------
st.set_page_config(page_title="LBSCEK Website Chatbot", page_icon="🎓")
st.title("LBSCEK Website Chatbot")
st.write("Ask any question about **LBS College of Engineering, Kasaragod**. "
         "Answers are generated using information from `https://lbscek.ac.in/` only.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
user_input = st.chat_input("Ask something about LBSCEK...")
if user_input:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get answer from RAG chain
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            qa_chain = get_qa_chain()
            result = qa_chain({"query": user_input})
            answer = result["result"]
            sources = result.get("source_documents", [])

        st.markdown(answer)

        # Optional: show sources in expander
        with st.expander("Sources (website chunks used)"):
            for i, doc in enumerate(sources):
                st.markdown(f"**Source {i+1}:** {doc.metadata.get('source', '')}")
                st.write(doc.page_content[:500] + "...")

    st.session_state.messages.append({"role": "assistant", "content": answer})
