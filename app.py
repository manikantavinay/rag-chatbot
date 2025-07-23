
import os
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from youtubesearchpython import VideosSearch

st.markdown(
    """
    <style>
    body, .stApp { background-color: #f7f7fa; }
    .stSidebar { background-color: #22223b !important; color: #f7f7fa !important; }
    .stTitle, .stHeader, .stSubheader { color: #4a4e69; }
    .stAlert { background-color: #c9ada7 !important; color: #22223b !important; }
    .st-bb { background: #fff; border-radius: 10px; box-shadow: 0 2px 8px #c9ada733; padding: 1em; }
    </style>
    """,
    unsafe_allow_html=True
)

st.set_page_config(page_title="vinay chatbot", layout="wide")
st.title("vinay chatbot")
st.sidebar.title("About")
st.sidebar.info("""
vinay chatbot

- Upload a PDF or TXT file
- Ask questions about your document
- Get YouTube recommendations
""")


uploaded = st.file_uploader("Upload a document", type=["pdf", "txt"])

# File Info
if uploaded is not None:
    st.sidebar.write("📎 File Info")
    st.sidebar.json({
        "Filename": uploaded.name,
        "Type": uploaded.type,
        "Size (KB)": f"{uploaded.size / 1024:.2f}"
    })

    # Extract text
    try:
        if uploaded.type == "application/pdf":
            text = "".join(p.extract_text() or "" for p in PdfReader(uploaded).pages)
        else:
            text = uploaded.read().decode("utf-8")
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        st.stop()

    # Display raw text (optional)
    with st.expander("📄 Raw Text Preview"):
        st.write(text[:3000] + "...")  # Limit for preview

    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_text(text)

    # Embed & store in FAISS
    with st.spinner("🔍 Generating embeddings..."):
        embeddings = HuggingFaceEmbeddings()
        vector_store = FAISS.from_texts(chunks, embedding=embeddings)

    # Load LLM (use a public, small model for local demo)
    with st.spinner("🚀 Loading language model..."):
        model_name = "distilgpt2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=100)
        llm = HuggingFacePipeline(pipeline=pipe)

    # Build QA chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vector_store.as_retriever(),
        return_source_documents=True
    )

    # Chat interface
    st.subheader("💬 Ask a question about the document")
    user_query = st.text_input("Your question")

    if user_query:
        with st.spinner("Generating answer..."):
            result = qa(user_query)
            st.success(result["result"])

        # Show related YouTube videos (list 3)
        with st.spinner("🔎 Searching YouTube for related videos..."):
            videos = VideosSearch(user_query, limit=3).result().get('result', [])
        if videos:
            st.markdown("**YouTube Recommendations:**")
            for vid in videos:
                st.markdown(f"**{vid['title']}**")
                st.video(vid['link'])
        else:
            st.info("No related YouTube videos found.")

        with st.expander("📚 Source Chunks"):
            for doc in result["source_documents"]:
                st.markdown(f"- {doc.page_content[:200]}...")
else:
    st.info("👈 Upload a `.pdf` or `.txt` file to get started.")
