import streamlit as st
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms.ollama import Ollama

# Global QA chain
qa_chain = None

# PDF Loader using LLaMA 3.2
def load_pdf(file):
    text = ""
    with fitz.open(stream=file.read(), filetype="pdf") as pdf:
        for page in pdf:
            text += page.get_text()

    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    docs = [Document(page_content=chunk) for chunk in chunks]

    # Use LLaMA 3.2 via Ollama
    embeddings = OllamaEmbeddings(model="llama3")
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever()

    return RetrievalQA.from_chain_type(
        llm=Ollama(model="llama3"),
        retriever=retriever,
        return_source_documents=False
    )

# Streamlit UI
st.set_page_config(page_title="📘 AI Tutor Bot", layout="centered")
st.title("📘 AI Tutor Chatbot (LLaMA 3.2 + Streamlit)")
st.write("Upload a PDF and ask any question related to it.")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file:
    st.success("✅ PDF uploaded. Processing with LLaMA 3.2...")
    qa_chain = load_pdf(uploaded_file)
    st.success("✅ Done! You can now ask questions below.")

    user_input = st.text_input("Ask a question from your PDF:")
    if user_input and qa_chain:
        with st.spinner("Thinking..."):
            answer = qa_chain.run(user_input)
            st.markdown(f"**Answer:** {answer}")
