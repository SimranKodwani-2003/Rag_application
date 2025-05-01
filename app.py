import streamlit as st
import os
from io import BytesIO
from docx import Document
from PyPDF2 import PdfReader
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
from dotenv import load_dotenv

# Load API key from environment
load_dotenv()
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

def process_input(input_type, input_data):
    if input_type == "Link":
        loader = WebBaseLoader(input_data)
        documents = loader.load()
        raw_text = "\n".join(doc.page_content for doc in documents)

    elif input_type in ["PDF", "TXT", "DOCX"]:
        if input_type == "PDF":
            pdf_reader = PdfReader(input_data)
            raw_text = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
        elif input_type == "TXT":
            raw_text = input_data.read().decode('utf-8')
        elif input_type == "DOCX":
            doc = Document(input_data)
            raw_text = "\n".join([para.text for para in doc.paragraphs])
    elif input_type == "Text":
        raw_text = input_data
    else:
        raise ValueError("Unsupported input type")

    # Split text into chunks
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_text(raw_text)

    # Generate embeddings and store in vector DB
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vectorstore = FAISS.from_texts(chunks, embeddings)

    return vectorstore

def answer_question(vectorstore, query):
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-large",  # Good RAG-compatible model
        model_kwargs={"temperature": 0.5, "max_length": 512},
        huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN
    )

    qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())
    result = qa.invoke({"query": query})
    return result.get("result", "No result returned.")

def main():
    st.title("📄 RAG Q&A App with HuggingFace + LangChain")

    input_type = st.selectbox("Choose Input Type", ["Link", "PDF", "Text", "DOCX", "TXT"])

    if input_type == "Link":
        input_data = st.text_input("Enter URL")
    elif input_type == "Text":
        input_data = st.text_area("Enter your text here")
    else:
        file_types = {"PDF": "pdf", "TXT": "txt", "DOCX": ["docx", "doc"]}
        input_data = st.file_uploader("Upload a file", type=file_types[input_type])

    if st.button("Process Document"):
        if input_data:
            vectorstore = process_input(input_type, input_data)
            st.session_state.vectorstore = vectorstore
            st.success("✅ Document processed and embedded!")
        else:
            st.warning("⚠️ Please provide valid input.")

    if "vectorstore" in st.session_state:
        query = st.text_input("Ask your question:")
        if st.button("Get Answer"):
            answer = answer_question(st.session_state.vectorstore, query)
            st.write("### 🧠 Answer:")
            st.write(answer)

if __name__ == "__main__":
    main()
