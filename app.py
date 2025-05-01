import streamlit as st
import os
from io import BytesIO
from docx import Document
from PyPDF2 import PdfReader

from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.in_memory import InMemoryDocstore
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub  # ✅ use this instead

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

    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_text(raw_text)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore

def answer_question(vectorstore, query):
    # ✅ Use HuggingFaceHub, not HuggingFaceEndpoint
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-large",  # ✅ Use a model that's fully compatible with langchain
        model_kwargs={"temperature": 0.5, "max_length": 512},
        huggingfacehub_api_token=os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    )

    qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())
    return qa.invoke({"query": query})

def main():
    st.title("RAG Q&A App")
    input_type = st.selectbox("Input Type", ["Link", "PDF", "Text", "DOCX", "TXT"])

    if input_type == "Link":
        url = st.text_input("Enter URL")
        input_data = url
    elif input_type == "Text":
        input_data = st.text_area("Enter your text")
    else:
        input_data = st.file_uploader("Upload your file", type={"PDF": "pdf", "TXT": "txt", "DOCX": ["docx", "doc"]}[input_type])

    if st.button("Process"):
        if input_data:
            vectorstore = process_input(input_type, input_data)
            st.session_state.vectorstore = vectorstore
            st.success("Document processed successfully!")
        else:
            st.warning("Please provide input data.")

    if "vectorstore" in st.session_state:
        query = st.text_input("Ask your question")
        if st.button("Submit"):
            answer = answer_question(st.session_state.vectorstore, query)
            st.write(answer["result"])

if __name__ == "__main__":
    main()
