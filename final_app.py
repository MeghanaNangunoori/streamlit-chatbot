import streamlit as st
import os
from dotenv import load_dotenv
from groq import Groq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
import tempfile

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
chroma_client = chromadb.Client()

st.title("Document Q&A with AI")

if "collection" not in st.session_state:
    st.session_state.collection = None

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file and st.session_state.collection is None:
    with st.spinner("Processing PDF..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(documents)

        collection = chroma_client.create_collection(name="pdf_docs")
        texts = [chunk.page_content for chunk in chunks]
        ids = [str(i) for i in range(len(chunks))]
        collection.add(documents=texts, ids=ids)

        st.session_state.collection = collection
        os.unlink(tmp_path)

    st.success(f"PDF processed! {len(chunks)} chunks stored.")

if st.session_state.collection:
    question = st.text_input("Ask a question about your PDF:")

    if question:
        with st.spinner("Searching..."):
            results = st.session_state.collection.query(
                query_texts=[question], n_results=1
            )
            relevant_chunk = results["documents"][0][0]

            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": f"Answer based on this context only:\n\n{relevant_chunk}"},
                    {"role": "user", "content": question}
                ]
            )
            st.write(response.choices[0].message.content)