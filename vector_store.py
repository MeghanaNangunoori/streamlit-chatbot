import os
from dotenv import load_dotenv
from groq import Groq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="my_documents")

pdf_path = input("Enter PDF path: ")
loader = PyPDFLoader(pdf_path)
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)

texts = [chunk.page_content for chunk in chunks]
ids = [str(i) for i in range(len(chunks))]

collection.add(documents=texts, ids=ids)
print(f"Stored {len(chunks)} chunks in ChromaDB.")

while True:
    question = input("\nYour question: ")
    
    results = collection.query(query_texts=[question], n_results=1)
    relevant_chunk = results["documents"][0][0]
    
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": f"Answer based on this context only:\n\n{relevant_chunk}"},
            {"role": "user", "content": question}
        ]
    )
    print(response.choices[0].message.content)