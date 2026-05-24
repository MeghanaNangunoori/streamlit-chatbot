from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.1-8b-instant"
)

pdf_path = input("Enter PDF path: ")
loader = PyPDFLoader(pdf_path)
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)
print(f"PDF split into {len(chunks)} chunks.")

def find_relevant_chunk(question, chunks):
    question_words = set(question.lower().split())
    best_chunk = ""
    best_score = 0
    for chunk in chunks:
        chunk_words = set(chunk.page_content.lower().split())
        score = len(question_words & chunk_words)
        if score > best_score:
            best_score = score
            best_chunk = chunk.page_content
    return best_chunk

prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer based on this context only:\n\n{context}"),
    ("human", "{question}")
])

chain = prompt | llm

while True:
    question = input("\nYour question: ")
    relevant_chunk = find_relevant_chunk(question, chunks)
    response = chain.invoke({"context": relevant_chunk, "question": question})
    print(response.content)