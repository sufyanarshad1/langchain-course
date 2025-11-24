import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS


load_dotenv()


def main():
    print("Hello from langchain-course!")


if __name__ == "__main__":
    print("Loading PDF document...")
    pdf_path = "/home/sufyan/Sufyan/Office/LangChain/langchain-course/AI engineer"
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
    docs = text_splitter.split_documents(documents=documents)

    embeddings = OllamaEmbeddings(model="mxbai-embed-large:latest")
    vectorstore = FAISS.from_documents(docs,embeddings)
    vectorstore.save_local("faiss_index_ai_engineer")
