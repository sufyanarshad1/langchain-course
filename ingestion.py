import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()



if __name__ == "__main__":
    print("ingesting...")
    loader = TextLoader("/home/sufyan/Sufyan/Office/LangChain/langchain-course/mediumblog1.txt", encoding="utf8")
    documents = loader.load()

    print("splitting...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    print(f"created {len(texts)} chunks")

    embeddings = OllamaEmbeddings(model="mxbai-embed-large:latest")

    print("ingesting to pinecone...")
    PineconeVectorStore.from_documents(
        texts,
        embeddings,
        index_name=os.getenv("INDEX_NAME"),
    )

    print("ingestion complete.")