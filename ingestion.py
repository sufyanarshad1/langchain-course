from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

load_dotenv()

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/"
]

docs = [WebBaseLoader(url).load() for url in urls]
doc_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=250, chunk_overlap=0
)

doc_splits = text_splitter.split_documents(doc_list)

# vectorStore = Chroma.from_documents(
#     documents=doc_splits,
#     collection_name="rag-chroma",
#     embedding=OllamaEmbeddings(
#         model="nomic-embed-text:latest"
#     ),
#     persist_directory="./.chroma_db"

# )

retriever = Chroma(
    collection_name="rag-chroma",
    embedding_function=OllamaEmbeddings(
        model="nomic-embed-text:latest"
    ),
    persist_directory="./.chroma_db"
).as_retriever()