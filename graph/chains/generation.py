from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langsmith import Client

llm = ChatOllama(model="llama3.2:latest", temperature=0)
client = Client()
prompt = client.pull_prompt("rlm/rag-prompt")

generation_chain = prompt | llm | StrOutputParser()