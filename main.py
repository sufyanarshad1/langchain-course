import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate

load_dotenv()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

if __name__ == "__main__":
    embeddings = OllamaEmbeddings(model="mxbai-embed-large:latest")
    llm = ChatOllama(model="llama3.2", temperature=0)
    query = "What is pinecone in machine learning?"

    vectorStore = PineconeVectorStore(
        index_name=os.getenv("INDEX_NAME"),
        embedding=embeddings
    )

    @tool(response_format="content_and_artifact")
    def retrieve_context(query: str):
        """Retrieve information from the vector database to help answer a query."""
        retrieved_docs = vectorStore.similarity_search(query, k=3)
        serialized = "\n\n".join(
            (f"Source: {doc.metadata}\nContent: {doc.page_content}")
            for doc in retrieved_docs
        )
        return serialized, retrieved_docs

    agent = create_agent(
        llm, 
        [retrieve_context], 
        system_prompt="You are a helpful assistant. Use the retrieve_context tool to answer questions."
    )

    result = agent.invoke({"messages": [{"role": "user", "content": query}]})
    print("\nAnswer:", result["messages"][-1].content)

    template = """Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Use three sentences maximum and keep the answer concise.
    Always say "Thanks for asking!" at the end of your response.

    {context}

    Question: {question}
    Helpful Answer:"""

    custom_rag_prompt = PromptTemplate.from_template(template)

    rag_chain = (
        {"context": vectorStore.as_retriever() | format_docs , "question": RunnablePassthrough()}
        | custom_rag_prompt
        | llm
    )

    res = rag_chain.invoke(query)