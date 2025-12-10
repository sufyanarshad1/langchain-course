from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama

class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource:Literal["vectorstore", "websearch"] = Field(
        description = "Given a user question, choose to route to 'vectorstore' or 'websearch' based on which datasource is more likely to have the answer."
    )

llm = ChatOllama(model="llama3.2:latest", temperature=0)
structured_llm_router = llm.with_structured_output(RouteQuery)

system = """You are an intelligent router that determines the best datasource to answer a user question. \n
    You have access to two datasources: a vectorstore containing documents, and a websearch tool. \n
    If the user question is specific and likely to be answered by existing documents, route to the vectorstore. \n
    If the user question is broad, timely, or likely to require up-to-date information, route to the websearch tool. \n
    Always choose the datasource that is most likely to provide a useful answer to the user's question."""

route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User question: {question}")
    ]
)

question_router = route_prompt | structured_llm_router 

