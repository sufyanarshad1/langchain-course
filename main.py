from typing import List

from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

import os
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch

class Source(BaseModel):
    url: str = Field(description="The URL of the source")

class AgentResponse(BaseModel):
    answer: str = Field(description="The final answer from the agent")
    sources: List[Source] = Field(default_factory=list)

llm = ChatOllama(model="llama3.2", temperature=0)

tools = [TavilySearch()]

agent = create_agent(
    model=llm,
    tools=tools
)

def main():
    print("Hello from langchain-course!")
    result = agent.invoke({
        "messages": [
            HumanMessage(
                content=(
                    "Search Google job results for 3 AI Engineer job postings "
                    "in the San Francisco Bay Area. "
                    "Return the job title, company, location, job URL, and a short summary for each."
                )
            )
        ]
    })
    print(result)

if __name__ == "__main__":
    main()
