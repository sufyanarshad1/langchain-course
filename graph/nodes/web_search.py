import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
from typing import Any, Dict
from langchain_core.documents import Document
from langchain_community.tools.tavily_search import TavilySearchResults

from graph.state import GraphState

load_dotenv()

web_search_tool = TavilySearchResults(max_results=3)

def web_search(state: GraphState) -> Dict[str, Any]:
    print("---Performing web search---")
    question = state["question"]
    documents = state["documents"]

    tavily_results = web_search_tool.invoke({"query": question})
    joined_tavily_results = "\n".join(
        [tavily_result["content"] for tavily_result in tavily_results]
    )
    web_results = Document(page_content=joined_tavily_results)
    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]
    return {
        "question": question,
        "documents": documents,
    }

if __name__ == "__main__":
    web_search(state={
        "question": "agent memory",
        "documents": None
    })