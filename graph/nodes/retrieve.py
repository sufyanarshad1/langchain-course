from typing import Any, Dict

from graph.state import GraphState
from ingestion import retriever


def rerieve(state: GraphState) -> Dict[str, Any]:
    """Retrieve documents based on the question in the graph state.

    Args:
        state (GraphState): The current state of the graph.
    """
    print("Retrieving documents...")
    question = state["question"]

    document = retriever.invoke(question)
    return {"docuements": document, "question": question}