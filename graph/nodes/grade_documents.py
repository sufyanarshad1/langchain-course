from typing import Any, Dict

from graph.chains.retrieval_grade import retrieval_grader
from graph.state import GraphState

def grade_documents(state: GraphState) -> Dict[str, Any]:
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state
    """
    print("---Check document relevance to the question---")
    quesstion = state["question"]
    documents = state["documents"]

    filtered_docs = []
    web_search = False

    for d in documents:
        score = retrieval_grader.invoke(
            {
                "document": d.page_content,
                "question": quesstion
            }
        )
        grade = score.binary_score.lower()
        if grade == "yes":
            print("Question is relevant.")
            filtered_docs.append(d)
        else:
            print("Question is not relevant. Will trigger web search.")
            web_search = True
            continue
    return {
        "documents": filtered_docs,
        "question": quesstion,
        "web_search": web_search
    }