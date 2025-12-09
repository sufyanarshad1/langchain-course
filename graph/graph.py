from dotenv import load_dotenv

from langgraph.graph import END, StateGraph

from graph.consts import RETRIEVE, GENERATE, GRADE_DOCUMENTS, WEB_SEARCH
from graph.nodes import retrieve, generate, grade_documents, web_search
from graph.state import GraphState


load_dotenv()

def decide_to_generate(state):
    print("---Assess graded documents---")

    if state["web_search"]:
        print("Decision: Not all documents are relevant, proceeding to generate with web search results.")
        return WEB_SEARCH
    else:
        print("Decision: All documents are relevant, proceeding to generate.")
        return GENERATE
    
workflow = StateGraph(GraphState)

workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(GENERATE, generate)
workflow.add_node(WEB_SEARCH, web_search)

workflow.set_entry_point(RETRIEVE)
workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
workflow.add_conditional_edges(
    GRADE_DOCUMENTS,
    decide_to_generate,
    {
        WEB_SEARCH: WEB_SEARCH,
        GENERATE: GENERATE
    },
)
workflow.add_edge(WEB_SEARCH, END)
workflow.add_edge(GENERATE, END)

app = workflow.compile()

app.get_graph().draw_mermaid_png(output_file_path="graph/graph.png")
