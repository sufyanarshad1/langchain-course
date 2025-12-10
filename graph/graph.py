from dotenv import load_dotenv

from langgraph.graph import END, StateGraph

from graph.chains.answer_grader import answer_grader
from graph.chains.hallucination_grader import hallucination_grader
from graph.consts import RETRIEVE, GENERATE, GRADE_DOCUMENTS, WEB_SEARCH
from graph.chains.router import RouteQuery, question_router
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
    
def grade_generation_grounded_in_documents_and_question(state: GraphState)->str:
    print("---Check Hallucinations---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader.invoke(
        {
            "documents": documents,
            "generation": generation
        }
    )

    if hallucination_grade := score.binary_score:
        print("---Decision: Generation is grounded in documents---")
        print("---Grade Generation vs Question---")
        score = answer_grader.invoke(
            {
                "generation": generation,
                "question": question
            }
        )
        if answer_grade := score.binary_score:
            print("---Decision: Generation Addresses Question---")
            return "useful"
        else:
            print("---Decision: Generation Does Not Address Question---")
            return "not useful"
    else:
        print("---Decision: Generation is not grounded in documents---")
        return "not supported"

def route_question(state:GraphState) -> str:
    print("---Routing Question---")
    question = state["question"]

    source: RouteQuery = question_router.invoke(
        {
            "question": question
        }
    )

    if source.datasource == WEB_SEARCH:
        print("---Decision: Question requires web search---")
        return WEB_SEARCH
    elif source.datasource == "vectorstore":
        print("---Decision: Question can be answered with retrieval---")
        return RETRIEVE 

workflow = StateGraph(GraphState)

workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(GENERATE, generate)
workflow.add_node(WEB_SEARCH, web_search)

workflow.set_conditional_entry_point(
    route_question,
    {
        RETRIEVE: RETRIEVE,
        WEB_SEARCH: WEB_SEARCH
    },
)

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
workflow.add_conditional_edges(
    GENERATE,
    grade_generation_grounded_in_documents_and_question,
    {
        "useful": END,
        "not useful": WEB_SEARCH,
        "not supported": GENERATE
    },
)
workflow.add_edge(WEB_SEARCH, GENERATE)

app = workflow.compile()

app.get_graph().draw_mermaid_png(output_file_path="graph/graph.png")
