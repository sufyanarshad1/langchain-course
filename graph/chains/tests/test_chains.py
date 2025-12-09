import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
from pprint import pprint

load_dotenv()

from graph.chains.retrieval_grade import retrieval_grader, GradeDocuments
from graph.chains.hallucination_grader import hallucination_grader, GradeHallucinations
from graph.chains.generation import generation_chain
from ingestion import retriever



# def test_retrieval_grader_answer_yes() -> None:
#     question = "agent memory"
#     docs = retriever.invoke(question)
#     doc_txt = docs[1].page_content

#     res: GradeDocuments = retrieval_grader.invoke(
#         {
#             "document": doc_txt,
#             "question": question
#         }
#     )

#     assert res.binary_score.lower() == "yes"

# def test_retrieval_grader_answer_no() -> None:
#     question = "how to make pizza dough?"
#     docs = retriever.invoke(question)
#     doc_txt = docs[1].page_content

#     res: GradeDocuments = retrieval_grader.invoke(
#         {
#             "document": doc_txt,
#             "question": question
#         }
#     )

#     assert res.binary_score.lower() == "no"


    # def test_generation_chain() -> None:
    #     question = "agent memory"
    #     docs = retriever.invoke(question)
    #     generation = generation_chain.invoke(
    #         {
    #             "question": question,
    #             "context": docs
    #         }
    #     )
    #     pprint(generation)

def test_hallucination_grader_answer_yes() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)

    generation = generation_chain.invoke({
        "question": question,
        "context": docs
    })
    res: GradeHallucinations = hallucination_grader.invoke(
        {
            "documents": docs,
            "generation": generation
        }
    )
    assert res.binary_score

   
def test_hallucination_grader_answer_no() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)

    res: GradeHallucinations = hallucination_grader.invoke(
        {
            "documents": docs,
            "generation": "In order to make pizza we need to first start with the dough",
        }
    )
    assert not res.binary_score