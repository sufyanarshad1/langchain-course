from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3.2:latest", temperature=0)   


class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description = "Documents are relevant to the question. Answer with 'yes' or 'no'."
    )


structured_llm_grader = llm.with_structured_output(GradeDocuments)


system = """You are a strict grader assessing relevance of a retrieved document to a user question. \n 
    The document must directly contain information that can answer the question or provide meaningful context about the topic being asked. \n
    Generic mentions, vague connections, or tangentially related content should be marked as NOT relevant. \n
    Only grade as relevant if the document specifically addresses the question topic with concrete information. \n
    Give a binary score 'yes' or 'no' to indicate whether the document is relevant to the question. \n
    When in doubt, answer 'no'."""

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n{document}\n\n User question: {question}")
    ]
)

retrieval_grader = grade_prompt | structured_llm_grader