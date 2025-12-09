from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama


llm = ChatOllama(model="llama3.2:latest", temperature=0)

class GradeHallucinations(BaseModel):
    """Binary score for hallucination check on generated answer."""

    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'."
    )


structured_llm_grader = llm.with_structured_output(GradeHallucinations)

system = """You are a strict grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n
     Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in the set of facts and contains no hallucinations or unsupported claims. \n
     'No' means that the answer contains information not present in the facts or makes claims that cannot be verified from the provided facts. \n
     The answer must be fully supported by the facts - if any part is not grounded in the facts, score it as 'no'."""

halllucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Set of facts: \n\n{documents} \n\n  LLM Generation:{generation}")
    ]
)

hallucination_grader : RunnableSequence = halllucination_prompt | structured_llm_grader