from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a viral X(twitter) influencer grading a tweet. Generate critique and recommendations for the user's tweet."
            "Always provide detail recommendation, including request for length, virality, syle, etc."

        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a X(twitter) techie influencer assistant tasked with writing excellent X(twitter) posts."
            "Generate the best X(twitter) post for the user's request."
            "If your user provide critique, respond with a revised version of your previous attempts"
        ),
        MessagesPlaceholder(variable_name="messages")
    ]
)

llm = ChatOllama(model="llama3.2:latest", temperature=0)
generate_chain = generation_prompt | llm 
reflect_chain  = reflection_prompt | llm