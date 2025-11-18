from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch

from schemas import AgentResponse

# Define tools
tools = [TavilySearch(max_results=5)]

# Initialize the LLM
llm = ChatOllama(model="llama3.2", temperature=0)

# Create agent using the new LangChain 1.0 API with structured output
# In v1.0, structured output is handled via response_format parameter
agent = create_agent(
    model=llm,
    tools=tools,
    # Use ToolStrategy for structured output with models that support tool calling
    response_format=ToolStrategy(AgentResponse),
    # System prompt guides the agent's behavior
    system_prompt="""You are a helpful assistant that searches for information and provides detailed summaries.
When you provide your final answer, make sure to include the answer text and list the sources (URLs) you used.""",
)


def main():
    # Invoke the agent with the new message format
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Search for three job postings for 'Software Engineer' in 'New York City' and summarize the key qualifications required for each position.",
                }
            ]
        }
    )

    # Print the final response
    print("\n=== Agent Response ===")
    print(result["messages"][-1].content)

    # Access structured output if available
    if "structured_response" in result:
        structured = result["structured_response"]
        print("\n=== Structured Response ===")
        print(f"Answer: {structured.answer}")
        print(f"\nSources:")
        for source in structured.source:
            print(f"  - {source.url}")


if __name__ == "__main__":
    main()
