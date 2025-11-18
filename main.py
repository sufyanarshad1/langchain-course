from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch

# Define tools
tools = [TavilySearch(max_results=5)]

# Initialize the LLM
llm = ChatOllama(model="llama3.2", temperature=0)

# Create agent using the new LangChain 1.0 API
# No need for AgentExecutor or custom prompts - create_agent handles everything
agent = create_agent(
    model=llm,
    tools=tools,
    # Optional: Add a system prompt to guide the agent's behavior
    system_prompt="You are a helpful assistant that searches for information and provides detailed summaries."
)

def main():
    # Invoke the agent with the new message format
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Search for three job postings for 'Software Engineer' in 'New York City' and summarize the key qualifications required for each position."
                }
            ]
        }
    )
    
    # Print the final response
    print("\n=== Agent Response ===")
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()