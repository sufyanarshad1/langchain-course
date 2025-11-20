from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain.agents import create_agent
from callbacks import CustomCallbackHandler

load_dotenv()


@tool
def get_text_length(text: str) -> int:
    """Returns the length of the given text by characters."""
    text = text.strip("'\n").strip('"')
    return len(text)


if __name__ == "__main__":
    print("LangChain ReAct Agent with Callbacks\n")
    tools = [get_text_length]

    llm = ChatOllama(model="llama3.2", temperature=0)

    # Create agent using LangChain 1.0 API
    agent = create_agent(llm, tools=tools)

    # Initialize custom callback handler for detailed tracking
    callback_handler = CustomCallbackHandler()

    # Invoke the agent with callback handler
    response = agent.invoke(
        {
            "messages": [
                {"role": "user", "content": "What is the length in characters of the text: DOG?"}
            ]
        },
        config={"callbacks": [callback_handler]}
    )
    
    # Get the final answer from the last message
    final_answer = response["messages"][-1].content
    print(f"\n{'='*50}")
    print(f"Final Answer: {final_answer}")
    print(f"{'='*50}")