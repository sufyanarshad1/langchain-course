from dotenv import load_dotenv
from langsmith import Client
from langchain_ollama import ChatOllama
from langchain_classic.agents import create_react_agent, AgentExecutor, create_csv_agent
from langchain_experimental.tools import PythonREPLTool 

load_dotenv()

def main():
    print("Start...")
    
    # *** THE SOLUTION INSTRUCTIONS ***
    # We explicitly tell the agent to wrap code in exec("""...""") 
    # This tricks the tool into accepting multi-line code without syntax errors.
    instructions = """You are an agent designed to write and execute Python code.
    You have access to a Python REPL.

    ## CRITICAL SYNTAX FIX:
    You are failing because you try to write 'import' and 'for' on one line.
    To fix this, you MUST wrap your code in an exec() block like this:

    Action Input: 
    exec(\"\"\"
    import qrcode
    for i in range(1, 16):
        img = qrcode.make('https://www.udemy.com/course/langchain')
        img.save(f'qrcode_{i}.png')
    \"\"\")

    ## YOUR TASK:
    Use the exact code format above to generate 15 QR codes. Do not change it.
    """

    client = Client()
    # Pull the prompt template
    base_prompt = client.pull_prompt("langchain-ai/react-agent-template")
    prompt = base_prompt.partial(instructions=instructions)

    tools = [PythonREPLTool()]

    agent = create_react_agent(
        # Llama 3.2 is faster and works perfectly with the exec() wrapper
        llm=ChatOllama(model="llama3.2", temperature=0),
        tools=tools,
        prompt=prompt,
    )

    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True
    )
    
    print("\n--- Invoking Agent Executor ---\n")
    
    agent_executor.invoke(
        input={
            "input": """Generate 15 QRcodes pointing to www.udemy.com/course/langchain. Name them qrcode_1.png to qrcode_15.png."""
        }
    )

    csv_agent = create_csv_agent(
        llm=ChatOllama(model="llama3.2", temperature=0),
        csv_file_path="data/sales_data.csv",
        verbose=True
    )

if __name__ == "__main__":
    main()