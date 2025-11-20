from langchain_core.callbacks import BaseCallbackHandler
from typing import Any, Dict


class CustomCallbackHandler(BaseCallbackHandler):
    """Custom callback handler to track agent execution."""
    
    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs) -> None:
        """Called when a tool starts running."""
        tool_name = serialized.get("name", "Unknown")
        print(f"\n🔧 Tool Started: {tool_name}")
        print(f"   Input: {input_str}")
    
    def on_tool_end(self, output: str, **kwargs) -> None:
        """Called when a tool finishes running."""
        print(f"   Output: {output}")
        print(f"✅ Tool Finished\n")
    
    def on_llm_start(self, serialized: Dict[str, Any], prompts: list[str], **kwargs) -> None:
        """Called when LLM starts."""
        print(f"\n🤖 LLM Thinking...")
    
    def on_llm_end(self, response, **kwargs) -> None:
        """Called when LLM finishes."""
        print(f"💭 LLM Response Generated\n")
