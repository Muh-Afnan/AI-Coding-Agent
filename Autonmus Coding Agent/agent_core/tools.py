# agent_core/tools.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List

@dataclass
class ToolResult:
    success: bool
    output: Any
    error: str = ""

class Tool(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        pass
    
    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        pass

class ReadFileTool(Tool):
    def __init__(self, file_manager: FileManager):
        self.fm = file_manager
    
    @property
    def name(self) -> str:
        return "read_file"
    
    @property
    def description(self) -> str:
        return "Read contents of a file. Args: filepath (str)"
    
    def execute(self, filepath: str) -> ToolResult:
        try:
            content = self.fm.read_file(filepath)
            return ToolResult(success=True, output=content)
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))

class ToolRegistry:
    def __init__(self):
        self.tools: Dict[str, Tool] = {}
    
    def register(self, tool: Tool):
        self.tools[tool.name] = tool
    
    def execute(self, tool_name: str, **kwargs) -> ToolResult:
        if tool_name not in self.tools:
            return ToolResult(False, None, f"Unknown tool: {tool_name}")
        return self.tools[tool_name].execute(**kwargs)
    
    def get_tools_description(self) -> str:
        """Format for LLM prompt."""
        return "\n".join([
            f"- {tool.name}: {tool.description}"
            for tool in self.tools.values()
        ])