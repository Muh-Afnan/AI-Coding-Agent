"""
Tool Registry System - Pluggable action framework for agents.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Callable
from enum import Enum
import json


class ToolStatus(Enum):
    """Status of tool execution."""
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"


@dataclass
class ToolResult:
    """Result from tool execution."""
    status: ToolStatus
    output: Any
    error: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        data = asdict(self)
        data['status'] = self.status.value
        return data
    
    def is_success(self) -> bool:
        """Check if execution was successful."""
        return self.status == ToolStatus.SUCCESS


@dataclass
class ToolParameter:
    """Parameter definition for a tool."""
    name: str
    type: str  # 'string', 'number', 'boolean', 'array', 'object'
    description: str
    required: bool = True
    default: Any = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


class Tool(ABC):
    """
    Abstract base class for all agent tools.
    
    A tool is any action the agent can take to interact with the world.
    Examples: read file, write file, run command, search web, etc.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this tool."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of what this tool does."""
        pass
    
    @property
    def parameters(self) -> List[ToolParameter]:
        """List of parameters this tool accepts."""
        return []
    
    @property
    def category(self) -> str:
        """Category of tool (e.g., 'file', 'code', 'execution')."""
        return "general"
    
    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """
        Execute the tool with given parameters.
        
        Returns:
            ToolResult with status and output
        """
        pass
    
    def validate_params(self, **kwargs) -> tuple[bool, Optional[str]]:
        """
        Validate parameters before execution.
        
        Returns:
            (is_valid, error_message)
        """
        for param in self.parameters:
            if param.required and param.name not in kwargs:
                return False, f"Missing required parameter: {param.name}"
        return True, None
    
    def to_dict(self) -> dict:
        """Convert tool definition to dictionary for LLM."""
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "parameters": [p.to_dict() for p in self.parameters]
        }


class ToolRegistry:
    """
    Central registry for all available tools.
    
    Manages tool registration, discovery, and execution.
    """
    
    def __init__(self, verbose: bool = False):
        self.tools: Dict[str, Tool] = {}
        self.verbose = verbose
        self._execution_history: List[dict] = []
    
    def register(self, tool: Tool) -> None:
        """
        Register a new tool.
        
        Args:
            tool: Tool instance to register
        
        Raises:
            ValueError: If tool with same name already exists
        """
        if tool.name in self.tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")
        
        self.tools[tool.name] = tool
        
        if self.verbose:
            print(f"✅ Registered tool: {tool.name}")
    
    def unregister(self, tool_name: str) -> bool:
        """
        Remove a tool from registry.
        
        Args:
            tool_name: Name of tool to remove
        
        Returns:
            True if tool was removed, False if not found
        """
        if tool_name in self.tools:
            del self.tools[tool_name]
            if self.verbose:
                print(f"🗑️  Unregistered tool: {tool_name}")
            return True
        return False
    
    def get_tool(self, tool_name: str) -> Optional[Tool]:
        """Get tool by name."""
        return self.tools.get(tool_name)
    
    def has_tool(self, tool_name: str) -> bool:
        """Check if tool exists."""
        return tool_name in self.tools
    
    def list_tools(self, category: Optional[str] = None) -> List[str]:
        """
        List all registered tools.
        
        Args:
            category: Optional filter by category
        
        Returns:
            List of tool names
        """
        if category:
            return [
                name for name, tool in self.tools.items()
                if tool.category == category
            ]
        return list(self.tools.keys())
    
    def execute(self, tool_name: str, **kwargs) -> ToolResult:
        """
        Execute a tool with given parameters.
        
        Args:
            tool_name: Name of tool to execute
            **kwargs: Parameters for the tool
        
        Returns:
            ToolResult with execution outcome
        """
        # Check if tool exists
        if tool_name not in self.tools:
            return ToolResult(
                status=ToolStatus.FAILURE,
                output=None,
                error=f"Tool '{tool_name}' not found. Available tools: {', '.join(self.tools.keys())}"
            )
        
        tool = self.tools[tool_name]
        
        # Validate parameters
        is_valid, error_msg = tool.validate_params(**kwargs)
        if not is_valid:
            return ToolResult(
                status=ToolStatus.FAILURE,
                output=None,
                error=f"Parameter validation failed: {error_msg}"
            )
        
        # Execute tool
        try:
            if self.verbose:
                print(f"🔧 Executing: {tool_name}")
                print(f"   Parameters: {kwargs}")
            
            result = tool.execute(**kwargs)
            
            # Track execution
            self._execution_history.append({
                "tool": tool_name,
                "params": kwargs,
                "status": result.status.value,
                "success": result.is_success()
            })
            
            if self.verbose:
                status_emoji = "✅" if result.is_success() else "❌"
                print(f"{status_emoji} Result: {result.status.value}")
            
            return result
        
        except Exception as e:
            if self.verbose:
                print(f"❌ Tool execution failed: {e}")
            
            return ToolResult(
                status=ToolStatus.FAILURE,
                output=None,
                error=f"Execution error: {str(e)}"
            )
    
    def get_tools_description(self, format: str = "text") -> str:
        """
        Get description of all tools formatted for LLM.
        
        Args:
            format: 'text' or 'json'
        
        Returns:
            Formatted string describing available tools
        """
        if format == "json":
            tools_list = [tool.to_dict() for tool in self.tools.values()]
            return json.dumps(tools_list, indent=2)
        
        # Text format
        description = "Available Tools:\n\n"
        
        # Group by category
        by_category: Dict[str, List[Tool]] = {}
        for tool in self.tools.values():
            if tool.category not in by_category:
                by_category[tool.category] = []
            by_category[tool.category].append(tool)
        
        for category, tools in sorted(by_category.items()):
            description += f"## {category.upper()}\n"
            for tool in tools:
                description += f"\n### {tool.name}\n"
                description += f"{tool.description}\n"
                
                if tool.parameters:
                    description += "Parameters:\n"
                    for param in tool.parameters:
                        req = "required" if param.required else "optional"
                        description += f"  - {param.name} ({param.type}, {req}): {param.description}\n"
                
                description += "\n"
        
        return description
    
    def get_execution_history(self) -> List[dict]:
        """Get history of tool executions."""
        return self._execution_history.copy()
    
    def clear_history(self):
        """Clear execution history."""
        self._execution_history.clear()
    
    def get_statistics(self) -> dict:
        """Get usage statistics for all tools."""
        stats = {
            "total_executions": len(self._execution_history),
            "by_tool": {},
            "success_rate": 0.0
        }
        
        if not self._execution_history:
            return stats
        
        # Count by tool
        for execution in self._execution_history:
            tool_name = execution["tool"]
            if tool_name not in stats["by_tool"]:
                stats["by_tool"][tool_name] = {
                    "count": 0,
                    "successes": 0,
                    "failures": 0
                }
            
            stats["by_tool"][tool_name]["count"] += 1
            if execution["success"]:
                stats["by_tool"][tool_name]["successes"] += 1
            else:
                stats["by_tool"][tool_name]["failures"] += 1
        
        # Calculate success rate
        total_successes = sum(1 for e in self._execution_history if e["success"])
        stats["success_rate"] = (total_successes / len(self._execution_history)) * 100
        
        return stats


# ============================================================================
# CONCRETE TOOL IMPLEMENTATIONS
# ============================================================================

class ReadFileTool(Tool):
    """Tool for reading file contents."""
    
    def __init__(self, file_manager):
        self.file_manager = file_manager
    
    @property
    def name(self) -> str:
        return "read_file"
    
    @property
    def description(self) -> str:
        return "Read the contents of a file from the project"
    
    @property
    def category(self) -> str:
        return "file"
    
    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="filepath",
                type="string",
                description="Relative path to the file to read",
                required=True
            )
        ]
    
    def execute(self, filepath: str) -> ToolResult:
        try:
            content = self.file_manager.read_file(filepath)
            return ToolResult(
                status=ToolStatus.SUCCESS,
                output=content,
                metadata={"filepath": filepath, "size": len(content)}
            )
        except Exception as e:
            return ToolResult(
                status=ToolStatus.FAILURE,
                output=None,
                error=str(e)
            )


class WriteFileTool(Tool):
    """Tool for writing content to a file."""
    
    def __init__(self, file_manager):
        self.file_manager = file_manager
    
    @property
    def name(self) -> str:
        return "write_file"
    
    @property
    def description(self) -> str:
        return "Write content to a file in the project"
    
    @property
    def category(self) -> str:
        return "file"
    
    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="filepath",
                type="string",
                description="Relative path to the file to write",
                required=True
            ),
            ToolParameter(
                name="content",
                type="string",
                description="Content to write to the file",
                required=True
            )
        ]
    
    def execute(self, filepath: str, content: str) -> ToolResult:
        try:
            self.file_manager.write_file(filepath, content)
            return ToolResult(
                status=ToolStatus.SUCCESS,
                output=f"Successfully wrote {len(content)} characters to {filepath}",
                metadata={"filepath": filepath, "size": len(content)}
            )
        except Exception as e:
            return ToolResult(
                status=ToolStatus.FAILURE,
                output=None,
                error=str(e)
            )


class ListFilesTool(Tool):
    """Tool for listing files in the project."""
    
    def __init__(self, project_root: str):
        self.project_root = project_root
    
    @property
    def name(self) -> str:
        return "list_files"
    
    @property
    def description(self) -> str:
        return "List files in the project directory"
    
    @property
    def category(self) -> str:
        return "file"
    
    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="pattern",
                type="string",
                description="Glob pattern to filter files (e.g., '*.py')",
                required=False,
                default="*"
            )
        ]
    
    def execute(self, pattern: str = "*") -> ToolResult:
        try:
            from pathlib import Path
            
            root = Path(self.project_root)
            files = []
            
            for file_path in root.rglob(pattern):
                if file_path.is_file():
                    # Exclude common directories
                    if any(excluded in file_path.parts for excluded in ['.git', '__pycache__', '.venv', 'node_modules']):
                        continue
                    files.append(str(file_path.relative_to(root)))
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                output=files,
                metadata={"count": len(files), "pattern": pattern}
            )
        except Exception as e:
            return ToolResult(
                status=ToolStatus.FAILURE,
                output=None,
                error=str(e)
            )


class RunCommandTool(Tool):
    """Tool for executing shell commands."""
    
    def __init__(self, executor):
        self.executor = executor
    
    @property
    def name(self) -> str:
        return "run_command"
    
    @property
    def description(self) -> str:
        return "Execute a shell command in the project directory"
    
    @property
    def category(self) -> str:
        return "execution"
    
    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="command",
                type="string",
                description="Shell command to execute",
                required=True
            )
        ]
    
    def execute(self, command: str) -> ToolResult:
        result = self.executor.run_command(command)
        
        if result.returncode == 0:
            return ToolResult(
                status=ToolStatus.SUCCESS,
                output={
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "returncode": result.returncode
                },
                metadata={"command": command}
            )
        else:
            return ToolResult(
                status=ToolStatus.FAILURE,
                output={
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "returncode": result.returncode
                },
                error=f"Command failed with exit code {result.returncode}"
            )


# Example usage
if __name__ == "__main__":
    from agent_core.file_manager import FileManager
    from agent_core.executor import Executor
    
    # Create registry
    registry = ToolRegistry(verbose=True)
    
    # Register tools
    fm = FileManager(project_root=".")
    executor = Executor(cwd=".")
    
    registry.register(ReadFileTool(fm))
    registry.register(WriteFileTool(fm))
    registry.register(ListFilesTool("."))
    registry.register(RunCommandTool(executor))
    
    # List available tools
    print("\n" + "="*60)
    print("REGISTERED TOOLS")
    print("="*60)
    print(registry.get_tools_description())
    
    # Execute a tool
    print("\n" + "="*60)
    print("EXECUTING TOOL")
    print("="*60)
    result = registry.execute("list_files", pattern="*.py")
    print(f"Status: {result.status.value}")
    print(f"Found {len(result.output)} Python files")
    
    # Show statistics
    print("\n" + "="*60)
    print("STATISTICS")
    print("="*60)
    stats = registry.get_statistics()
    print(f"Total executions: {stats['total_executions']}")
    print(f"Success rate: {stats['success_rate']:.1f}%")