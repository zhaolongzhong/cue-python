from typing import Any, Dict, List

# Import v1 tools we want to reuse
try:
    from ..tools.base import BaseTool, ToolResult
    from ..tools.edit import EditTool
    from ..tools.bash_tool import BashTool
except ImportError:
    # Fallback if v1 tools not available
    ToolResult = None
    BaseTool = None


class SimpleToolResult:
    """Simplified tool result for v2"""

    def __init__(self, output: str = "", error: str = "", success: bool = True):
        self.output = output
        self.error = error
        self.success = success

    def __str__(self):
        return self.output or self.error or "Tool completed"


class ToolExecutor:
    """Hybrid tool executor - reuses v1 tools with v2 simplicity"""

    def __init__(self):
        self.tools: Dict[str, BaseTool] = {}
        self._initialize_tools()

    def _initialize_tools(self):
        """Initialize available tools"""
        if BashTool:
            self.tools["bash"] = BashTool()
        if EditTool:
            self.tools["edit"] = EditTool()

    async def execute(self, name: str, arguments: Dict[str, Any]) -> SimpleToolResult:
        """Execute a tool and return simplified result"""
        tool = self.tools.get(name)
        if not tool:
            return SimpleToolResult(
                error=f"Tool '{name}' not found. Available: {list(self.tools.keys())}", success=False
            )

        try:
            # Execute the v1 tool
            result = await tool(**arguments)

            # Convert v1 ToolResult to v2 SimpleToolResult
            if hasattr(result, "output") and hasattr(result, "error"):
                return SimpleToolResult(output=result.output or "", error=result.error or "", success=not result.error)
            else:
                # Handle string results
                return SimpleToolResult(output=str(result), success=True)

        except Exception as e:
            return SimpleToolResult(error=f"Tool execution failed: {str(e)}", success=False)

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Get tool schemas for LLM function calling"""
        schemas = []
        for name, tool in self.tools.items():
            try:
                # Use v1 tool's to_json() method to get schema
                schema = tool.to_json()
                schemas.append(schema)
            except Exception:
                # Fallback basic schema
                schemas.append(
                    {
                        "type": "function",
                        "function": {
                            "name": name,
                            "description": f"{name} tool",
                            "parameters": {"type": "object", "properties": {}, "required": []},
                        },
                    }
                )
        return schemas

    def has_tool(self, name: str) -> bool:
        """Check if tool is available"""
        return name in self.tools


# Convenience function for quick tool execution
async def execute_tool(name: str, **kwargs) -> str:
    """Quick tool execution - returns string result"""
    executor = ToolExecutor()
    result = await executor.execute(name, kwargs)
    return str(result)
