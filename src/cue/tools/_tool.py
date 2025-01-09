import logging
from enum import Enum
from typing import Any, Dict, List, Union, Optional

from .base import BaseTool
from .edit import EditTool
from .drive import GoogleDriveTool
from .email import EmailTool
from .browse import BrowseTool
from .memory import MemoryTool
from .restart import RestartTool
from ..services import ServiceManager
from .bash_tool import BashTool
from .coordinate import CoordinateTool
from .read_image import ReadImageTool
from .run_script import PythonRunner
from .mcp_manager import MCPServerManager
from .system_tool import SystemTool
from .github_project import GitHubProjectTool
from .project_context import ProjectContextTool
from .utils.function_utils import get_definition_by_model
from .utils.function_to_json import function_to_json

logger = logging.getLogger(__name__)


class Tool(Enum):
    Bash = BashTool.name
    Edit = EditTool.name
    Python = PythonRunner.name
    Browse = BrowseTool.name
    Email = EmailTool.name
    Drive = GoogleDriveTool.name
    Image = ReadImageTool.name
    Memory = MemoryTool.name
    Coordinate = CoordinateTool.name
    Restart = RestartTool.name
    GitHubProject = GitHubProjectTool.name
    ProjectContextTool = ProjectContextTool.name
    SystemTool = SystemTool.name


class ToolManager:
    def __init__(
        self,
        service_manager: Optional[ServiceManager] = None,
        mcp: Optional[MCPServerManager] = None,
    ):
        self.tools: Dict[str, BaseTool] = {
            Tool.Bash.value: BashTool(),
            Tool.Edit.value: EditTool(),
            Tool.Python.value: PythonRunner(),
            Tool.Browse.value: BrowseTool(),
            Tool.Email.value: EmailTool(),
            Tool.Drive.value: GoogleDriveTool(),
            Tool.Image.value: ReadImageTool(),
            Tool.Coordinate.value: CoordinateTool(),
            Tool.Restart.value: RestartTool(),
            Tool.GitHubProject.value: GitHubProjectTool(),
        }
        self.service_manager = service_manager
        if self.service_manager:
            self.tools[Tool.Memory.value] = MemoryTool(self.service_manager.memories)
            self.tools[Tool.ProjectContextTool.value] = ProjectContextTool(self.service_manager.assistants)
            self.tools[Tool.SystemTool.value] = SystemTool(self.service_manager.assistants)
        self._definition_cache: Dict[str, dict] = {}
        self.mcp = mcp  # disable for now
        self._mcp_tools_json = []
        self.mcp_tools_map: Dict[str, Dict[str, Any]] = {}

    async def initialize(self):
        if self.mcp:
            self._mcp_tools_json = self.mcp.list_tools_json()
            self._map_mcp_tools()

    def _map_mcp_tools(self):
        """
        Maps the MCP tools JSON list to a dictionary for efficient access.
        Assumes each tool in the list has a unique 'id' or 'name'.
        """
        for tool in self._mcp_tools_json:
            tool_id = tool.get("id") or tool.get("name")
            if tool_id:
                self.mcp_tools_map[tool_id] = tool
            else:
                # Handle tools without an 'id' or 'name'
                raise ValueError(f"Tool entry missing 'id' or 'name': {tool}")

    async def clean_up(self):
        pass

    def _get_cache_key(self, tool_id: str, model: str) -> str:
        """Generate a unique cache key for a tool definition."""
        return f"{tool_id}_{model}"

    def _get_cached_definition(self, tool_id: str, model: str) -> Optional[dict]:
        """Retrieve cached tool definition if it exists."""
        return self._definition_cache.get(self._get_cache_key(tool_id, model))

    def _cache_definition(self, tool_id: str, model: str, definition: dict):
        """Cache a tool definition."""
        self._definition_cache[self._get_cache_key(tool_id, model)] = definition

    def has_tool(self, tool_name: str) -> bool:
        if tool_name in self.tools:
            return True
        if tool_name in self.mcp_tools_map:
            return True
        return False

    def get_mcp_tools(self, model: str) -> list[dict]:
        tools = []
        for item in self._mcp_tools_json:
            tool_dict = get_definition_by_model(item, model)
            tools.append(tool_dict)
        return tools

    def get_tool_definitions(self, model: str, tools: Optional[Union[List[Tool], List[str]]] = None) -> list[dict]:
        """Iterate through the tools and gather their JSON configurations.

        Args:
            model: Original model name from original provider
            tools: Optional list of Tool enum values or tool string identifiers

        Returns:
            List of tool parameter dictionaries
        """
        if tools is None:
            tools_to_iterate = list(self.tools.keys())
        else:
            tools_to_iterate = []
            for tool in tools:
                if isinstance(tool, Tool):
                    tools_to_iterate.append(tool.value)
                elif isinstance(tool, str):  # name if the tool
                    tools_to_iterate.append(tool)
                elif isinstance(tool, dict):
                    name = tool.get("name", None)
                    if name:
                        tools_to_iterate.append(name)
                elif callable(tool):
                    # Convert callable to JSON definition and add to tools
                    tool_id = tool.__name__
                    cached_def = self._get_cached_definition(tool_id, model)
                    if cached_def is None:
                        definition = get_definition_by_model(function_to_json(tool), model)
                        self.tools[tool_id] = definition
                        self._cache_definition(tool_id, model, definition)
                    else:
                        self.tools[tool_id] = cached_def
                    tools_to_iterate.append(tool_id)
                else:
                    raise ValueError(f"Unsupported tool type: {type(tool)}")

        func_definitions = []
        for tool_id in tools_to_iterate:
            if not self.has_tool(tool_id):
                raise ValueError(
                    f"Unsupported tool type: {type(tool)}, tool_id: {tool_id}, make sure to register first in ToolManager"
                )

            cached_def = self._get_cached_definition(tool_id, model)
            if cached_def is not None:
                func_definitions.append(cached_def)
            else:
                tool = self.tools.get(tool_id, None)
                if isinstance(tool, dict):
                    definition = get_definition_by_model(tool, model=model)
                else:
                    definition = tool.to_json(model, use_file_definition=True)
                if definition:
                    self._cache_definition(tool_id, model, definition)
                    func_definitions.append(definition)

        return func_definitions
