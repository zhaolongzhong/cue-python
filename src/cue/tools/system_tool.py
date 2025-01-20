import logging
from typing import Union, Literal, ClassVar, Optional, get_args
from pathlib import Path

from .base import BaseTool, ToolError, ToolResult
from ..schemas import AssistantUpdate, AssistantMetadata
from ..services import AssistantClient

logger = logging.getLogger(__name__)

Command = Literal[
    "view",
    "update",
]


class SystemTool(BaseTool):
    """
    A system tool that allows the agent to view and update project context
    """

    name: ClassVar[Literal["system_tool"]] = "system_tool"

    _file_history: dict[Path, list[str]]

    def __init__(self, assistant_client: Optional[AssistantClient]):
        self._function = self.system_tool
        self.assistant_client = assistant_client
        super().__init__()

    async def __call__(
        self,
        *,
        command: Command,
        new_content: Optional[str] = None,
        **kwargs,
    ):
        return await self.system_tool(
            command=command,
            new_content=new_content,
            **kwargs,
        )

    async def system_tool(
        self,
        *,
        command: Command,
        new_content: Optional[Union[dict, str]] = None,
        **kwargs,
    ):
        """Perform system operations like "view", "update"""
        if self.assistant_client is None:
            error_msg = "System tool is called but external assistant service is not enabled."
            logger.error(error_msg)
            raise ToolError(error_msg)
        if command == "view":
            return await self.view()
        elif command == "update":
            if not new_content:
                raise ToolError("Parameter `new_content` is required for command: update")
            return await self.update(new_content=new_content)
        raise ToolError(
            f"Unrecognized command {command}. The allowed commands for the {self.name} tool are: "
            f"{', '.join(get_args(Command))}"
        )

    async def view(self) -> ToolResult:
        response = await self.assistant_client.get_system_context()
        return ToolResult(output=str(response))

    async def update(
        self,
        new_content: Optional[Union[dict, str]] = None,
    ):
        try:
            await self.assistant_client.update(
                assistant=AssistantUpdate(metadata=AssistantMetadata(system=new_content))
            )
            success_msg = "System context has been edited."
            return ToolResult(output=success_msg)
        except Exception as e:
            raise ToolError(f"Ran into {e} while trying to update context for {new_content}") from None
