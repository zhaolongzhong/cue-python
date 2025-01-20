from typing import Literal, ClassVar, Optional
from pathlib import Path

from .base import BaseTool, ToolError, ToolResult
from ..types.tool_response_wrapper import DEFAULT_MAX_MESSAGES, MAX_ALLOWED_MESSAGES, AgentTransfer

Command = Literal["transfer"]


class CoordinateTool(BaseTool):
    """
    An agent coordination tool that handle agent coordination like transfer task
    """

    name: ClassVar[Literal["coordinate"]] = "coordinate"

    _file_history: dict[Path, list[str]]

    def __init__(self):
        self._function = self.coordinate
        super().__init__()

    async def __call__(
        self,
        *,
        command: Command,
        to_agent_id: str,
        message: str,
        **kwargs,
    ):
        return await self.coordinate(
            command=command,
            to_agent_id=to_agent_id,
            message=message,
            **kwargs,
        )

    async def coordinate(
        self,
        *,
        command: Command,
        to_agent_id: str,
        message: str,
        max_messages: Optional[int] = DEFAULT_MAX_MESSAGES,
        **kwargs,
    ):
        if command == "transfer":
            return self.transfer(
                to_agent_id=to_agent_id,
                message=message,
                max_messages=max_messages,
                **kwargs,
            )
        else:
            return ToolError(f"Unsupport command '{command}', supported command are {Command}")

    def transfer(
        self,
        *,
        to_agent_id: str,
        message: str,
        max_messages: Optional[int] = 6,
        **kwargs,
    ) -> AgentTransfer:
        max_messages = DEFAULT_MAX_MESSAGES if max_messages is None else min(max(0, max_messages), MAX_ALLOWED_MESSAGES)
        transfer_info = AgentTransfer(to_agent_id=to_agent_id, message=message, max_messages=max_messages)
        mode_desc = (
            "using message only"
            if max_messages == 0
            else f"with up to {max_messages} previous messages (excluding this transfer command)"
        )
        return ToolResult(
            output=(
                f"Transfer request created successfully ({mode_desc}). "
                "Details: {transfer_info.model_dump_json(indent=2)}"
            ),
            agent_transfer=transfer_info,
        )
