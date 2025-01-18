import logging
from enum import Enum
from typing import Union, Literal, ClassVar, Optional, Annotated
from pathlib import Path

from pydantic import Field, BaseModel

from .base import BaseTool, ToolError, ToolResult
from ..services import AssistantClient

logger = logging.getLogger(__name__)


class Command(str, Enum):
    CHAT = "chat"
    SCHEDULE = "schedule"


class Schedule(BaseModel):
    title: Optional[str] = None
    prompt: Optional[str] = None
    schedule: Optional[str] = None


class ChatRequest(BaseModel):
    conversation_id: Optional[str] = None
    command: Command
    content: Annotated[
        Union[str, Schedule],
        Field(
            discriminator="command",
            mapping={Command.SCHEDULE: Schedule, Command.CHAT: str},
        ),
    ]


class ChatTool(BaseTool):
    """
    A chat tool that allows to chat with command
    """

    name: ClassVar[Literal["chat"]] = "chat"
    _file_history: dict[Path, list[str]]

    def __init__(self, assistant_client: Optional[AssistantClient]):
        self._function = self.chat
        self.assistant_client = assistant_client
        super().__init__()

    async def __call__(
        self,
        *,
        command: Command,
        content: Union[str, Schedule],
        conversation_id: Optional[str] = None,
        **kwargs,
    ):
        return await self.chat(
            command=command,
            content=content,
            conversation_id=conversation_id,
            **kwargs,
        )

    async def chat(
        self,
        *,
        command: Command,
        content: Union[str, Schedule],
        conversation_id: Optional[str] = None,
    ) -> ToolResult:
        """
        Send chat request to the assistant service

        Args:
            command: Type of chat request (e.g., schedule,)
            content: Content of the request (Schedule for scheduling)
            conversation_id: Optional ID for conversation tracking
        """
        if self.assistant_client is None:
            error_msg = "Chat tool is called but external assistant service is not enabled."
            logger.error(error_msg)
            raise ToolError(error_msg)

        try:
            chat_request = ChatRequest(command=command, content=content, conversation_id=conversation_id)
            response = await self.assistant_client.chat(chat_request)
            return ToolResult(output=response)
        except Exception as e:
            raise ToolError(f"Failed to execute chat: {str(e)}")
