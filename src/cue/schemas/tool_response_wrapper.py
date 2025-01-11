from typing import List, Union, Optional

from pydantic import Field, BaseModel

from .message import Author, Content, Metadata, ContentType, MessageCreate
from .run_metadata import RunMetadata

DEFAULT_MAX_MESSAGES = 6
MAX_ALLOWED_MESSAGES = 12


class AgentTransfer(BaseModel):
    to_agent_id: Optional[str] = Field(None, description="ID of the target agent")
    message: str = Field(..., description="Message to be sent to the target agent")
    max_messages: int = Field(
        default=DEFAULT_MAX_MESSAGES,
        ge=0,  # Changed to allow 0
        le=MAX_ALLOWED_MESSAGES,
        description="Maximum number of messages to transfer. 0 means using only the message field",
    )
    context: Optional[Union[str, List]] = None
    transfer_to_primary: bool = False
    run_metadata: Optional[RunMetadata] = None


class ToolResponseWrapper(BaseModel):
    msg_id: Optional[str] = Field(
        None,
        description="Unique identifier for the message in persistence layer",
    )
    author: Optional[Author] = None
    tool_messages: Optional[List[dict]] = None
    tool_result_message: Optional[dict] = None
    agent_transfer: Optional[AgentTransfer] = None
    base64_images: Optional[list] = None
    model: str

    def get_text(self) -> str:
        text = ""
        if "claude" in self.model:
            contents = self.tool_result_message.get("content", [])
            for content in contents:
                text += f"{content.get('content', '')}\n"
        else:
            for message in self.tool_messages:
                if message.get("content", ""):
                    text += f"{message.get('text', '')}\n"

        return text.strip()

    def to_message_create(self) -> MessageCreate:
        if "claude" in self.model:
            author = Author(role="user")
            # tool_result_message = {"role": "user", "content": tool_results}

            content = Content(type=ContentType.tool_result, content=self.tool_result_message["content"])
            metadata = Metadata(model=self.model)
            return MessageCreate(author=author, content=content, metadata=metadata)
        else:
            author = Author(role="tool")
            content = Content(type=ContentType.tool_message, content=self.tool_messages)
            metadata = Metadata(model=self.model)
            return MessageCreate(author=author, content=content, metadata=metadata)
