from enum import Enum
from typing import Any, Dict, List, Union, Optional
from datetime import datetime

from pydantic import Field, BaseModel, ConfigDict, computed_field


class AuthorRole(str, Enum):
    user = "user"
    assistant = "assistant"
    system = "system"
    tool = "tool"


class Author(BaseModel):
    """Represents the author of a message.

    Attributes:
        role: The role of the author (e.g., "user", "assistant", "system", "tool")
        name: Optional name of the author
        metadata: Optional provider-specific author metadata
    """

    role: AuthorRole = Field(..., description="Role of the message author")
    name: Optional[str] = Field(None, description="Name of the author")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional author metadata")


class ContentType(str, Enum):
    text = "text"
    code = "code"
    tool_message = "tool_message"
    tool_calls = "tool_calls"
    tool_result = "tool_result"
    tool_use = "tool_use"


class Content(BaseModel):
    """Represents the content of a message.

    The content can be:
    - A simple string
    - A list of content blocks (e.g., text + images)
    - A dictionary (e.g., tool calls, structured responses)
    """

    type: Optional[ContentType] = None

    content: Union[str, List[Dict[str, Any]], Dict[str, Any]] = Field(
        ...,
        description="Message content in various formats",
        alias="text",
    )

    language: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = Field(None, description="tool calls")

    def get_text(self) -> str:
        if isinstance(self.content, str):
            return self.content
        elif isinstance(self.content, dict):
            return str(self.content)
        elif isinstance(self.content, list):
            return str(self.content)
        else:
            raise Exception("Unexpected content type")


class Metadata(BaseModel):
    """Message metadata including model information and original payload.

    Attributes:
        model: The model used for generation (e.g., "gpt-4", "claude-3")
        payload: The original message from the provider (OpenAI, Anthropic, etc.)
    """

    model: Optional[str] = Field(None, description="Model identifier")
    payload: Optional[Any] = Field(None, description="Original provider message")


class MessageBase(BaseModel):
    """Base message model with core attributes.

    All message-related models inherit from this base class.
    """

    id: Optional[str] = Field(None, description="ID of the this message")
    conversation_id: Optional[str] = Field(None, description="ID of the conversation this message belongs to")
    author: Author = Field(..., description="Author of the message")
    content: Content = Field(..., description="Content of the message")
    metadata: Optional[Metadata] = Field(None, description="Message metadata")


class MessageCreate(MessageBase):
    """Model for creating new messages."""

    pass


class MessageUpdate(MessageBase):
    """Model for updating existing messages.

    All fields are optional since updates might be partial.
    """

    conversation_id: Optional[str] = None
    author: Optional[Author] = None
    content: Optional[Content] = None
    metadata: Optional[Metadata] = None


class Message(MessageBase):
    """Complete message model including database fields.

    Extends MessageBase with id and timestamp fields.
    """

    id: str = Field(..., description="Unique message identifier")
    created_at: datetime = Field(..., description="Timestamp of message creation")
    updated_at: datetime = Field(..., description="Timestamp of last update")

    model_config = ConfigDict(from_attributes=True)

    @computed_field
    @property
    def created_at_iso(self) -> str:
        return self.created_at.isoformat()

    @computed_field
    @property
    def updated_at_iso(self) -> str:
        """ISO formatted update timestamp."""
        return self.updated_at.isoformat()


class MessageChunk(BaseModel):
    """Model for stream chunks of a message.

    Used when receiving streaming responses from AI providers.
    """

    id: str = Field(..., description="Chunk identifier")
    content: Optional[str] = Field(None, description="Chunk content")
    created_at: datetime = Field(..., description="Timestamp of chunk creation")
    updated_at: datetime = Field(..., description="Timestamp of last chunk update")

    model_config = ConfigDict(from_attributes=True)

    @computed_field
    @property
    def created_at_iso(self) -> str:
        return self.created_at.isoformat()

    @computed_field
    @property
    def updated_at_iso(self) -> str:
        """ISO formatted update timestamp."""
        return self.updated_at.isoformat()
