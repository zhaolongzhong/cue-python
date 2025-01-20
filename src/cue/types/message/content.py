from enum import Enum
from typing import Any, Union, Optional

from pydantic import Field, BaseModel, ConfigDict


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

    content: Union[str, list[dict[str, Any]], dict[str, Any]] = Field(
        ...,
        description="Message content in various formats",
        validation_alias="text",
        serialization_alias="text",
    )

    language: Optional[str] = None
    tool_calls: Optional[list[dict[str, Any]]] = Field(None, description="tool calls")

    model_config = ConfigDict(populate_by_name=True)

    def get_text(self) -> str:
        if isinstance(self.content, str):
            return self.content
        elif isinstance(self.content, dict):
            return str(self.content)
        elif isinstance(self.content, list):
            return str(self.content)
        else:
            raise Exception("Unexpected content type")
