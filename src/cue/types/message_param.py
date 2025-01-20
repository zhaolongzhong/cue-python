from typing import Any, Union, Optional

from pydantic import Field, BaseModel

__all__ = ["MessageParam"]


class MessageParam(BaseModel):
    """
    Represents a message parameter structure for communication between components.
    Used for both in-memory operations and persistence mapping.
    """

    role: str = Field(
        ...,
        description="The role of the message author (e.g., 'user', 'assistant', 'system')",
    )
    content: Union[str, list[dict[str, Any]], dict[str, Any]] = Field(
        ...,
        description="Message content in various formats (text, structured data, or tool calls)",
    )
    name: Optional[str] = Field(
        None,
        description="Optional identifier or name for the message author",
    )

    # Persistence-related fields
    model: Optional[str] = Field(
        None,
        description="Model identifier used for message generation in persistence layer",
    )
    msg_id: Optional[str] = Field(
        None,
        description="Unique identifier for the message in persistence layer",
    )

    def get_text(self) -> str:
        return str(self.content)
