from typing import Any, Dict, List, Union, Optional

from pydantic import Field, BaseModel

from .message import Author, Content, Message, Metadata, ContentType, MessageCreate


class MessageParam(BaseModel):
    """
    Represents a message parameter structure for communication between components.
    Used for both in-memory operations and persistence mapping.
    """

    role: str = Field(
        ...,
        description="The role of the message author (e.g., 'user', 'assistant', 'system')",
    )
    content: Union[str, List[Dict[str, Any]], Dict[str, Any]] = Field(
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

    def to_message_create(self) -> MessageCreate:
        author = Author(role=self.role)
        content = Content(type=ContentType.text, content=self.content)
        metadata = Metadata(model=self.model)
        return MessageCreate(author=author, content=content, metadata=metadata)

    @classmethod
    def from_message(
        cls,
        message: Message,
        force_str_content: bool = False,
        truncate_length: Optional[int] = None,
        truncation_indicator: str = " ...",
        show_visibility: bool = True,
    ) -> "MessageParam":
        """
        Create a MessageParam instance from a Message.

        Args:
            message: The Message instance to convert from.
            force_str_content: If True, converts content to string format.
            truncate_length: Maximum length to truncate content to, if provided.
            truncation_indicator: String to append when content is truncated
            show_visibility: If True, includes visibility percentage in truncation indicator
        """
        role = message.author.role
        content = message.content.content
        if force_str_content:
            content = str(message.content.content)
            original_length = len(content)

            if truncate_length is not None and original_length > truncate_length:
                # Calculate visibility based on truncate_length
                visibility_percent = round((truncate_length / original_length) * 100)

                # Create indicator with visibility
                if show_visibility:
                    indicator = f"{truncation_indicator} ({visibility_percent}% visible)"
                else:
                    indicator = truncation_indicator

                # Calculate actual content length
                actual_length = truncate_length - len(indicator)
                content = content[:actual_length] + indicator

            if "tool" == message.author.role:
                role = "assistant"

        return cls(
            role=role,
            content=content,
            name=message.author.name if hasattr(message.author, "name") else None,
            msg_id=message.id,
        )
