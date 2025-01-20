from typing import Optional
from datetime import datetime

from pydantic import Field, BaseModel, ConfigDict, computed_field

from ..types.message import Author, Content, Metadata


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
