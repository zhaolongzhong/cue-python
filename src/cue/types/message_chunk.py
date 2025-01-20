from typing import Optional
from datetime import datetime

from pydantic import Field, BaseModel, ConfigDict, computed_field


class MessageChunk(BaseModel):
    """Model for stream chunks of a message.

    Used when receiving streaming responses from AI providers.
    """

    id: Optional[str] = Field(None, description="Chunk identifier")
    content: Optional[str] = Field(None, description="Chunk content")
    created_at: Optional[datetime] = Field(None, description="Timestamp of chunk creation")

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
