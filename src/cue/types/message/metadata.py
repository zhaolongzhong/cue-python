from typing import Any, Optional

from pydantic import Field, BaseModel

__all__ = ["Metadata"]


class Metadata(BaseModel):
    """Message metadata including model information and original payload.

    Attributes:
        model: The model used for generation (e.g., "gpt-4", "claude-3")
        payload: The original message from the provider (OpenAI, Anthropic, etc.)
    """

    model: Optional[str] = Field(None, description="Model identifier")
    payload: Optional[Any] = Field(None, description="Original provider message")
