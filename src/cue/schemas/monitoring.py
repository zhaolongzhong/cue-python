from enum import Enum
from typing import Literal, Optional
from datetime import datetime

from pydantic import Field, BaseModel, field_serializer


class ErrorType(str, Enum):
    SYSTEM = "system"
    AGENT = "agent"
    TOOL = "tool"
    LLM = "llm"
    TRANSFER = "transfer"


class ErrorReport(BaseModel):
    type: ErrorType
    message: str
    severity: Literal["info", "warning", "error", "critical"] = "error"

    conversation_id: Optional[str] = None
    assistant_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)

    metadata: Optional[dict] = Field(
        default=None, description="Additional context: tool_name, error_trace, state, etc."
    )

    @field_serializer("timestamp")
    def serialize_timestamp(self, v: datetime) -> str:
        return v.isoformat()


class ErrorReportResponse(BaseModel):
    status: str
    timestamp: datetime
