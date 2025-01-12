from enum import Enum
from typing import Union, Optional

from pydantic import Field, BaseModel, ConfigDict


class EventMessageType(str, Enum):
    # User interaction events
    USER = "user"
    ASSISTANT = "assistant"

    # System events
    GENERIC = "generic"
    ERROR = "error"

    # Connection events
    CLIENT_CONNECT = "client_connect"
    CLIENT_DISCONNECT = "client_disconnect"
    CLIENT_STATUS = "client_status"
    PING = "ping"
    PONG = "pong"

    # Control events
    CONTROL = "control"  # Generic control event
    CONTROL_ACK = "control_ack"  # Acknowledgment for control events


class MessageMetadata(BaseModel):
    """Standard metadata structure for messages"""
    timestamp: str = Field(..., description="ISO format timestamp")
    sequence: Optional[int] = Field(None, description="Message sequence number")
    source: Optional[str] = Field(None, description="Message source identifier")
    correlation_id: Optional[str] = Field(None, description="Correlation ID for related messages")
    tags: list[str] = Field(default_factory=list, description="Message tags")
    custom: Optional[dict] = Field(None, description="Custom metadata fields")

    model_config = ConfigDict(frozen=True)

class MessagePayloadBase(BaseModel):
    """Base class for all message payloads"""
    message: Optional[str] = Field(None, description="Message content")
    sender: Optional[str] = Field(None, description="Sender identifier")
    recipient: Optional[str] = Field(None, description="Recipient identifier")
    websocket_request_id: Optional[str] = Field(None, description="Request tracking ID")
    metadata: Optional[MessageMetadata] = Field(None, description="Structured metadata")
    payload: Optional[dict] = Field(None, description="Additional payload data")
    version: str = Field("1.0", description="Schema version")

    model_config = ConfigDict(frozen=True)


class ErrorPayload(MessagePayloadBase):
    """Payload for error messages"""
    error_code: str = Field(..., description="Error code")
    error_message: str = Field(..., description="Error description")
    severity: str = Field("error", description="Error severity (warning/error/critical)")
    details: Optional[dict] = Field(None, description="Additional error details")


class ControlPayload(MessagePayloadBase):
    """Payload for control messages"""
    control_type: str = Field(..., description="Type of control operation")
    operation: str = Field(..., description="Control operation name")
    parameters: Optional[dict] = Field(None, description="Operation parameters")
    requires_ack: bool = Field(False, description="Whether acknowledgment is required")


class ControlAckPayload(MessagePayloadBase):
    """Payload for control acknowledgments"""
    control_type: str = Field(..., description="Type of control operation")
    operation: str = Field(..., description="Control operation name")
    status: str = Field(..., description="Operation status (success/failure)")
    details: Optional[dict] = Field(None, description="Operation details")


class MessagePayload(MessagePayloadBase):
    """Payload for user/assistant messages"""
    user_id: Optional[str] = Field(None, description="User identifier")
    msg_id: Optional[str] = Field(None, description="Message identifier")
    in_reply_to: Optional[str] = Field(None, description="Reference to previous message")
    content_type: str = Field("text", description="Content type (text/markdown/html)")


class ClientEventPayload(MessagePayloadBase):
    """Payload for client events"""
    client_id: str = Field(..., description="Client identifier")
    user_id: Optional[str] = Field(None, description="User identifier")
    client_type: str = Field("unknown", description="Type of client")
    capabilities: list[str] = Field(default_factory=list, description="Client capabilities")


class ConnectionPayload(MessagePayloadBase):
    """Payload for connection-related events"""
    type: str = Field(..., description="Connection event type")
    client_info: Optional[dict] = Field(None, description="Client information")


EventPayload = Union[
    MessagePayload,
    ErrorPayload,
    ControlPayload,
    ControlAckPayload,
    ClientEventPayload,
    ConnectionPayload,
]


class EventMessage(BaseModel):
    """WebSocket event message"""
    type: EventMessageType = Field(..., description="Type of event")
    payload: EventPayload = Field(..., description="Event payload")
    client_id: Optional[str] = Field(None, description="Client identifier")
    metadata: Optional[MessageMetadata] = Field(None, description="Event metadata")
    websocket_request_id: Optional[str] = Field(None, description="Request tracking ID")
    version: str = Field("1.0", description="Message schema version")

    model_config = ConfigDict(frozen=True)
