from enum import Enum
from typing import Union, Optional

from pydantic import Field, BaseModel, ConfigDict


class EventMessageType(str, Enum):
    GENERIC = "generic"
    USER = "user"
    ASSISTANT = "assistant"
    CLIENT_CONNECT = "client_connect"
    CLIENT_DISCONNECT = "client_disconnect"
    CLIENT_STATUS = "client_status"
    PING = "ping"
    PONG = "pong"
    ERROR = "error"
    # Control events
    CONTROL_RESET = "control_reset"
    CONTROL_GET_STATE = "control_get_state"
    CONTROL_STATE = "control_state"
    CONTROL_PERMISSION = "control_permission"
    CONTROL_PERMISSION_RESPONSE = "control_permission_response"


class MessagePayloadBase(BaseModel):
    message: Optional[str] = Field(None, description="Message content")
    sender: Optional[str] = Field(None, description="Sender identifier")
    recipient: Optional[str] = Field(None, description="Recipient identifier")
    websocket_request_id: Optional[str] = Field(None, description="Request tracking ID")
    metadata: Optional[dict] = Field(None, description="Metadata related to the message")
    payload: Optional[dict] = None

    model_config = ConfigDict(frozen=True)


class GenericMessagePayload(MessagePayloadBase):
    user_id: Optional[str] = Field(None, description="User identifier")


class MessagePayload(MessagePayloadBase):
    user_id: Optional[str] = None
    msg_id: Optional[str] = None


class ClientEventPayload(MessagePayloadBase):
    client_id: str
    user_id: Optional[str] = None


class PingPongEventPayload(MessagePayloadBase):
    type: str


class ControlEventPayload(MessagePayloadBase):
    """Base class for control events"""
    control_type: str


class ControlResetPayload(ControlEventPayload):
    """Reset agent state"""
    control_type: str = "reset"
    reset_type: str = Field("full", description="Type of reset: 'full' or 'soft'")


class ControlStatePayload(ControlEventPayload):
    """Get/Return agent state"""
    control_type: str = "state"
    state: Optional[dict] = Field(None, description="Agent state information")


class ControlPermissionPayload(ControlEventPayload):
    """Permission request/response"""
    control_type: str = "permission"
    permission_type: str = Field(..., description="Type of permission requested")
    reason: str = Field(..., description="Reason for permission request")
    details: Optional[dict] = Field(None, description="Additional details")


class ControlPermissionResponsePayload(ControlEventPayload):
    """Permission response"""
    control_type: str = "permission_response"
    permission_type: str
    granted: bool
    reason: Optional[str] = None


EventPayload = Union[
    ClientEventPayload,
    PingPongEventPayload,
    MessagePayload,
    GenericMessagePayload,
    ControlResetPayload,
    ControlStatePayload,
    ControlPermissionPayload,
    ControlPermissionResponsePayload,
]


class EventMessage(BaseModel):
    type: EventMessageType = Field(..., description="Type of event")
    payload: EventPayload
    client_id: Optional[str] = None
    metadata: Optional[dict] = Field(None, description="Metadata related to the event")
    websocket_request_id: Optional[str] = None
