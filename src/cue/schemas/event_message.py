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
    CONTROL = "control"  # For agent control commands (stop, max_turns, etc)


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


class ControlMessagePayload(MessagePayloadBase):
    """Payload for agent control messages"""
    command: str = Field(..., description="Control command (e.g. 'stop', 'increase_turns')")
    args: Optional[dict] = Field(default=None, description="Optional command arguments")


EventPayload = Union[
    ClientEventPayload,
    PingPongEventPayload,
    MessagePayload,
    GenericMessagePayload,
    ControlMessagePayload,
]


class EventMessage(BaseModel):
    type: EventMessageType = Field(..., description="Type of event")
    payload: EventPayload
    client_id: Optional[str] = None
    metadata: Optional[dict] = Field(None, description="Metadata related to the event")
    websocket_request_id: Optional[str] = None
