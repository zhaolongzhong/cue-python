from enum import Enum
from typing import Union, Optional

from pydantic import Field, BaseModel, ConfigDict

__all__ = [
    "EventMessageType",
    "ClientEventPayload",
    "PingPongEventPayload",
    "MessagePayload",
    "GenericMessagePayload",
    "MessageChunkEventPayload",
    "MessageEventPayload",
    "EventPayload",
    "EventMessage",
]


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
    MESSAGE = "message"
    MESSAGE_CHUNK = "message_chunk"


class MessagePayloadBase(BaseModel):
    message: Optional[str] = Field(None, description="Message content")
    sender: Optional[str] = Field(None, description="Sender identifier")
    recipient: Optional[str] = Field(None, description="Recipient identifier")
    websocket_request_id: Optional[str] = Field(None, description="Request tracking ID")
    metadata: Optional[dict] = Field(None, description="Metadata related to the message")
    payload: Optional[dict] = None
    user_id: Optional[str] = None

    model_config = ConfigDict(frozen=True)


class GenericMessagePayload(MessagePayloadBase):
    pass


class MessagePayload(MessagePayloadBase):
    msg_id: Optional[str] = None


class ClientEventPayload(MessagePayloadBase):
    client_id: str


class PingPongEventPayload(MessagePayloadBase):
    type: str


class MessageChunkEventPayload(MessagePayloadBase):
    pass


class MessageEventPayload(MessagePayloadBase):
    pass


EventPayload = Union[
    ClientEventPayload,
    PingPongEventPayload,
    MessagePayload,
    GenericMessagePayload,
]


class EventMessage(BaseModel):
    type: EventMessageType = Field(..., description="Type of event")
    payload: EventPayload
    client_id: Optional[str] = None
    metadata: Optional[dict] = Field(None, description="Metadata related to the event")
    websocket_request_id: Optional[str] = None
