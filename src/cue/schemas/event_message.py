from enum import Enum
from typing import Union, Optional

from pydantic import Field, BaseModel, ConfigDict


class EventMessageType(str, Enum):
    GENERIC = "generic"
    USER = "user"
    ASSISTANT = "assistant"
    ASSISTANT_TO_ASSISTANT = "assistant_to_assistant"  # Direct assistant communication
    CLIENT_CONNECT = "client_connect"
    CLIENT_DISCONNECT = "client_disconnect"
    CLIENT_STATUS = "client_status"
    PING = "ping"
    PONG = "pong"
    ERROR = "error"


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


class AssistantCommunicationPayload(MessagePayloadBase):
    source_assistant_id: str = Field(..., description="ID of the sending assistant")
    target_assistant_id: str = Field(..., description="ID of the target assistant")
    conversation_id: Optional[str] = Field(None, description="Primary conversation ID if exists")
    message_type: str = Field("request", description="Type of message: request, response, or notification")
    requires_response: bool = Field(True, description="Whether this message requires a response")
    thread_id: Optional[str] = Field(None, description="For threaded communications")


EventPayload = Union[
    ClientEventPayload,
    PingPongEventPayload,
    MessagePayload,
    GenericMessagePayload,
    AssistantCommunicationPayload,
]


class EventMessage(BaseModel):
    type: EventMessageType = Field(..., description="Type of event")
    payload: EventPayload
    client_id: Optional[str] = None
    metadata: Optional[dict] = Field(None, description="Metadata related to the event")
    websocket_request_id: Optional[str] = None
