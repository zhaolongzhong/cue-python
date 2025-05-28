from enum import Enum
from typing import Union, Optional

from pydantic import Field, BaseModel, ConfigDict

from .agent_event import AgentEventPayload, AgentStatePayload, AgentControlPayload

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
    AGENT_STATE = "agent_state"
    AGENT_CONTROL = "agent_control"
    AGENT_EVENT = "agent_event"


class MessagePayloadBase(BaseModel):
    message: Optional[str] = Field(None, description="Message content")
    sender: Optional[str] = Field(None, description="Sender identifier")
    recipient: Optional[str] = Field(None, description="Recipient identifier")
    conversation_id: Optional[str] = Field(None, description="Conversation identifier")
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
    MessageChunkEventPayload,
    MessageEventPayload,
    AgentEventPayload,
    AgentControlPayload,
    AgentStatePayload,
]


class EventMessage(BaseModel):
    type: EventMessageType = Field(..., description="Type of event")
    payload: EventPayload
    client_id: Optional[str] = None
    metadata: Optional[dict] = Field(None, description="Metadata related to the event")
    websocket_request_id: Optional[str] = None

    def __init__(self, **data):
        # Custom parsing logic based on message type
        message_type = data.get("type")
        payload_data = data.get("payload", {})

        # If payload is already a payload object, use it directly
        if isinstance(
            payload_data,
            (
                ClientEventPayload,
                PingPongEventPayload,
                MessagePayload,
                GenericMessagePayload,
                MessageChunkEventPayload,
                MessageEventPayload,
                AgentEventPayload,
                AgentControlPayload,
                AgentStatePayload,
            ),
        ):
            data["payload"] = payload_data
        else:
            # Parse from dictionary based on message type
            if message_type == EventMessageType.AGENT_CONTROL:
                payload = AgentControlPayload(**payload_data)
            elif message_type == EventMessageType.AGENT_EVENT:
                payload = AgentEventPayload(**payload_data)
            elif message_type == EventMessageType.AGENT_STATE:
                payload = AgentStatePayload(**payload_data)
            elif message_type in (
                EventMessageType.CLIENT_CONNECT,
                EventMessageType.CLIENT_DISCONNECT,
                EventMessageType.CLIENT_STATUS,
            ):
                payload = ClientEventPayload(**payload_data)
            elif message_type in (EventMessageType.PING, EventMessageType.PONG):
                payload = PingPongEventPayload(**payload_data)
            elif message_type in (EventMessageType.USER, EventMessageType.ASSISTANT):
                payload = MessagePayload(**payload_data)
            elif message_type == EventMessageType.MESSAGE_CHUNK:
                payload = MessageChunkEventPayload(**payload_data)
            elif message_type == EventMessageType.MESSAGE:
                payload = MessageEventPayload(**payload_data)
            else:
                # Default to GenericMessagePayload for GENERIC and ERROR types
                payload = GenericMessagePayload(**payload_data)

            data["payload"] = payload

        super().__init__(**data)
