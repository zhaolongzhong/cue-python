import logging
from typing import Optional

from ..schemas import Message, MessageCreate, MessageUpdate
from .transport import HTTPTransport, ResourceClient, WebSocketTransport

logger = logging.getLogger(__name__)


class MessageClient(ResourceClient):
    """Client for message-related operations"""

    def __init__(self, http: HTTPTransport, ws: Optional[WebSocketTransport] = None):
        super().__init__(http, ws)
        self.default_conversation_id: Optional[str] = None

    def set_default_conversation_id(self, conversation_id):
        self.default_conversation_id = conversation_id

    async def create(self, message: MessageCreate) -> Message:
        if message.conversation_id is None and self.default_conversation_id:
            message.conversation_id = self.default_conversation_id
        response = await self._http.request("POST", "/messages", data=message.model_dump())
        return Message(**response)

    async def get(self, message_id: str) -> Message:
        response = await self._http.request("GET", f"/messages/{message_id}")
        return Message(**response)

    async def get_conversation_messages(
        self,
        conversation_id: Optional[str] = None,
        skip: int = 0,
        limit: int = 15,
        role: Optional[str] = None,
        content_type: Optional[str] = None,
    ) -> list[Message]:
        if not conversation_id:
            conversation_id = self.default_conversation_id
        if not conversation_id:
            raise Exception("No conversation id provided")

        params = {"skip": skip, "limit": limit}
        if role:
            params["role"] = role
        if content_type:
            params["content_type"] = content_type

        response = await self._http.request("GET", f"/conversations/{conversation_id}/messages", params=params)
        return [Message(**msg) for msg in response]

    async def update(self, message_id: str, update_data: MessageUpdate) -> Message:
        response = await self._http.request("PUT", f"/messages/{message_id}", data=update_data.model_dump())
        return Message(**response)

    async def delete(self, message_id: str) -> None:
        await self._http.request("DELETE", f"/messages/{message_id}")
