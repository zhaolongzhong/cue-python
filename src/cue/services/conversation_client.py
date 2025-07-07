import logging
import builtins
from typing import Optional

from ..schemas import (
    Conversation,
    ConversationCreate,
    ConversationUpdate,
)
from .transport import HTTPTransport, ResourceClient, WebSocketTransport

logger = logging.getLogger(__name__)


class ConversationClient(ResourceClient):
    """Client for conversation-related operations"""

    def __init__(self, http: HTTPTransport, ws: Optional[WebSocketTransport] = None):
        super().__init__(http, ws)

    async def create(
        self,
        title: Optional[str] = None,
        assistant_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> Optional[Conversation]:
        data = ConversationCreate(title=title, assistant_id=assistant_id, metadata=metadata).model_dump()
        response = await self._http.request("POST", "/conversations", data=data)
        if not response:
            return None
        return Conversation(**response)

    async def get(self, conversation_id: str) -> Conversation:
        response = await self._http.request("GET", f"/conversations/{conversation_id}")
        return Conversation(**response)

    async def list(self, skip: int = 0, limit: int = 50) -> list[Conversation]:
        response = await self._http.request("GET", f"/conversations?skip={skip}&limit={limit}")
        return [Conversation(**conv) for conv in response]

    async def update(self, conversation_id: str, update_data: ConversationUpdate) -> Conversation:
        response = await self._http.request("PUT", f"/conversations/{conversation_id}", data=update_data.model_dump())
        return Conversation(**response)

    async def delete(self, conversation_id: str) -> None:
        await self._http.request("DELETE", f"/conversations/{conversation_id}")

    async def create_default_conversation(self, assistant_id: Optional[str] = None) -> Optional[Conversation]:
        """
        Create a default conversation
        If assistant_id is provided, try to use it to get conversation first.
        """
        if assistant_id:
            conversations = await self.get_conversation_by_assistant_id(assistant_id)
            for conversation in conversations:
                if conversation.metadata and conversation.metadata.is_primary:
                    return conversation
        conversation = await self.create(title="Default", metadata={"is_primary": True}, assistant_id=assistant_id)
        if not conversation:
            return
        return conversation

    async def get_conversation_by_assistant_id(
        self, assistant_id: str, skip: int = 0, limit: int = 50
    ) -> builtins.list[Conversation]:
        response = await self._http.request(
            "GET", f"/assistants/{assistant_id}/conversations?skip={skip}&limit={limit}"
        )
        if response is None:
            return []
        return [Conversation(**conv) for conv in response]

    async def broadcast_message(
        self,
        conversation_id: str,
        message_text: str,
        role: str,
        author: dict,
        model: Optional[str] = None,
        msg_id: Optional[str] = None,
        payload: Optional[dict] = None,
        websocket_request_id: Optional[str] = None,
    ) -> Optional[dict]:
        """Send message to the lightweight broadcast endpoint for persistence + broadcasting."""
        import uuid
        
        request_data = {
            "content": {
                "type": "text",
                "texts": [message_text]
            },
            "author": author,
            "conversation_id": conversation_id,
            "websocket_request_id": websocket_request_id or str(uuid.uuid4()),
        }
        
        if model:
            request_data["model"] = model
        if msg_id:
            request_data["msg_id"] = msg_id
        if payload:
            request_data["payload"] = payload

        try:
            response = await self._http.request(
                "POST", 
                f"/conversation/{conversation_id}/broadcast", 
                data=request_data
            )
            logger.debug(f"Message broadcast successfully: {message_text[:50]}...")
            return response
        except Exception as e:
            logger.error(f"Failed to broadcast message via HTTP: {e}")
            return None
