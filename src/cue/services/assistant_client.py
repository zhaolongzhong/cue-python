import logging
from typing import Dict, List, Union, Optional

from cue.schemas.completion_response import CompletionResponse

from ..schemas import (
    Assistant,
    AssistantCreate,
    AssistantUpdate,
    AssistantMetadata,
)
from .transport import HTTPTransport, ResourceClient, WebSocketTransport

logger = logging.getLogger(__name__)


class AssistantClient(ResourceClient):
    """Client for assistant-related operations"""

    def __init__(self, http: HTTPTransport, ws: Optional[WebSocketTransport] = None):
        super().__init__(http, ws)
        self._default_assistant_id: Optional[str] = None

    async def create(self, assistant: AssistantCreate) -> Assistant:
        response = await self._http.request("POST", "/assistants", data=assistant.model_dump())
        if not response:
            logger.error("Create assistant failed")
            return
        return Assistant(**response)

    async def get(self, assistant_id: str) -> Assistant:
        response = await self._http.request("GET", f"/assistants/{assistant_id}")
        return Assistant(**response)

    async def get_project_context(self) -> Optional[Union[dict, str]]:
        response = await self._http.request("GET", f"/assistants/{self._default_assistant_id}")
        asssistant = Assistant(**response)
        return asssistant.metadata.context if asssistant.metadata else None

    async def get_system_context(self) -> Optional[Union[dict, str]]:
        response = await self._http.request("GET", f"/assistants/{self._default_assistant_id}")
        asssistant = Assistant(**response)
        system_context = ""
        if asssistant.metadata.instruction:
            system_context = f"<user_set_context>{asssistant.metadata.instruction}</user_set_context>"
        if asssistant.metadata.system:
            system_context += f"<model_set_context>{asssistant.metadata.system}</model_set_context>"
        return system_context

    async def update(self, assistant: AssistantUpdate) -> Assistant:
        response = await self._http.request(
            "PUT", f"/assistants/{self._default_assistant_id}", data=assistant.model_dump()
        )
        return Assistant(**response)

    async def list(self, skip: int = 0, limit: int = 100) -> List[Assistant]:
        response = await self._http.request("GET", f"/assistants?skip={skip}&limit={limit}")
        return [Assistant(**asst) for asst in response]

    async def delete(self, assistant_id: str) -> None:
        await self._http.request("DELETE", f"/assistants/{assistant_id}")

    async def create_default_assistant(self, name: Optional[str] = "default") -> Optional[str]:
        """
        Create a default assistant to persist memories across multiple conversation
        """
        assistant = await self.create(AssistantCreate(name=name, metadata=AssistantMetadata(is_primary=True)))
        if not assistant:
            return
        self._default_assistant_id = assistant.id
        return self._default_assistant_id

    async def chat(self, chat_request: Dict) -> CompletionResponse:
        response = await self._http.request("POST", "/assistants/chat", json=chat_request)
        if not response:
            logger.error("Chat request failed")
            return
        return CompletionResponse(**response)
