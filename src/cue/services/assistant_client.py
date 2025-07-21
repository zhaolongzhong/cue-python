import logging
from typing import Union, Optional

from ..schemas import (
    Assistant,
    AssistantCreate,
    AssistantUpdate,
)
from .transport import HTTPTransport, ResourceClient, WebSocketTransport

logger = logging.getLogger(__name__)


class AssistantClient(ResourceClient):
    """Client for assistant-related operations"""

    def __init__(self, http: HTTPTransport, ws: Optional[WebSocketTransport] = None):
        super().__init__(http, ws)

    async def create(self, assistant: AssistantCreate) -> Assistant:
        response = await self._http.request("POST", "/assistants", data=assistant.model_dump())
        if not response:
            logger.error("Create assistant failed")
            return
        return Assistant(**response)

    async def get(self, assistant_id: str) -> Assistant:
        response = await self._http.request("GET", f"/assistants/{assistant_id}")
        if not response:
            logger.warning(f"Get assistant failed {assistant_id}")
            return
        return Assistant(**response)

    async def get_project_context(self, assistant_id: str) -> Optional[Union[dict, str]]:
        assistant = await self.get(assistant_id=assistant_id)
        if not assistant:
            return
        return assistant.metadata.context if assistant.metadata else None

    async def get_system_context(self, assistant_id: str) -> Optional[Union[dict, str]]:
        assistant = await self.get(assistant_id=assistant_id)
        if not assistant:
            return
        system_context = ""
        if assistant.metadata and assistant.metadata.instruction:
            system_context = f"<user_set_context>{assistant.metadata.instruction}</user_set_context>"
        if assistant.metadata and assistant.metadata.system:
            system_context += f"<model_set_context>{assistant.metadata.system}</model_set_context>"
        return system_context

    async def update(self, assistant_id: str, assistant: AssistantUpdate) -> Assistant:
        response = await self._http.request("PUT", f"/assistants/{assistant_id}", data=assistant.model_dump())
        return Assistant(**response)

    async def list(self, skip: int = 0, limit: int = 100) -> list[Assistant]:
        response = await self._http.request("GET", f"/assistants?skip={skip}&limit={limit}")
        return [Assistant(**asst) for asst in response]

    async def delete(self, assistant_id: str) -> None:
        await self._http.request("DELETE", f"/assistants/{assistant_id}")
