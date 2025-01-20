import logging
from typing import Any, Optional

from pydantic import TypeAdapter

from ..schemas import (
    AssistantMemory,
    AssistantMemoryCreate,
    AssistantMemoryUpdate,
    RelevantMemoriesResponse,
)
from .transport import HTTPTransport, ResourceClient, WebSocketTransport

logger = logging.getLogger(__name__)


class MemoryClient(ResourceClient):
    """Client for memory-related operations"""

    def __init__(self, http: HTTPTransport, ws: Optional[WebSocketTransport] = None):
        super().__init__(http, ws)
        self._default_assistant_id: Optional[str] = None

    def set_default_assistant_id(self, assistant_id):
        self._default_assistant_id = assistant_id

    async def get_safe_assistant_id(self, assistant_id: Optional[str] = None) -> str:
        """If no assistant id provided, use default assistant"""
        if not assistant_id:
            if not self._default_assistant_id:
                raise Exception("No default assistant id provided.")
            assistant_id = self._default_assistant_id
        return assistant_id

    async def create(self, memory: AssistantMemoryCreate, assistant_id: Optional[str] = None) -> AssistantMemory:
        assistant_id = await self.get_safe_assistant_id(assistant_id)
        response = await self._http.request("POST", f"/assistants/{assistant_id}/memories", data=memory.model_dump())
        return AssistantMemory(**response)

    async def get_memory(self, memory_id: str, assistant_id: Optional[str] = None) -> Optional[AssistantMemory]:
        """Get memory by ID"""
        assistant_id = await self.get_safe_assistant_id(assistant_id)
        response = await self._http.request("GET", f"/assistants/{assistant_id}/memories/{memory_id}")
        if response:
            return AssistantMemory(**response)
        return None

    async def get_memories(
        self, assistant_id: Optional[str] = None, skip: int = 0, limit: int = 100
    ) -> list[AssistantMemory]:
        """Get memories for an assistant in desc order by updated_at"""
        assistant_id = await self.get_safe_assistant_id(assistant_id)
        response = await self._http.request("GET", f"/assistants/{assistant_id}/memories?skip={skip}&limit={limit}")
        if not response:
            return []
        return [AssistantMemory(**memory) for memory in response]

    async def update_memory(
        self,
        memory_id: str,
        memory: AssistantMemoryUpdate,
        assistant_id: Optional[str] = None,
    ) -> AssistantMemory:
        """Update memory"""
        assistant_id = await self.get_safe_assistant_id(assistant_id)
        response = await self._http.request(
            "PUT", f"/assistants/{assistant_id}/memories/{memory_id}", memory.model_dump()
        )
        return AssistantMemory(**response)

    async def delete_memories(self, memory_ids: list[str], assistant_id: Optional[str] = None) -> dict[str, Any]:
        """Bulk delete multiple memories for an assistant."""
        assistant_id = await self.get_safe_assistant_id(assistant_id)
        response = await self._http.request(
            "DELETE", f"/assistants/{assistant_id}/memories", data={"memory_ids": memory_ids}
        )
        return response

    async def search_memories(
        self, query: str, limit: int = 5, assistant_id: Optional[str] = None
    ) -> RelevantMemoriesResponse:
        """Search memories by query"""
        assistant_id = await self.get_safe_assistant_id(assistant_id)
        validated_limit = min(max(limit, 1), 20)

        params = (
            {
                "query": query,
                "limit": validated_limit,
            },
        )
        self.memories_adapter = TypeAdapter(RelevantMemoriesResponse)
        try:
            response = await self._http.request(
                method="GET", endpoint=f"/assistants/{assistant_id}/memories", params=params
            )

            return self.memories_adapter.validate_python(response)

        except ValueError as e:
            raise ValueError(f"Failed to parse memory search response: {e}")
        except Exception as e:
            raise Exception(f"Memory search failed: {e}")
