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

    async def create(self, assistant_id: str, memory: AssistantMemoryCreate) -> AssistantMemory:
        response = await self._http.request("POST", f"/assistants/{assistant_id}/memories", data=memory.model_dump())
        return AssistantMemory(**response)

    async def get_memory(self, assistant_id: str, memory_id: str) -> Optional[AssistantMemory]:
        """Get memory by ID"""
        response = await self._http.request("GET", f"/assistants/{assistant_id}/memories/{memory_id}")
        if response:
            return AssistantMemory(**response)
        return None

    async def get_memories(self, assistant_id: str, skip: int = 0, limit: int = 100) -> list[AssistantMemory]:
        """Get memories for an assistant in desc order by updated_at"""
        response = await self._http.request("GET", f"/assistants/{assistant_id}/memories?skip={skip}&limit={limit}")
        if not response:
            return []
        return [AssistantMemory(**memory) for memory in response]

    async def update_memory(
        self,
        assistant_id: str,
        memory_id: str,
        memory: AssistantMemoryUpdate,
    ) -> AssistantMemory:
        """Update memory"""
        response = await self._http.request(
            "PUT", f"/assistants/{assistant_id}/memories/{memory_id}", memory.model_dump()
        )
        return AssistantMemory(**response)

    async def delete_memories(
        self,
        assistant_id: str,
        memory_ids: list[str],
    ) -> dict[str, Any]:
        """Bulk delete multiple memories for an assistant."""
        response = await self._http.request(
            "DELETE", f"/assistants/{assistant_id}/memories", data={"memory_ids": memory_ids}
        )
        return response

    async def search_memories(
        self,
        assistant_id: str,
        query: str,
        limit: int = 5,
    ) -> RelevantMemoriesResponse:
        """Search memories by query"""
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
