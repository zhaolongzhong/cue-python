from unittest.mock import Mock, AsyncMock

import pytest

from cue.schemas import (
    AssistantMemoryCreate,
    AssistantMemoryUpdate,
)
from cue.services import MemoryClient
from cue.services.transport import HTTPTransport


@pytest.fixture
def mock_timestamps():
    return {"created_at": "2025-12-31T00:00:00Z", "updated_at": "2025-12-31T00:00:00Z"}


@pytest.fixture
def base_memory_data(mock_timestamps):
    return {
        "id": "mem_123",
        "assistant_id": "asst_123",
        "content": "Test memory content",
        "metadata": {"type": "observation"},
        **mock_timestamps,
    }


@pytest.fixture
def mock_http_transport():
    transport = Mock(spec=HTTPTransport)
    transport.request = AsyncMock()
    return transport


@pytest.fixture
def client(mock_http_transport):
    client = MemoryClient(http=mock_http_transport)
    client.set_default_assistant_id("asst_123")
    return client


def create_mock_response(base_data, **kwargs):
    """Helper function to create mock response data"""
    response_data = base_data.copy()
    for key, value in kwargs.items():
        response_data[key] = value
    return response_data


@pytest.mark.asyncio
async def test_create_memory(mock_http_transport, base_memory_data):
    mock_http_transport.request.return_value = base_memory_data
    client = MemoryClient(http=mock_http_transport)
    client.set_default_assistant_id("asst_123")

    memory_create = AssistantMemoryCreate(content="Test memory content", metadata={"type": "observation"})

    result = await client.create(memory_create)

    # Verify the request
    mock_http_transport.request.assert_called_once_with(
        "POST", "/assistants/asst_123/memories", data=memory_create.model_dump()
    )

    # Verify the response
    assert result.id == "mem_123"
    assert result.content == "Test memory content"
    assert result.metadata["type"] == "observation"


@pytest.mark.asyncio
async def test_create_memory_with_specific_assistant(mock_http_transport, base_memory_data):
    mock_http_transport.request.return_value = base_memory_data
    client = MemoryClient(http=mock_http_transport)

    memory_create = AssistantMemoryCreate(content="Test memory content", metadata={"type": "observation"})

    await client.create(memory_create, assistant_id="asst_456")

    mock_http_transport.request.assert_called_once_with(
        "POST", "/assistants/asst_456/memories", data=memory_create.model_dump()
    )


@pytest.mark.asyncio
async def test_create_memory_no_assistant_id(mock_http_transport):
    client = MemoryClient(http=mock_http_transport)
    memory_create = AssistantMemoryCreate(content="Test content")

    with pytest.raises(Exception, match="No default assistant id provided."):
        await client.create(memory_create)


@pytest.mark.asyncio
async def test_get_memory(mock_http_transport, base_memory_data):
    mock_http_transport.request.return_value = base_memory_data
    client = MemoryClient(http=mock_http_transport)
    client.set_default_assistant_id("asst_123")

    result = await client.get_memory("mem_123")

    mock_http_transport.request.assert_called_once_with("GET", "/assistants/asst_123/memories/mem_123")

    assert result.id == "mem_123"


@pytest.mark.asyncio
async def test_get_memory_not_found(mock_http_transport):
    mock_http_transport.request.return_value = None
    client = MemoryClient(http=mock_http_transport)
    client.set_default_assistant_id("asst_123")

    result = await client.get_memory("mem_nonexistent")

    assert result is None


@pytest.mark.asyncio
async def test_get_memories(mock_http_transport, base_memory_data):
    mock_data = [base_memory_data, create_mock_response(base_memory_data, id="mem_456", content="Second memory")]
    mock_http_transport.request.return_value = mock_data
    client = MemoryClient(http=mock_http_transport)
    client.set_default_assistant_id("asst_123")

    result = await client.get_memories(skip=0, limit=10)

    mock_http_transport.request.assert_called_once_with("GET", "/assistants/asst_123/memories?skip=0&limit=10")

    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0].id == "mem_123"
    assert result[1].id == "mem_456"


@pytest.mark.asyncio
async def test_get_memories_empty_response(mock_http_transport):
    mock_http_transport.request.return_value = None
    client = MemoryClient(http=mock_http_transport)
    client.set_default_assistant_id("asst_123")

    result = await client.get_memories()

    assert isinstance(result, list)
    assert len(result) == 0


@pytest.mark.asyncio
async def test_update_memory(mock_http_transport, base_memory_data):
    updated_content = "Updated memory content"
    mock_data = create_mock_response(base_memory_data, content=updated_content)
    mock_http_transport.request.return_value = mock_data
    client = MemoryClient(http=mock_http_transport)
    client.set_default_assistant_id("asst_123")

    memory_update = AssistantMemoryUpdate(content=updated_content)
    result = await client.update_memory("mem_123", memory_update)

    mock_http_transport.request.assert_called_once_with(
        "PUT", "/assistants/asst_123/memories/mem_123", memory_update.model_dump()
    )

    assert result.content == updated_content


@pytest.mark.asyncio
async def test_delete_memories(mock_http_transport):
    mock_response = {"deleted": ["mem_123", "mem_456"]}
    mock_http_transport.request.return_value = mock_response
    client = MemoryClient(http=mock_http_transport)
    client.set_default_assistant_id("asst_123")

    memory_ids = ["mem_123", "mem_456"]
    result = await client.delete_memories(memory_ids)

    mock_http_transport.request.assert_called_once_with(
        "DELETE", "/assistants/asst_123/memories", data={"memory_ids": memory_ids}
    )

    assert result == mock_response


@pytest.mark.asyncio
async def test_search_memories(mock_http_transport):
    mock_response = {
        "memories": [
            {
                "id": "mem_123",
                "content": "Test memory content",
                "similarity_score": 0.95,
                "created_at": "2025-12-31T00:00:00Z",
                "updated_at": "2025-12-31T00:00:00Z",
            }
        ],
        "total_count": 1,
    }
    mock_http_transport.request.return_value = mock_response
    client = MemoryClient(http=mock_http_transport)
    client.set_default_assistant_id("asst_123")

    result = await client.search_memories("test query", limit=5)

    mock_http_transport.request.assert_called_once_with(
        method="GET", endpoint="/assistants/asst_123/memories", params=({"query": "test query", "limit": 5},)
    )

    assert len(result.memories) == 1
    assert result.memories[0].id == "mem_123"
    assert result.memories[0].similarity_score == 0.95
    assert result.total_count == 1


@pytest.mark.asyncio
async def test_search_memories_invalid_limit(mock_http_transport):
    mock_response = {"memories": [], "total_count": 0}
    mock_http_transport.request.return_value = mock_response
    client = MemoryClient(http=mock_http_transport)
    client.set_default_assistant_id("asst_123")

    # Test with limit below minimum
    await client.search_memories("test", limit=0)
    mock_http_transport.request.assert_called_with(
        method="GET", endpoint="/assistants/asst_123/memories", params=({"query": "test", "limit": 1},)
    )

    # Test with limit above maximum
    await client.search_memories("test", limit=25)
    mock_http_transport.request.assert_called_with(
        method="GET", endpoint="/assistants/asst_123/memories", params=({"query": "test", "limit": 20},)
    )


@pytest.mark.asyncio
async def test_search_memories_request_error(mock_http_transport):
    mock_http_transport.request.side_effect = Exception("Request failed")
    client = MemoryClient(http=mock_http_transport)
    client.set_default_assistant_id("asst_123")

    with pytest.raises(Exception, match="Memory search failed: Request failed"):
        await client.search_memories("test query")
