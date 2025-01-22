from unittest.mock import Mock, AsyncMock

import pytest

from cue.schemas import AssistantCreate
from cue.services import AssistantClient
from cue.schemas.assistant import Assistant, AssistantUpdate, AssistantMetadata
from cue.services.transport import HTTPTransport


@pytest.fixture
def mock_timestamps():
    return {"created_at": "2025-12-31T00:00:00Z", "updated_at": "2025-12-31T00:00:00Z"}


@pytest.fixture
def base_assistant_data(mock_timestamps):
    return {"id": "asst_123", "name": "Test Assistant", **mock_timestamps}


@pytest.fixture
def mock_http_transport():
    transport = Mock(spec=HTTPTransport)
    transport.request = AsyncMock()
    return transport


@pytest.fixture
def client(mock_http_transport):
    return AssistantClient(http=mock_http_transport)


def create_mock_response(base_data, **kwargs):
    """Helper function to create mock response data"""
    response_data = base_data.copy()
    for key, value in kwargs.items():
        if key == "metadata":
            response_data[key] = value
        else:
            response_data[key] = value
    return response_data


@pytest.mark.asyncio
async def test_create_assistant(mock_http_transport, base_assistant_data):
    mock_data = create_mock_response(base_assistant_data, metadata={"is_primary": True})
    mock_http_transport.request.return_value = mock_data
    client = AssistantClient(http=mock_http_transport)

    assistant_create = AssistantCreate(name="Test Assistant", metadata=AssistantMetadata(is_primary=True))

    result = await client.create(assistant_create)

    mock_http_transport.request.assert_called_once_with("POST", "/assistants", data=assistant_create.model_dump())
    assert isinstance(result, Assistant)
    assert result.id == "asst_123"
    assert result.name == "Test Assistant"
    assert result.metadata.is_primary is True


@pytest.mark.asyncio
async def test_get_assistant(mock_http_transport, base_assistant_data):
    mock_data = create_mock_response(base_assistant_data, metadata={"is_primary": True})
    mock_http_transport.request.return_value = mock_data

    client = AssistantClient(http=mock_http_transport)

    result = await client.get("asst_123")

    mock_http_transport.request.assert_called_once_with("GET", "/assistants/asst_123")
    assert isinstance(result, Assistant)
    assert result.id == "asst_123"


@pytest.mark.asyncio
async def test_get_project_context(mock_http_transport, base_assistant_data):
    mock_data = create_mock_response(base_assistant_data, metadata={"context": {"project": "test_project"}})
    mock_http_transport.request.return_value = mock_data

    client = AssistantClient(http=mock_http_transport)
    result = await client.get_project_context(assistant_id="asst_123")
    assert result == {"project": "test_project"}


@pytest.mark.asyncio
async def test_get_system_context(mock_http_transport, base_assistant_data):
    mock_data = create_mock_response(
        base_assistant_data, metadata={"instruction": "test instruction", "system": "test system"}
    )
    mock_http_transport.request.return_value = mock_data
    client = AssistantClient(http=mock_http_transport)
    result = await client.get_system_context(assistant_id="asst_123")

    assert (
        result
        == "<user_set_context>test instruction</user_set_context><model_set_context>test system</model_set_context>"
    )


@pytest.mark.asyncio
async def test_update_assistant(mock_http_transport, base_assistant_data):
    mock_data = create_mock_response(base_assistant_data, name="Updated Assistant", metadata={"is_primary": True})
    mock_http_transport.request.return_value = mock_data

    client = AssistantClient(http=mock_http_transport)
    assistant_update = AssistantUpdate(name="Updated Assistant")
    result = await client.update(assistant_id="asst_123", assistant=assistant_update)

    mock_http_transport.request.assert_called_once_with(
        "PUT", "/assistants/asst_123", data=assistant_update.model_dump()
    )
    assert isinstance(result, Assistant)
    assert result.name == "Updated Assistant"


@pytest.mark.asyncio
async def test_list_assistants(mock_http_transport, base_assistant_data):
    mock_data = [
        create_mock_response(base_assistant_data, metadata={"is_primary": True}),
        create_mock_response(
            {**base_assistant_data, "id": "asst_456", "name": "Assistant 2"}, metadata={"is_primary": False}
        ),
    ]
    mock_http_transport.request.return_value = mock_data
    client = AssistantClient(http=mock_http_transport)

    result = await client.list(skip=0, limit=10)

    mock_http_transport.request.assert_called_once_with("GET", "/assistants?skip=0&limit=10")
    assert len(result) == 2
    assert all(isinstance(asst, Assistant) for asst in result)
    assert result[0].id == "asst_123"
    assert result[1].id == "asst_456"


@pytest.mark.asyncio
async def test_delete_assistant(mock_http_transport):
    mock_http_transport.request.return_value = None
    client = AssistantClient(http=mock_http_transport)

    await client.delete("asst_123")

    mock_http_transport.request.assert_called_once_with("DELETE", "/assistants/asst_123")


@pytest.mark.asyncio
async def test_create_assistant_failure(mock_http_transport):
    mock_http_transport.request.return_value = None
    client = AssistantClient(http=mock_http_transport)

    assistant_create = AssistantCreate(name="Test Assistant", metadata=AssistantMetadata(is_primary=True))

    result = await client.create(assistant_create)

    assert result is None
    mock_http_transport.request.assert_called_once_with("POST", "/assistants", data=assistant_create.model_dump())
