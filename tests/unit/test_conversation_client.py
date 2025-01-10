from unittest.mock import Mock, AsyncMock

import pytest

from cue.schemas import ConversationCreate, ConversationUpdate
from cue.services import ConversationClient
from cue.services.transport import HTTPTransport
from cue.schemas.conversation import Conversation

pytestmark = pytest.mark.unit


@pytest.fixture
def mock_timestamps():
    return {"created_at": "2025-12-31T00:00:00Z", "updated_at": "2025-12-31T00:00:00Z"}


@pytest.fixture
def base_conversation_data(mock_timestamps):
    return {"id": "conv_123", "title": "Test Conversation", "assistant_id": "asst_123", **mock_timestamps}


@pytest.fixture
def mock_http_transport():
    transport = Mock(spec=HTTPTransport)
    transport.request = AsyncMock()
    return transport


@pytest.fixture
def client(mock_http_transport):
    return ConversationClient(http=mock_http_transport)


def create_mock_response(base_data, **kwargs):
    """Helper function to create mock response data"""
    response_data = base_data.copy()
    for key, value in kwargs.items():
        response_data[key] = value
    return response_data


@pytest.mark.asyncio
async def test_create_conversation(mock_http_transport, base_conversation_data):
    mock_data = create_mock_response(base_conversation_data, metadata={"is_primary": True})
    mock_http_transport.request.return_value = mock_data
    client = ConversationClient(http=mock_http_transport)

    result = await client.create(title="Test Conversation", assistant_id="asst_123", metadata={"is_primary": True})

    expected_data = ConversationCreate(
        title="Test Conversation", assistant_id="asst_123", metadata={"is_primary": True}
    ).model_dump()

    mock_http_transport.request.assert_called_once_with("POST", "/conversations", data=expected_data)
    assert isinstance(result, Conversation)
    assert result.id == "conv_123"
    assert result.title == "Test Conversation"
    assert result.metadata.is_primary is True


@pytest.mark.asyncio
async def test_create_conversation_failure(mock_http_transport):
    mock_http_transport.request.return_value = None
    client = ConversationClient(http=mock_http_transport)

    result = await client.create(title="Test Conversation")

    mock_http_transport.request.assert_called_once()
    assert result is None


@pytest.mark.asyncio
async def test_get_conversation(mock_http_transport, base_conversation_data):
    mock_data = create_mock_response(base_conversation_data)
    mock_http_transport.request.return_value = mock_data
    client = ConversationClient(http=mock_http_transport)

    result = await client.get("conv_123")

    mock_http_transport.request.assert_called_once_with("GET", "/conversations/conv_123")
    assert isinstance(result, Conversation)
    assert result.id == "conv_123"


@pytest.mark.asyncio
async def test_list_conversations(mock_http_transport, base_conversation_data):
    mock_data = [
        create_mock_response(base_conversation_data),
        create_mock_response({**base_conversation_data, "id": "conv_456", "title": "Conversation 2"}),
    ]
    mock_http_transport.request.return_value = mock_data
    client = ConversationClient(http=mock_http_transport)

    result = await client.list(skip=0, limit=10)

    mock_http_transport.request.assert_called_once_with("GET", "/conversations?skip=0&limit=10")
    assert len(result) == 2
    assert all(isinstance(conv, Conversation) for conv in result)
    assert result[0].id == "conv_123"
    assert result[1].id == "conv_456"


@pytest.mark.asyncio
async def test_update_conversation(mock_http_transport, base_conversation_data):
    updated_title = "Updated Conversation"
    mock_data = create_mock_response(base_conversation_data, title=updated_title)
    mock_http_transport.request.return_value = mock_data
    client = ConversationClient(http=mock_http_transport)

    update_data = ConversationUpdate(title=updated_title)
    result = await client.update("conv_123", update_data)

    mock_http_transport.request.assert_called_once_with("PUT", "/conversations/conv_123", data=update_data.model_dump())
    assert isinstance(result, Conversation)
    assert result.title == updated_title


@pytest.mark.asyncio
async def test_delete_conversation(mock_http_transport):
    mock_http_transport.request.return_value = None
    client = ConversationClient(http=mock_http_transport)

    await client.delete("conv_123")

    mock_http_transport.request.assert_called_once_with("DELETE", "/conversations/conv_123")


@pytest.mark.asyncio
async def test_create_default_conversation_without_assistant(mock_http_transport, base_conversation_data):
    mock_data = create_mock_response(base_conversation_data, title="Default", metadata={"is_primary": True})
    mock_http_transport.request.return_value = mock_data
    client = ConversationClient(http=mock_http_transport)

    result = await client.create_default_conversation()

    expected_data = ConversationCreate(title="Default", metadata={"is_primary": True}).model_dump()

    mock_http_transport.request.assert_called_once_with("POST", "/conversations", data=expected_data)
    assert result == "conv_123"
    assert client._default_conversation_id == "conv_123"


@pytest.mark.asyncio
async def test_create_default_conversation_with_existing_primary(mock_http_transport, base_conversation_data):
    # First, mock the get_conversation_by_assistant_id response
    primary_conv = create_mock_response(base_conversation_data, metadata={"is_primary": True})
    mock_http_transport.request.return_value = [primary_conv]
    client = ConversationClient(http=mock_http_transport)

    result = await client.create_default_conversation(assistant_id="asst_123")

    mock_http_transport.request.assert_called_once_with("GET", "/assistants/asst_123/conversations?skip=0&limit=50")
    assert result == "conv_123"


@pytest.mark.asyncio
async def test_get_conversation_by_assistant_id(mock_http_transport, base_conversation_data):
    mock_data = [
        create_mock_response(base_conversation_data),
        create_mock_response({**base_conversation_data, "id": "conv_456", "title": "Conversation 2"}),
    ]
    mock_http_transport.request.return_value = mock_data
    client = ConversationClient(http=mock_http_transport)

    result = await client.get_conversation_by_assistant_id("asst_123", skip=0, limit=10)

    mock_http_transport.request.assert_called_once_with("GET", "/assistants/asst_123/conversations?skip=0&limit=10")
    assert len(result) == 2
    assert all(isinstance(conv, Conversation) for conv in result)
