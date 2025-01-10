from datetime import datetime
from unittest.mock import Mock, AsyncMock

import pytest

from cue.schemas import (
    Author,
    Content,
    AuthorRole,
    MessageCreate,
    MessageUpdate,
)
from cue.services import MessageClient
from cue.services.transport import HTTPTransport

pytestmark = pytest.mark.unit


@pytest.fixture
def mock_timestamps():
    return {"created_at": datetime(2025, 12, 31), "updated_at": datetime(2025, 12, 31)}


@pytest.fixture
def base_message_data(mock_timestamps):
    return {
        "id": "msg_123",
        "conversation_id": "conv_123",
        "author": {"role": "user", "name": None, "metadata": None},
        "content": {"content": "Test message content", "tool_calls": None},
        "metadata": None,
        **mock_timestamps,
    }


@pytest.fixture
def mock_http_transport():
    transport = Mock(spec=HTTPTransport)
    transport.request = AsyncMock()
    return transport


@pytest.fixture
def client(mock_http_transport):
    client = MessageClient(http=mock_http_transport)
    client.set_default_conversation_id("conv_123")
    return client


@pytest.mark.asyncio
async def test_create_message(mock_http_transport, base_message_data):
    """Test creating a message with default conversation ID"""
    mock_http_transport.request.return_value = base_message_data
    client = MessageClient(http=mock_http_transport)
    client.set_default_conversation_id("conv_123")

    message_create = MessageCreate(author=Author(role=AuthorRole.user), content=Content(content="Test message content"))

    result = await client.create(message_create)

    # Verify the request
    message_create.conversation_id = "conv_123"  # Should be set by client
    mock_http_transport.request.assert_called_once_with("POST", "/messages", data=message_create.model_dump())

    # Verify the response
    assert result.id == "msg_123"
    assert result.conversation_id == "conv_123"
    assert result.content.get_text() == "Test message content"
    assert result.author.role == AuthorRole.user


@pytest.mark.asyncio
async def test_create_message_with_specific_conversation(mock_http_transport, base_message_data):
    """Test creating a message with specific conversation ID"""
    mock_http_transport.request.return_value = base_message_data
    client = MessageClient(http=mock_http_transport)

    message_create = MessageCreate(
        conversation_id="conv_456", author=Author(role=AuthorRole.user), content=Content(content="Test message content")
    )

    result = await client.create(message_create)

    mock_http_transport.request.assert_called_once_with("POST", "/messages", data=message_create.model_dump())
    assert result.conversation_id == "conv_123"  # From mock response


@pytest.mark.asyncio
async def test_get_message(mock_http_transport, base_message_data):
    """Test getting a message by ID"""
    mock_http_transport.request.return_value = base_message_data
    client = MessageClient(http=mock_http_transport)

    result = await client.get("msg_123")

    mock_http_transport.request.assert_called_once_with("GET", "/messages/msg_123")

    assert result.id == "msg_123"
    assert result.conversation_id == "conv_123"
    assert result.content.get_text() == "Test message content"


@pytest.mark.asyncio
async def test_get_conversation_messages(mock_http_transport, base_message_data):
    """Test getting messages for a conversation"""
    mock_data = [
        base_message_data,
        {**base_message_data, "id": "msg_456", "content": {"content": "Second message", "tool_calls": None}},
    ]
    mock_http_transport.request.return_value = mock_data
    client = MessageClient(http=mock_http_transport)
    client.set_default_conversation_id("conv_123")

    result = await client.get_conversation_messages(skip=0, limit=10)

    mock_http_transport.request.assert_called_once_with(
        "GET", "/conversations/conv_123/messages", params={"skip": 0, "limit": 10}
    )

    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0].id == "msg_123"
    assert result[1].id == "msg_456"
    assert result[1].content.get_text() == "Second message"


@pytest.mark.asyncio
async def test_get_conversation_messages_specific_conversation(mock_http_transport, base_message_data):
    """Test getting messages for a specific conversation"""
    mock_data = [base_message_data]
    mock_http_transport.request.return_value = mock_data
    client = MessageClient(http=mock_http_transport)

    result = await client.get_conversation_messages(conversation_id="conv_456")

    mock_http_transport.request.assert_called_once_with(
        "GET", "/conversations/conv_456/messages", params={"skip": 0, "limit": 15}
    )

    assert len(result) == 1
    assert result[0].id == "msg_123"


@pytest.mark.asyncio
async def test_get_conversation_messages_no_id(mock_http_transport):
    """Test getting messages without conversation ID"""
    client = MessageClient(http=mock_http_transport)

    with pytest.raises(Exception, match="No conversation id provided"):
        await client.get_conversation_messages()


@pytest.mark.asyncio
async def test_update_message(mock_http_transport, base_message_data):
    """Test updating a message"""
    updated_content = "Updated message content"
    mock_data = {**base_message_data, "content": {"content": updated_content, "tool_calls": None}}
    mock_http_transport.request.return_value = mock_data
    client = MessageClient(http=mock_http_transport)

    message_update = MessageUpdate(content=Content(content=updated_content))
    result = await client.update("msg_123", message_update)

    mock_http_transport.request.assert_called_once_with("PUT", "/messages/msg_123", data=message_update.model_dump())

    assert result.id == "msg_123"
    assert result.content.get_text() == updated_content


@pytest.mark.asyncio
async def test_delete_message(mock_http_transport):
    """Test deleting a message"""
    mock_http_transport.request.return_value = None
    client = MessageClient(http=mock_http_transport)

    await client.delete("msg_123")

    mock_http_transport.request.assert_called_once_with("DELETE", "/messages/msg_123")
