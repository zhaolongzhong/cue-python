"""Tests for message broadcasting functionality"""

import json
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from cue.schemas import (
    Author,
    FeatureFlag,
    RunMetadata,
    EventMessageType,
    ToolResponseWrapper,
)
from cue.services.service_manager import ServiceManager


class MockDBMessage:
    def __init__(self, **kwargs):
        self.id = kwargs.get("id", str(uuid.uuid4()))
        self.content = kwargs.get("content", "")
        self.metadata = kwargs.get("metadata", {})
        self.payload = kwargs.get("payload", {})


class MockMessage:
    def __init__(self, **kwargs):
        self.db_message = kwargs.get("db_message")
        self.get_text = MagicMock(return_value=kwargs.get("text", ""))


@pytest.mark.asyncio
async def test_broadcast_db_message():
    # Setup
    run_metadata = RunMetadata(id=str(uuid.uuid4()), mode="server")
    feature_flag = FeatureFlag()
    session = MagicMock()
    session.get = AsyncMock()
    session.get.return_value.__aenter__.return_value.status = 200
    session.get.return_value.__aenter__.return_value.json = AsyncMock(return_value={"status": "ok"})

    service = ServiceManager(
        run_metadata=run_metadata, feature_flag=feature_flag, base_url="http://test", session=session
    )
    service._ws_manager = MagicMock()
    service._ws_manager.send_message = AsyncMock()
    service.is_server_available = True

    # Test DB message broadcasting
    db_message = MockDBMessage(content="Test message", metadata={"author": {"role": "user"}}, payload={"extra": "data"})
    message = MockMessage(db_message=db_message, text="Test message")

    await service.send_message_to_user(message)

    # Verify broadcast
    broadcast_call = service._ws_manager.send_message.call_args[0][0]
    broadcast_data = json.loads(broadcast_call)

    assert broadcast_data["type"] == EventMessageType.ASSISTANT
    assert broadcast_data["payload"]["message"] == db_message.content
    assert broadcast_data["payload"]["metadata"] == db_message.metadata
    assert broadcast_data["payload"]["payload"] == db_message.payload
    assert broadcast_data["payload"]["msg_id"] == db_message.id


@pytest.mark.asyncio
async def test_broadcast_non_db_message():
    # Setup similar to above
    run_metadata = RunMetadata(id=str(uuid.uuid4()), mode="server")
    feature_flag = FeatureFlag()
    session = MagicMock()
    session.get = AsyncMock()
    session.get.return_value.__aenter__.return_value.status = 200
    session.get.return_value.__aenter__.return_value.json = AsyncMock(return_value={"status": "ok"})

    service = ServiceManager(
        run_metadata=run_metadata, feature_flag=feature_flag, base_url="http://test", session=session
    )
    service._ws_manager = MagicMock()
    service._ws_manager.send_message = AsyncMock()
    service.is_server_available = True

    # Test non-DB message (ToolResponseWrapper)
    tool_response = ToolResponseWrapper(
        tool_name="test_tool",
        tool_result="success",
        tool_messages=[{"content": True, "text": "Test message"}],
        model="test-model",
        author=Author(role="tool", name="test_tool"),
        msg_id=str(uuid.uuid4()),
    )

    await service.send_message_to_user(tool_response)

    # Verify broadcast uses fallback logic
    broadcast_call = service._ws_manager.send_message.call_args[0][0]
    broadcast_data = json.loads(broadcast_call)

    assert broadcast_data["type"] == EventMessageType.ASSISTANT
    assert broadcast_data["payload"]["message"] == "Test message"
    assert broadcast_data["payload"]["metadata"]["author"]["role"] == "tool"
    assert broadcast_data["payload"]["metadata"]["author"]["name"] == "test_tool"
    assert broadcast_data["payload"]["metadata"]["model"] == "test-model"
