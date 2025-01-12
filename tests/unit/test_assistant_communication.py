"""Tests for assistant-to-assistant communication functionality"""
import json
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from cue.schemas import (
    AgentConfig,
    FeatureFlag,
    RunMetadata,
    EventMessage,
    EventMessageType,
    AssistantCommunicationPayload,
)
from cue.services.service_manager import ServiceManager


@pytest.mark.asyncio
async def test_send_to_assistant():
    # Setup
    run_metadata = RunMetadata(id=str(uuid.uuid4()), mode="server")
    feature_flag = FeatureFlag()
    session = MagicMock()
    session.get = AsyncMock()
    session.get.return_value.__aenter__.return_value.status = 200
    session.get.return_value.__aenter__.return_value.json = AsyncMock(return_value={"status": "ok"})

    service = ServiceManager(
        run_metadata=run_metadata,
        feature_flag=feature_flag,
        base_url="http://test",
        session=session,
        agent=AgentConfig(id="assistant_1", name="test_assistant")
    )
    service._ws_manager = MagicMock()
    service._ws_manager.send_message = AsyncMock()
    service.messages = MagicMock()
    service.messages.create = AsyncMock()
    service.messages.default_conversation_id = "conv_1"
    service.is_server_available = True

    # Test sending assistant message
    target_assistant_id = "assistant_2"
    message_content = "Test assistant message"
    metadata = {"test_key": "test_value"}

    await service.send_to_assistant(
        target_assistant_id=target_assistant_id,
        message=message_content,
        metadata=metadata,
        message_type="request",
        requires_response=True,
    )

    # Verify message was broadcast
    broadcast_call = service._ws_manager.send_message.call_args[0][0]
    broadcast_data = json.loads(broadcast_call)

    assert broadcast_data["type"] == EventMessageType.ASSISTANT_TO_ASSISTANT
    assert broadcast_data["payload"]["message"] == message_content
    assert broadcast_data["payload"]["source_assistant_id"] == "assistant_1"
    assert broadcast_data["payload"]["target_assistant_id"] == target_assistant_id
    assert broadcast_data["payload"]["message_type"] == "request"
    assert broadcast_data["payload"]["requires_response"] is True

    # Verify message was stored
    create_call = service.messages.create.call_args.kwargs
    assert create_call["content"] == message_content
    assert create_call["conversation_id"] == "conv_1"
    assert create_call["metadata"]["type"] == "assistant_to_assistant"
    assert create_call["metadata"]["source_assistant_id"] == "assistant_1"
    assert create_call["metadata"]["target_assistant_id"] == target_assistant_id
    assert create_call["metadata"]["test_key"] == "test_value"


@pytest.mark.asyncio
async def test_handle_assistant_to_assistant():
    # Setup similar to above
    run_metadata = RunMetadata(id=str(uuid.uuid4()), mode="server")
    feature_flag = FeatureFlag()
    session = MagicMock()
    session.get = AsyncMock()
    session.get.return_value.__aenter__.return_value.status = 200
    session.get.return_value.__aenter__.return_value.json = AsyncMock(return_value={"status": "ok"})

    message_handler = AsyncMock()
    service = ServiceManager(
        run_metadata=run_metadata,
        feature_flag=feature_flag,
        base_url="http://test",
        session=session,
        agent=AgentConfig(id="assistant_2", name="test_assistant"),
        on_message_received=message_handler
    )
    service.messages = MagicMock()
    service.messages.create = AsyncMock()
    service.is_server_available = True

    # Test handling incoming message
    thread_id = str(uuid.uuid4())
    message = EventMessage(
        type=EventMessageType.ASSISTANT_TO_ASSISTANT,
        payload=AssistantCommunicationPayload(
            message="Test message",
            source_assistant_id="assistant_1",
            target_assistant_id="assistant_2",
            conversation_id="conv_1",
            message_type="request",
            requires_response=True,
            thread_id=thread_id,
        ),
        websocket_request_id=str(uuid.uuid4()),
    )

    await service._handle_assistant_to_assistant(message)

    # Verify message was stored
    create_call = service.messages.create.call_args.kwargs
    assert create_call["content"] == "Test message"
    assert create_call["conversation_id"] == "conv_1"
    assert create_call["metadata"]["type"] == "assistant_to_assistant"
    assert create_call["metadata"]["source_assistant_id"] == "assistant_1"
    assert create_call["metadata"]["target_assistant_id"] == "assistant_2"
    assert create_call["metadata"]["thread_id"] == thread_id

    # Verify message handler was called
    message_handler.assert_called_once()
    handler_msg = message_handler.call_args[0][0]
    assert handler_msg == message
