import logging
from uuid import uuid4
from typing import AsyncGenerator
from datetime import datetime

import pytest
import pytest_asyncio

from cue.schemas import Author, Content, RunMetadata, MessageCreate
from cue.services import MessageClient, ServiceManager, AssistantClient, ConversationClient

logger = logging.getLogger(__name__)


@pytest_asyncio.fixture
async def service_manager() -> AsyncGenerator[ServiceManager, None]:
    """Fixture to create and manage the ServiceManager instance."""
    service_manager = await ServiceManager.create(run_metadata=RunMetadata())
    await service_manager.connect()
    try:
        yield service_manager
    finally:
        await service_manager.close()


@pytest_asyncio.fixture
async def assistant_client(service_manager: ServiceManager) -> AssistantClient:
    """Fixture to provide the AssistantClient from the ServiceManager."""
    return service_manager.assistants


@pytest_asyncio.fixture
async def conversation_client(service_manager: ServiceManager) -> ConversationClient:
    """Fixture to provide the MemoryClient from the ServiceManager."""
    return service_manager.conversations


@pytest_asyncio.fixture
async def message_client(service_manager: ServiceManager) -> MessageClient:
    """Fixture to provide the MessageClient from the ServiceManager."""
    return service_manager.messages


@pytest_asyncio.fixture
async def test_conversation(conversation_client: ConversationClient):
    """Fixture to create a test conversation and yield its object."""
    conversation = await conversation_client.create(
        title=f"Test Conversation {uuid4()}", metadata={"test": True, "created_at": datetime.now().isoformat()}
    )
    try:
        yield conversation
    finally:
        try:
            await conversation_client.delete(conversation.id)
        except Exception as e:
            logger.warning(f"Failed to delete test conversation: {e}")


@pytest.mark.integration
class TestMessageClient:
    @pytest.mark.asyncio
    async def test_create_message(self, message_client: MessageClient, test_conversation) -> None:
        """Test creating a single message."""
        message_content = "Test message content"

        conversation_id = test_conversation.id

        created_message = await message_client.create(
            MessageCreate(
                conversation_id=conversation_id,
                author=Author(role="user"),
                content=Content(content=message_content),
                metadata={"model": "gpt-4o"},
            )
        )

        assert created_message is not None, "Message should not be None"
        assert created_message.content.content == message_content, "Message content should match"
        assert created_message.conversation_id == conversation_id, "Conversation ID should match"
        assert created_message.author.role == "user", "Message role should be 'user'"
        assert created_message.metadata.model == "gpt-4o", "Message metadata should contain gpt-4o"
