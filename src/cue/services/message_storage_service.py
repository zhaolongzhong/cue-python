import logging
from typing import Union, Optional

from ..types import MessageParam, CompletionResponse, ToolResponseWrapper
from ..schemas import Message, MessageCreate, MessageParamFactory, MessageCreateFactory
from .message_client import MessageClient

logger = logging.getLogger(__name__)


class MessageStorageService:
    def __init__(self, message_client: MessageClient):
        self.message_client = message_client

    async def persist_message(
        self, conversation_id: str, message: Union[CompletionResponse, ToolResponseWrapper, MessageParam]
    ) -> Union[CompletionResponse, ToolResponseWrapper, MessageParam]:
        if not isinstance(message, (CompletionResponse, ToolResponseWrapper, MessageParam)):
            logger.error("Unexpect message type to persist")
            return message
        if message.msg_id:
            logger.error(f"Message is already persisted with id: {message.msg_id}")
            return message

        try:
            message_create = MessageCreateFactory.create_from(message=message)
            message_create.conversation_id = conversation_id
            persisted_message = await self._persist_message(message_create)
            if persisted_message:
                msg_id = persisted_message.id
                message.msg_id = msg_id
                return message
        except Exception as e:
            logger.error(f"Ran into error when persist message: {e}")
        return message

    async def _persist_message(self, message_create: MessageCreate) -> Optional[Message]:
        """
        Persist a new message to storage.

        Args:
            message_create: The message data to persist
        """
        return await self.message_client.create(message_create)

    async def get_messages_asc(self, conversation_id: str, limit: int = 10) -> list[MessageParam]:
        """
        Get messages in ascending order (oldest to newest) matching natural conversation flow.

        Args:
            limit: Maximum number of messages to load.

        Returns:
            List[MessageParam]: List of messages in ASC order (oldest first),
                              or empty list if service manager is not set.

        Example:
            messages = await message_manager.get_messages_asc(limit=10)
            # Messages are already in chronological order, ready for display
        """

        messages = await self.message_client.get_conversation_messages(conversation_id=conversation_id, limit=limit)
        message_params = [
            MessageParamFactory.from_message(message, force_str_content=True, truncate_length=250)
            for message in reversed(messages)  # Reverse the DESC order from DB to get ASC
        ]

        logger.debug(f"Loaded {len(message_params)} messages in ASC order")
        return message_params
