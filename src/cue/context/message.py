import logging
from typing import Union, Optional

from ..schemas import Message, MessageParam, MessageCreate
from ..services.service_manager import ServiceManager
from ..schemas.completion_response import CompletionResponse
from ..schemas.tool_response_wrapper import ToolResponseWrapper

logger = logging.getLogger(__name__)


class MessageManager:
    def __init__(
        self,
    ):
        self.service_manager: Optional[ServiceManager] = None

    def set_service_manager(self, service_manager: ServiceManager):
        self.service_manager = service_manager

    async def persist_message(
        self, message: Union[CompletionResponse, ToolResponseWrapper, MessageParam]
    ) -> Union[CompletionResponse, ToolResponseWrapper, MessageParam]:
        if not isinstance(message, (CompletionResponse, ToolResponseWrapper, MessageParam)):
            logger.error("Unexpect message type to persist")
            return message
        if message.msg_id:
            logger.error(f"Message is already persisted with id: {message.msg_id}")
            return message

        try:
            message_create = message.to_message_create()
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
        if not self.service_manager:
            logger.warning("Skip persisting message: service_manager is not set.")
            return None

        return await self.service_manager.messages.create(message_create)

    async def get_messages_asc(self, limit: int = 10) -> list[MessageParam]:
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
        if not self.service_manager:
            logger.warning("Skip loading messages: service_manager is not set.")
            return []

        messages = await self.service_manager.messages.get_conversation_messages(limit=limit)
        message_params = [
            MessageParam.from_message(message, force_str_content=True, truncate_length=250)
            for message in reversed(messages)  # Reverse the DESC order from DB to get ASC
        ]

        logger.debug(f"Loaded {len(message_params)} messages in ASC order")
        return message_params
