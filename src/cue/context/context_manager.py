"""Dynamic context management system."""

import logging
from uuid import uuid4
from typing import Set, Dict, List, Union, Optional

from ..utils import TokenCounter
from .config import ContextConfig
from .errors import TokenLimitError, ToolSequenceError, InvalidMessageError
from ..schemas import FeatureFlag, MessageParam, CompletionResponse
from .token_manager import TokenManager
from .._agent_summarizer import ContentSummarizer

logger = logging.getLogger(__name__)


class DynamicContextManager:
    """Manages dynamic context for conversation agents.
    
    Handles message management, token limits, and context optimization
    with improved error handling and configuration management.
    """
    def __init__(
        self,
        model: str,
        max_tokens: int,
        feature_flag: Optional[FeatureFlag] = None,
        summarizer: Optional[ContentSummarizer] = None,
    ):
        """Initialize with better configuration management."""
        self.config = ContextConfig(
            model=model,
            max_tokens=max_tokens,
            feature_flag=feature_flag or FeatureFlag(),
        )
        self.token_manager = TokenManager(
            config=self.config,
            counter=TokenCounter(),
        )
        self.summarizer = summarizer

        # Core state
        self.messages: List[Dict] = []
        self.summaries: List[str] = []
        self._current_tokens = 0

    async def add_messages(
        self,
        new_messages: List[Union[Dict, MessageParam, CompletionResponse]]
    ) -> bool:
        """Add messages with improved error handling and token management."""
        try:
            if not new_messages:
                return False

            # Process and validate messages
            processed_messages = [
                self._prepare_message_dict(msg)
                for msg in new_messages
            ]

            # Add messages and check tokens
            for message in processed_messages:
                await self._add_single_message(message)

            # Check if truncation needed
            token_info = self.token_manager.analyze_tokens(self.messages)
            if token_info.needs_truncation:
                return await self._handle_truncation(token_info.tokens_to_remove)

            return False

        except Exception as e:
            logger.error(f"Error adding messages: {e}")
            if isinstance(e, (TokenLimitError, InvalidMessageError, ToolSequenceError)):
                raise
            raise ContextError(f"Failed to add messages: {e}") from e

    async def _add_single_message(self, message: Dict) -> None:
        """Add a single message with token checking."""
        msg_tokens = self.token_manager.estimate_message_tokens(message)

        if not self.token_manager.can_add_message(self._current_tokens, msg_tokens):
            raise TokenLimitError(
                f"Adding message would exceed token limit of {self.config.max_tokens}"
            )

        self.messages.append(message)
        self._current_tokens += msg_tokens

    async def _handle_truncation(self, tokens_to_remove: int) -> bool:
        """Handle message truncation with improved logic."""
        if not tokens_to_remove:
            return False

        # Find messages to remove
        total_tokens = 0
        remove_indices = set()

        # Start from oldest messages (beginning of list)
        for i, msg in enumerate(self.messages):
            if i in self._find_preserved_indices():
                continue

            msg_tokens = self.token_manager.estimate_message_tokens(msg)
            total_tokens += msg_tokens
            remove_indices.add(i)

            if total_tokens >= tokens_to_remove:
                break

        if not remove_indices:
            return False

        # Create summary if needed
        if self.summarizer:
            removed_messages = [
                msg for i, msg in enumerate(self.messages)
                if i in remove_indices
            ]
            summary = await self.summarizer.summarize(removed_messages)
            if summary:
                self.summaries.append(summary)
                while len(self.summaries) > self.config.max_summaries:
                    self.summaries.pop(0)

        # Remove messages
        self.messages = [
            msg for i, msg in enumerate(self.messages)
            if i not in remove_indices
        ]

        # Recalculate tokens
        self._current_tokens = self.token_manager.analyze_tokens(
            self.messages
        ).total_tokens

        return True

    def _find_preserved_indices(self) -> Set[int]:
        """Find indices of messages that should be preserved."""
        preserved = set()

        # Always preserve the most recent message
        if self.messages:
            preserved.add(len(self.messages) - 1)

        # Find and preserve tool sequences
        for i, msg in enumerate(self.messages):
            if self._is_tool_call(msg):
                sequence = self._find_tool_sequence_indices(i)
                preserved.update(sequence)

        return preserved

    def _prepare_message_dict(
        self,
        message: Union[Dict, MessageParam, CompletionResponse],
        msg_id: Optional[str] = None
    ) -> Dict:
        """Prepare message with better validation."""
        if isinstance(message, (MessageParam, CompletionResponse)):
            message = message.dict()

        if not isinstance(message, dict):
            raise InvalidMessageError(f"Invalid message type: {type(message)}")

        if "role" not in message:
            raise InvalidMessageError("Message missing required 'role' field")

        message_copy = message.copy()
        message_copy["msg_id"] = msg_id or f"msg_{uuid4().hex[:8]}"

        return message_copy

    def _is_tool_call(self, message: Dict) -> bool:
        """Check if message is a tool call."""
        return (
            message.get("role") == "assistant" and
            message.get("tool_calls") is not None
        )

    def _find_tool_sequence_indices(self, start_idx: int) -> Set[int]:
        """Find indices for a complete tool sequence."""
        if start_idx >= len(self.messages):
            return set()

        sequence = {start_idx}
        start_msg = self.messages[start_idx]

        if not self._is_tool_call(start_msg):
            return sequence

        # Find matching tool results
        tool_calls = start_msg.get("tool_calls", [])
        call_ids = {call.get("id") for call in tool_calls}

        for i in range(start_idx + 1, len(self.messages)):
            msg = self.messages[i]
            if (
                msg.get("role") == "tool" and
                msg.get("tool_call_id") in call_ids
            ):
                sequence.add(i)

        return sequence
