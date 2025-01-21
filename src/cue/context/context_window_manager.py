import json
import logging
from typing import Any, Union, Optional

from pydantic import BaseModel

from ..types import FeatureFlag, MessageParam, CompletionResponse, ToolResponseWrapper
from ..utils import DebugUtils, TokenCounter
from ..schemas import MessageFields
from .._agent_summarizer import ContentSummarizer
from ..utils.mesage_params_utils import has_tool_calls, is_tool_result

logger = logging.getLogger(__name__)


class ContextWindowManager:
    def __init__(
        self,
        model: str,
        max_tokens: int = 20000,
        feature_flag: FeatureFlag = FeatureFlag(),
        batch_remove_percentage: float = 0.30,
        summarizer: Optional[ContentSummarizer] = None,
    ):
        """
        Initialize the DynamicContextManager with a maximum token limit.

        Args:
            model (str): LLM model id
            max_tokens (int): Maximum number of tokens to maintain in context
            batch_remove_percentage (float): Percentage of tokens to remove when limit is reached
            summarizer (ContentSummarizer): Summarize messages
        """
        self.model = model
        self.max_tokens = max_tokens
        self.feature_flag = feature_flag
        self.batch_remove_percentage = batch_remove_percentage
        self.messages: list[dict] = []
        self.summaries: list[str] = []
        self.token_counter = TokenCounter()
        self.summarizer = summarizer
        # Track window stability for cache optimization
        self.last_removal_tokens = 0  # Track tokens at last removal
        self.messages_since_removal = 0  # Track messages added since last removal
        self.summaries_content: Optional[str] = None

    def _get_batch_remove_size(self) -> int:
        total = self._get_total_tokens()
        extra_remove_size = 0
        if total > self.max_tokens:
            extra_remove_size = total - self.max_tokens
        batch_remove_size = int(self.max_tokens * self.batch_remove_percentage) + extra_remove_size
        return batch_remove_size

    def _get_total_tokens(self) -> int:
        """Get the total token count for all messages in the current window."""
        return self.token_counter.count_messages_tokens(self.messages)

    def get_context_stats(self) -> dict[str, Any]:
        """Get statistics about the current context window."""
        total_tokens = self._get_total_tokens()
        return {
            "message_count": len(self.messages),
            "total_tokens": total_tokens,
            "max_tokens": self.max_tokens,
            "remaining_tokens": self.max_tokens - total_tokens,
            "is_at_capacity": total_tokens >= self.max_tokens,
        }

    def _update_summaries_content(self):
        if not self.summaries:
            return
        # Take only the last few summaries if we have more
        recent_summaries = self.summaries[-6:] if len(self.summaries) > 6 else self.summaries

        numbered_summaries = [f"{i + 1}. {summary.strip()}" for i, summary in enumerate(recent_summaries) if summary]

        contents = "\n".join(numbered_summaries)
        self.summaries_content = contents

    def get_summaries(self) -> Optional[str]:
        if not self.summaries_content:
            return None
        return f"Here is summaries of truncated old messages <summaries>{self.summaries_content}</summaries>"

    def get_messages(self) -> list[dict]:
        """
        Get all messages in the context window, excluding internal tracking fields.

        Returns:
            list[dict]: Messages with internal tracking fields removed
        """
        return [{k: v for k, v in message.items() if k != MessageFields.MSG_ID} for message in self.messages]

    def clear_messages(self) -> None:
        """Clear all messages from the context window."""
        self.messages.clear()

    def _prepare_message_dict(
        self,
        message: Union[CompletionResponse, ToolResponseWrapper, MessageParam, dict[str, Any], BaseModel],
        msg_id: Optional[str] = None,
    ) -> Union[dict[str, Any], list[dict[str, Any]]]:
        """
        Convert message to dictionary format and optionally add msg_id.

        Args:
            message: Message to convert
            msg_id: Optional internal ID to add to the message

        Returns:
            dict or list[dict]: Prepared message dictionary or list of dictionaries
        """
        if isinstance(message, CompletionResponse):
            message_dict = message.to_params()
        elif isinstance(message, ToolResponseWrapper):
            if message.tool_messages:
                # For tool messages list, add msg_id to each message
                if msg_id:
                    return [{**msg, MessageFields.MSG_ID: msg_id} for msg in message.tool_messages]
                return message.tool_messages
            elif message.tool_result_message:
                message_dict = message.tool_result_message
            else:
                raise Exception(
                    f"Only params or tool result allowed to add message list, receive type: {type(message)}"
                )
        elif isinstance(message, (MessageParam, dict)):
            message_dict = (
                message.model_dump(exclude="model", exclude_none=True) if isinstance(message, BaseModel) else message
            )
        else:
            raise Exception(
                "Only params, tool result, dict or BaseModel allowed to add message list, "
                f"receive type: {type(message)}"
            )

        if msg_id and (MessageFields.MSG_ID not in message_dict or not message_dict[MessageFields.MSG_ID]):
            message_dict[MessageFields.MSG_ID] = msg_id

        return message_dict

    async def add_messages(
        self,
        new_messages: list[Union[CompletionResponse, ToolResponseWrapper, MessageParam, dict[str, Any], BaseModel]],
    ) -> bool:
        """Add new messages to the conversation history and manage the sliding window token limit.

        This method handles the addition of new messages while maintaining the conversation
        within the specified token limit. If the token limit is exceeded, older messages
        may be removed and optionally summarized.

        Args:
            new_messages: A list of messages to add. Can contain CompletionResponse,
                ToolResponseWrapper, MessageParam, dict, or BaseModel instances.
        Returns:
            bool: True if any messages were removed due to token limit management,
                False if all messages were retained.

        Raises:
            Exception: If an unsupported message type is provided.
        """
        if not new_messages:
            logger.warning("No messages to add, skipping")
            return False

        original_size = len(self.messages)
        for message in new_messages:
            msg_id = message.get("msg_id", None) if isinstance(message, dict) else message.msg_id
            message_dict = self._prepare_message_dict(message, msg_id)

            if isinstance(message_dict, list):
                # Handle tool messages list
                self.messages.extend(message_dict)
                self.messages_since_removal += len(message_dict)
            elif message_dict:
                self.messages.append(message_dict)
                self.messages_since_removal += 1
            else:
                logger.error(f"Unexpected message: {message}")
                continue

        total_tokens = self._get_total_tokens()
        original_tokens = total_tokens

        has_truncated = False
        # Only remove when we exceed the limit
        if total_tokens > self.max_tokens:
            orignal_len = len(self.messages)
            DebugUtils.take_snapshot(messages=self.messages, suffix="batch_remove_before", subfolder="batch_remove")
            removed_messages = self._batch_remove_messages()
            DebugUtils.take_snapshot(messages=self.messages, suffix="batch_remove_after", subfolder="batch_remove")

            new_len = len(self.messages)
            has_truncated = len(removed_messages) > 0
            total_tokens = self._get_total_tokens()
            if not has_truncated:
                logger.error(
                    f"Reach max tokens, but not remove any messages, orignal_len: {orignal_len}, new_len:{new_len}"
                )
            if self.summarizer and has_truncated:
                logger.debug(
                    f"Summarize removed messages: {len(removed_messages)}, new stats: {self.get_context_stats()}"
                )
                summary = await self.summarizer.summarize(self.model, removed_messages)
                self.summaries.append(summary)

        metadata = {
            "max_token": self.max_tokens,
            "original_size": original_size,
            "final_size": len(self.messages),
            "original_tokens": original_tokens,
            "new_total_tokens": total_tokens,
            "messages_since_removal": self.messages_since_removal,
            "has_truncated": has_truncated,
        }
        logger.debug(f"add_messages result: {json.dumps(metadata, indent=4)}")
        self._update_summaries_content()
        return has_truncated

    def _find_tool_sequence_indices(self, start_idx: int) -> set[int]:
        """
        Find all indices that belong to a tool sequence starting from a given index.
        Matches tool calls with their corresponding tool results using tool_call_id.
        """
        sequence_indices = set()
        if start_idx >= len(self.messages):
            return sequence_indices

        msg = self.messages[start_idx]
        if not has_tool_calls(msg):
            return sequence_indices

        sequence_indices.add(start_idx)
        tool_call_ids = set()

        # Collect all tool_call_ids from the tool calls message
        if msg.get("tool_calls"):
            for call in msg["tool_calls"]:
                logger.debug(f"Processing tool call: {call}")
                tool_call_ids.add(call["id"])
        elif isinstance(msg.get("content", []), list):
            for item in msg.get("content", []):
                if isinstance(item, dict) and item.get("type") == "tool_use":
                    tool_call_ids.add(item.get("id"))

        logger.debug(f"Collected tool_call_ids: {tool_call_ids}")

        # Find corresponding tool results
        remaining_ids = tool_call_ids.copy()
        for i in range(start_idx + 1, len(self.messages)):
            cur_msg = self.messages[i]
            if not is_tool_result(cur_msg):
                break

            tool_ids = []
            tool_call_id = cur_msg.get("tool_call_id")
            if tool_call_id:
                tool_ids.append(tool_call_id)
            elif isinstance(cur_msg.get("content", []), list):
                for tool_result in cur_msg.get("content", []):
                    if isinstance(tool_result, dict) and tool_result.get("type") == "tool_result":
                        id = tool_result.get("tool_use_id")
                        tool_ids.append(id)
                    else:
                        logger.error(f"Content has unexpected type: {tool_result}")
            else:
                logger.error(f"Unexpected type: {cur_msg}")

            logger.debug(f"Checking tool result with ID: {tool_ids}, remaining IDs: {remaining_ids}")
            for tool_id in tool_ids:
                if tool_id in remaining_ids:
                    sequence_indices.add(i)
                    remaining_ids.remove(tool_id)
                    logger.debug(f"Added index {i} to sequence, remaining IDs: {remaining_ids}")
                    if not remaining_ids:  # All tool results found
                        break

        if remaining_ids:
            logger.warning(f"Not all tool results found. Missing IDs: {remaining_ids}")

        logger.debug(f"Final sequence indices: {sequence_indices}")
        return sequence_indices

    def _batch_remove_messages(self) -> list[Any]:
        if not self.messages:
            return []

        tokens_to_remove = self._get_batch_remove_size()
        message_tokens = {}
        removed_tokens = 0
        messages_to_remove = 0
        current_idx = 0

        def get_message_tokens(idx):
            if idx not in message_tokens:
                message_tokens[idx] = self.token_counter.count_dict_tokens(self.messages[idx])
            return message_tokens[idx]

        while removed_tokens < tokens_to_remove and current_idx < len(self.messages):
            if has_tool_calls(self.messages[current_idx]):
                sequence_indices = self._find_tool_sequence_indices(current_idx)
                if not sequence_indices:
                    current_idx += 1
                    continue

                sequence_tokens = sum(get_message_tokens(i) for i in sequence_indices)

                # Check if adding sequence would exceed target by too much
                if removed_tokens + sequence_tokens > tokens_to_remove:
                    excess = (removed_tokens + sequence_tokens) - tokens_to_remove
                    metrics = {
                        "tokens_to_remove": tokens_to_remove,
                        "sequence_tokens": sequence_tokens,
                        "removed_tokens": removed_tokens,
                        "excess": excess,
                        "messages_to_remove": messages_to_remove,
                    }
                    logger.debug(f"About to remove extra tokens: {json.dumps(metrics, indent=4)}")
                    if excess > tokens_to_remove * 0.25:  # 25% threshold
                        break

                removed_tokens += sequence_tokens
                max_sequence_idx = max(sequence_indices)
                messages_to_remove = max(messages_to_remove, max_sequence_idx + 1)
                current_idx = max_sequence_idx + 1

            else:
                msg_tokens = get_message_tokens(current_idx)
                removed_tokens += msg_tokens
                messages_to_remove = current_idx + 1
                current_idx += 1

        # Remove messages and update tracking
        logger.debug(f"messages_to_remove: {messages_to_remove}, removed_tokens: {removed_tokens}")
        removed = self.messages[:messages_to_remove]
        self.messages = self.messages[messages_to_remove:]
        self.last_removal_tokens = self._get_total_tokens()
        self.messages_since_removal = 0

        return removed

    def build_context_for_next_agent(self, max_messages: int = 6) -> str:
        """
        Build context to be passed to the next agent, excluding the current transfer command and its result.

        Args:
            max_messages: Number of previous messages to include (0-12).
                        If 0, returns empty string as no history should be included.

        Returns:
            str: Formatted message history, excluding current transfer command and result
        """
        if max_messages == 0:
            return ""

        history = self.get_messages()

        # Since the last two messages are the transfer command and its result,
        # we exclude them and then take up to max_messages from the remaining history
        history_without_transfer = history[:-2] if len(history) >= 2 else []

        # Calculate how many messages to take
        start_idx = max(0, len(history_without_transfer) - max_messages)
        messages = history_without_transfer[start_idx:]

        messages_content = ",".join(str(msg) for msg in messages)

        return messages_content
