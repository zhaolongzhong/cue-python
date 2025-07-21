import logging
from typing import Any, Optional
from datetime import datetime

from ..schemas import Message
from .._session_context import SessionContext
from ..utils.token_counter import TokenCounter
from ..services.service_manager import ServiceManager

logger = logging.getLogger(__name__)


class TaskContextManager:
    def __init__(
        self,
        session_context: SessionContext,
        service_manager: ServiceManager,
        max_tokens: int = 2000,
        max_chars: int = 1000,
    ):
        """
        Initialize the TaskContextManager with a maximum token limit.

        Args:
            session_context (SessionContext): The session context
            max_tokens (int): Maximum number of tokens to maintain in task context
            max_chars (int): Maximum characters for each message entry
            message_client (Optional[MessageClient]): Client for loading messages from remote
        """
        self.session_context = session_context
        self.service_manager = service_manager
        self.max_tokens = max_tokens
        self.max_chars: int = max_chars
        self.task_messages: dict[str, str] = {}  # msg_id -> formatted content
        self.token_counter = TokenCounter()
        self.recent_task_context: Optional[str] = None
        self.message_param: Optional[dict] = None
        self.task_state = {
            "status": "active",  # active, debugging, completed, paused
            "start_time": datetime.now().isoformat(),
            "current_goal": None,
            "conversation_id": None,  # Set when loading from remote
        }

    async def load_from_remote(self) -> None:
        """
        Load task context from remote message history.
        """
        if not self.service_manager.messages:
            logger.warning("No message client configured, skipping remote load")
            return
        conversation_id = self.session_context.conversation_id
        if not conversation_id:
            logger.debug("conversation_id is none")
            return
        try:
            # Load user messages from remote
            messages = await self.service_manager.messages.get_conversation_messages(
                conversation_id=conversation_id,
                role="user",
                content_type="text",
                skip=0,
                limit=10,
            )

            # Update conversation ID
            self.task_state["conversation_id"] = conversation_id

            # Process messages
            if messages:
                self.add_task_messages(messages)
                logger.info(f"Loaded {len(messages)} messages from conversation {conversation_id}")
            else:
                logger.info(f"No messages found for conversation {conversation_id}")

        except Exception as e:
            logger.error(f"Failed to load messages from remote: {str(e)}")
            raise

    def _get_total_tokens(self) -> int:
        """Get the total token count for all task messages in the current window."""
        if not self.task_messages:
            return 0
        combined_messages = "\n".join(self.task_messages.values())
        return self.token_counter.count_token(content=combined_messages)

    def _truncate_center(self, text: str, max_length: int) -> str:
        """Truncate text from the center if it exceeds max_length."""
        if len(text) <= max_length:
            return text
        half_length = (max_length - 3) // 2
        return text[:half_length] + "..." + text[-half_length:]

    def _format_task_message(self, message: Message) -> str:
        """Format a message for task context inclusion."""
        content = message.content.get_text()
        role = message.author.role
        msg_type = self._determine_message_type(message)

        return f"[{msg_type}] ({role}): {content}"

    def _determine_message_type(self, message: Message) -> str:
        """Determine the type/importance of a message for task context."""
        content = message.content.get_text().lower()

        if any(word in content for word in ["goal", "task", "objective", "need to", "please", "help"]):
            if not self.task_state["current_goal"]:
                self.task_state["current_goal"] = message.content.get_text()
            return "TASK_GOAL"

        if message.author.role == "user":
            return "USER_INPUT"

        return "TASK_PROGRESS"

    def add_task_messages(self, messages: list[Message]) -> None:
        """
        Update task context with new messages while respecting token limits.
        Messages are assumed to be already sorted by recency (newest to oldest).

        Args:
            messages (List[Message]): List of messages to process
        """
        # Keep original task goal and error context if they exist
        goal_msg = None
        error_msg = None

        for msg_id, content in self.task_messages.items():
            if "TASK_GOAL" in content:
                goal_msg = (msg_id, content)

        self.task_messages.clear()

        # Restore goal and error context first if they exist
        if goal_msg:
            self.task_messages[goal_msg[0]] = goal_msg[1]
        if error_msg and self.task_state["status"] == "debugging":
            self.task_messages[error_msg[0]] = error_msg[1]

        # Process each message while respecting token limits
        for message in messages:
            if not message.id:
                continue

            # Format and truncate if necessary
            formatted_message = self._format_task_message(message)
            truncated_message = self._truncate_center(formatted_message, self.max_chars)

            # Add to task messages
            self.task_messages[message.id] = truncated_message

            # Check token limit
            if self._get_total_tokens() > self.max_tokens:
                # Remove the message we just added (unless it's a goal)
                if "TASK_GOAL" not in truncated_message:
                    self.task_messages.pop(message.id)
                logger.debug(
                    f"Stopped adding task messages due to token limit. "
                    f"Current tokens: {self._get_total_tokens()}/{self.max_tokens}"
                )
                break

        # Update the formatted task context
        self.update_task_context()

    def get_formatted_task_context(self) -> Optional[str]:
        """
        Get task context formatted for the LLM.
        Returns None if no task context is present.
        """
        if not self.task_messages:
            logger.debug("no task context")
            return None

        combined_messages = "\n".join(self.task_messages.values())

        status_info = f"Task Status: {self.task_state['status']}\nStart Time: {self.task_state['start_time']}\n"

        if self.task_state["conversation_id"]:
            status_info += f"Conversation ID: {self.task_state['conversation_id']}\n"

        return f"""Current task context and state:
{status_info}
<task_context>
{combined_messages}
</task_context>
"""

    def update_task_context(self) -> None:
        """Update the task context string representation."""
        previous = self.recent_task_context
        self.recent_task_context = self.get_formatted_task_context()
        logger.debug(f"update_task_context, \nprevious: {previous}, \nnew: {self.recent_task_context}")
        if self.recent_task_context:
            self.message_param = {"role": "user", "content": self.recent_task_context}

    def get_task_context_param(self) -> Optional[dict]:
        """Get the task context as a message parameter."""
        return self.message_param

    def get_task_stats(self) -> dict[str, Any]:
        """Get statistics about the current task context."""
        total_tokens = self._get_total_tokens()
        return {
            "message_count": len(self.task_messages),
            "total_tokens": total_tokens,
            "remaining_tokens": self.max_tokens - total_tokens,
            "is_at_capacity": total_tokens >= self.max_tokens,
            "task_state": self.task_state.copy(),
        }

    def clear_task_context(self) -> None:
        """Clear all task context."""
        self.task_messages.clear()
        self.task_state["status"] = "completed"
