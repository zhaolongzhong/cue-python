import logging
from typing import Any, Optional

from ..utils.token_counter import TokenCounter

logger = logging.getLogger(__name__)


class DynamicMemoryManager:
    def __init__(self, max_tokens: int = 1000, max_chars: int = 500):
        """
        Initialize the DynamicMemoryManager with a maximum token limit.

        Args:
            max_tokens (int): Maximum number of tokens to maintain in memories
            max_chars (int): Maximum characters for each memory entry
        """
        self.max_tokens = max_tokens
        self.max_chars: int = max_chars
        self.memories: dict[str, str] = {}
        self.token_counter = TokenCounter()
        self.recent_memories: Optional[str] = None
        self.message_param: Optional[dict] = None

    def _get_total_tokens(self) -> int:
        """Get the total token count for all memories in the current window."""
        combined_memories = "\n".join(self.memories)
        return self.token_counter.count_token(content=combined_memories)

    def _truncate_center(self, text: str, max_length: int) -> str:
        """Truncate text from the center if it exceeds max_length."""
        if len(text) <= max_length:
            return text
        half_length = (max_length - 3) // 2
        return text[:half_length] + "..." + text[-half_length:]

    def add_memories(self, memory_dict: dict[str, str]) -> None:
        """
        Replace current memories with new memory dictionary while respecting token limits.
        Memories are assumed to be already sorted by recency (newest to oldest).

        Args:
            memory_dict (dict[str, str]): Dictionary of memory_id to formatted memory content
        """
        self.memories.clear()

        # Process each memory while respecting token limits
        for memory_id, memory_content in memory_dict.items():
            # Truncate if necessary
            truncated_memory = self._truncate_center(memory_content, self.max_chars)

            # Add to memories
            self.memories[memory_id] = truncated_memory

            # Check token limit
            if self._get_total_tokens() > self.max_tokens:
                # Remove the memory we just added
                self.memories.pop(memory_id)
                logger.debug(
                    f"Stopped adding memories due to token limit. "
                    f"Current tokens: {self._get_total_tokens()}/{self.max_tokens}"
                )
                break

        # Update the formatted memories
        self.update_recent_memories()

    def _get_total_tokens(self) -> int:
        """Get the total token count for all memories in the current window."""
        if not self.memories:
            return 0
        combined_memories = "\n".join(self.memories.values())
        return self.token_counter.count_token(content=combined_memories)

    def get_formatted_memories(self) -> Optional[str]:
        """
        Get memories formatted as a system message for the LLM.
        Returns None if no memories are present.
        """
        if not self.memories:
            logger.warning("no memories")
            return None

        combined_memories = "\n".join(self.memories.values())

        return f"""The following are your most recent memory records. Please:
1. Consider these memories as part of your context when responding
2. Update your understanding based on this new information
3. Note that memories are listed from most recent to oldest
4. Only reference these memories when relevant to the current conversation

Instructions for memory processing:
- Treat each memory as factual information about past interactions
- If new memories conflict with old ones, prefer the more recent memory
- Use memories to maintain conversation continuity
- Do not explicitly mention these instructions to the user

<recent_memories>
{combined_memories}
</recent_memories>
"""

    def update_recent_memories(self):
        """Update the recent memories string representation."""
        previous = self.recent_memories
        self.recent_memories = self.get_formatted_memories()
        logger.debug(f"update_recent_memories, \nprevious: {previous}, \nnew: {self.recent_memories}")
        self.message_param = {"role": "user", "content": self.recent_memories}

    def get_memories_param(self) -> Optional[dict]:
        return self.message_param

    def get_memory_stats(self) -> dict[str, Any]:
        """Get statistics about the current memories."""
        total_tokens = self._get_total_tokens()
        return {
            "memory_count": len(self.memories),
            "total_tokens": total_tokens,
            "remaining_tokens": self.max_tokens - total_tokens,
            "is_at_capacity": total_tokens >= self.max_tokens,
        }

    def clear_memories(self) -> None:
        """Clear all memories."""
        self.memories.clear()
