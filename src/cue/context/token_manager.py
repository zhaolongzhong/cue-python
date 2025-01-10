"""Token management for context system."""

from typing import Dict, List, Optional
from dataclasses import dataclass

from ..utils import TokenCounter
from .config import ContextConfig


@dataclass
class TokenInfo:
    """Information about token usage and truncation needs."""
    total_tokens: int
    needs_truncation: bool
    tokens_to_remove: Optional[int] = None


class TokenManager:
    """Handles token counting and truncation decisions.
    
    Centralizes token management logic for better maintainability
    and consistent behavior across the context system.
    """
    def __init__(self, config: ContextConfig, counter: TokenCounter):
        self.config = config
        self.counter = counter

    def analyze_tokens(self, messages: List[Dict]) -> TokenInfo:
        """Analyze messages and determine if truncation is needed."""
        total_tokens = self.counter.count_messages_tokens(messages)
        needs_truncation = total_tokens > self.config.max_tokens

        tokens_to_remove = None
        if needs_truncation:
            target = int(self.config.max_tokens * (1 - self.config.batch_remove_percentage))
            tokens_to_remove = total_tokens - target

        return TokenInfo(
            total_tokens=total_tokens,
            needs_truncation=needs_truncation,
            tokens_to_remove=tokens_to_remove
        )

    def estimate_message_tokens(self, message: Dict) -> int:
        """Estimate tokens for a single message."""
        return self.counter.count_dict_tokens(message)

    def can_add_message(self, current_tokens: int, message_tokens: int) -> bool:
        """Check if a new message can be added without exceeding limits."""
        total = current_tokens + message_tokens
        # Allow if under max or if removing the excess would still leave us above min
        return (
            total <= self.config.max_tokens or
            (total - self.config.min_tokens_to_keep) >= message_tokens
        )
