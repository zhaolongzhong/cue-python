"""
Clean message parameters utilities.

Simple, focused utilities for message parameter handling.
No over-engineering - just what's needed.
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass


@dataclass
class MessageParams:
    """Clean message parameters structure"""

    content: str
    role: str = "user"
    metadata: Optional[Dict[str, Any]] = None


class MessageParamsUtils:
    """Simple message parameters utilities"""

    @staticmethod
    def create_user_message(content: str, **kwargs) -> MessageParams:
        """Create a user message with content"""
        return MessageParams(content=content, role="user", metadata=kwargs if kwargs else None)

    @staticmethod
    def create_assistant_message(content: str, **kwargs) -> MessageParams:
        """Create an assistant message with content"""
        return MessageParams(content=content, role="assistant", metadata=kwargs if kwargs else None)

    @staticmethod
    def create_system_message(content: str, **kwargs) -> MessageParams:
        """Create a system message with content"""
        return MessageParams(content=content, role="system", metadata=kwargs if kwargs else None)

    @staticmethod
    def to_dict(message: MessageParams) -> Dict[str, Any]:
        """Convert message to dictionary format"""
        result = {"role": message.role, "content": message.content}
        if message.metadata:
            result.update(message.metadata)
        return result

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> MessageParams:
        """Create message from dictionary"""
        content = data.get("content", "")
        role = data.get("role", "user")

        # Extract metadata (everything except content and role)
        metadata = {k: v for k, v in data.items() if k not in ["content", "role"]}

        return MessageParams(content=content, role=role, metadata=metadata if metadata else None)

    @staticmethod
    def format_for_provider(messages: List[MessageParams], provider: str) -> List[Dict[str, Any]]:
        """Format messages for specific provider"""
        formatted = []

        for msg in messages:
            base = MessageParamsUtils.to_dict(msg)

            # Provider-specific formatting can be added here
            if provider == "anthropic":
                # Anthropic-specific formatting
                formatted.append(base)
            elif provider == "openai":
                # OpenAI-specific formatting
                formatted.append(base)
            elif provider == "google":
                # Google-specific formatting
                formatted.append(base)
            else:
                # Default formatting
                formatted.append(base)

        return formatted
