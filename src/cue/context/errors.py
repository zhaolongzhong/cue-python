"""Error types for context management system."""

class ContextError(Exception):
    """Base error for context management system."""
    pass


class TokenLimitError(ContextError):
    """Raised when token limits are exceeded and cannot be resolved."""
    pass


class MessageError(ContextError):
    """Base class for message-related errors."""
    pass


class InvalidMessageError(MessageError):
    """Raised when a message fails validation."""
    pass


class ToolSequenceError(MessageError):
    """Raised when there are issues with tool call sequences."""
    pass
