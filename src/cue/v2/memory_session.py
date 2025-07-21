from __future__ import annotations

import uuid
from typing import Optional

from .types import InputItem, SessionABC


class InMemorySession(SessionABC):
    """In-memory session implementation for conversation history.

    Stores conversation items in memory for the duration of the process.
    Suitable for simple use cases and testing.
    """

    def __init__(self, session_id: Optional[str] = None):
        """Initialize in-memory session.

        Args:
            session_id: Optional session ID. If not provided, generates a random UUID.
        """
        self.session_id = session_id or str(uuid.uuid4())
        self._items: list[InputItem] = []

    async def get_items(self, limit: int | None = None) -> list[InputItem]:
        """Retrieve the conversation history for this session.

        Args:
            limit: Maximum number of items to retrieve. If None, retrieves all items.
                   When specified, returns the latest N items in chronological order.

        Returns:
            List of input items representing the conversation history
        """
        if limit is None:
            return self._items.copy()

        # Return the latest N items
        if limit <= 0:
            return []

        return self._items[-limit:] if len(self._items) >= limit else self._items.copy()

    async def add_items(self, items: list[InputItem]) -> None:
        """Add new items to the conversation history.

        Args:
            items: List of input items to add to the history
        """
        self._items.extend(items)

    async def pop_item(self) -> InputItem | None:
        """Remove and return the most recent item from the session.

        Returns:
            The most recent item if it exists, None if the session is empty
        """
        if not self._items:
            return None
        return self._items.pop()

    async def clear_session(self) -> None:
        """Clear all items for this session."""
        self._items.clear()

    def __len__(self) -> int:
        """Return the number of items in the session."""
        return len(self._items)

    def __repr__(self) -> str:
        """String representation of the session."""
        return f"InMemorySession(session_id='{self.session_id}', items={len(self._items)})"
