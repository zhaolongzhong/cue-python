from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable
from dataclasses import field, dataclass


@dataclass
class Message:
    role: str
    content: str


@dataclass
class Tool:
    name: str
    description: str
    parameters: Dict[str, Any]


@dataclass
class InputItem:
    type: str  # "text", "image", etc.
    content: Any


# NextStep types for multi-turn decision making
class NextStep(ABC):  # noqa: B024
    """Base class for next step decisions in multi-turn conversations"""

    pass


@dataclass
class NextStepRunAgain(NextStep):
    """Continue conversation - tools were called, need model response"""

    pass


@dataclass
class NextStepFinalOutput(NextStep):
    """Stop conversation - we have final text output"""

    pass


@dataclass
class StepResult:
    """Result from a single model call/step"""

    content: str
    usage: Dict[str, int] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    next_step: NextStep = field(default_factory=NextStepFinalOutput)


@dataclass
class RunResult:
    """Result from a complete multi-turn conversation"""

    content: str  # Final output
    steps: List[StepResult] = field(default_factory=list)  # All intermediate steps
    usage: Dict[str, int] = field(default_factory=dict)  # Accumulated usage
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SimpleAgent:
    model: str
    system_prompt: str = ""
    messages: List[Message] = field(default_factory=list)
    tools: List[Tool] = field(default_factory=list)
    max_turns: int = 10
    api_key: Optional[str] = None


@runtime_checkable
class Session(Protocol):
    """Protocol for session implementations.

    Session stores conversation history for a specific session, allowing
    agents to maintain context without requiring explicit manual memory management.
    """

    session_id: str

    async def get_items(self, limit: int | None = None) -> list[InputItem]:
        """Retrieve the conversation history for this session.

        Args:
            limit: Maximum number of items to retrieve. If None, retrieves all items.
                   When specified, returns the latest N items in chronological order.

        Returns:
            List of input items representing the conversation history
        """
        ...

    async def add_items(self, items: list[InputItem]) -> None:
        """Add new items to the conversation history.

        Args:
            items: List of input items to add to the history
        """
        ...

    async def pop_item(self) -> InputItem | None:
        """Remove and return the most recent item from the session.

        Returns:
            The most recent item if it exists, None if the session is empty
        """
        ...

    async def clear_session(self) -> None:
        """Clear all items for this session."""
        ...


class SessionABC(ABC):
    """Abstract base class for session implementations.

    Session stores conversation history for a specific session, allowing
    agents to maintain context without requiring explicit manual memory management.

    This ABC is intended for internal use and as a base class for concrete implementations.
    Third-party libraries should implement the Session protocol instead.
    """

    session_id: str

    @abstractmethod
    async def get_items(self, limit: int | None = None) -> list[InputItem]:
        """Retrieve the conversation history for this session.

        Args:
            limit: Maximum number of items to retrieve. If None, retrieves all items.
                   When specified, returns the latest N items in chronological order.

        Returns:
            List of input items representing the conversation history
        """
        ...

    @abstractmethod
    async def add_items(self, items: list[InputItem]) -> None:
        """Add new items to the conversation history.

        Args:
            items: List of input items to add to the history
        """
        ...

    @abstractmethod
    async def pop_item(self) -> InputItem | None:
        """Remove and return the most recent item from the session.

        Returns:
            The most recent item if it exists, None if the session is empty
        """
        ...

    @abstractmethod
    async def clear_session(self) -> None:
        """Clear all items for this session."""
        ...
