"""Agent state management module."""

import logging
from typing import Dict, Optional
from datetime import datetime

from pydantic import Field, BaseModel, field_serializer

from ..schemas import ConversationContext
from ..utils.token_counter import TokenCounter

logger = logging.getLogger(__name__)


class TokenStats(BaseModel):
    """Token statistics for different components."""

    system: int = Field(default=0, description="System message tokens")
    tool: int = Field(default=0, description="Tool definition tokens")
    project: int = Field(default=0, description="Project context tokens")
    memories: int = Field(default=0, description="Memory tokens")
    summaries: int = Field(default=0, description="Message summary tokens")
    messages: int = Field(default=0, description="Current message tokens")
    context_window: Optional[Dict] = Field(default=None, description="Context window stats")
    actual_usage: Dict = Field(default_factory=dict, description="Actual token usage from API")


class AgentMetrics(BaseModel):
    """Agent runtime metrics."""

    token_stats: TokenStats = Field(default_factory=TokenStats)
    start_time: datetime = Field(default_factory=datetime.now)
    total_messages: int = Field(default=0)
    tool_calls: int = Field(default=0)
    errors: int = Field(default=0)
    last_error: Optional[str] = None

    @field_serializer("start_time")
    def serialize_start_time(self, v: datetime) -> str:
        return v.isoformat()


class AgentState:
    """Encapsulates agent state management.

    This class manages the runtime state of an agent, including:
    - Initialization state
    - Token statistics
    - Performance metrics
    - Conversation context
    - System context
    """

    def __init__(self):
        """Initialize agent state."""
        self.has_initialized: bool = False
        self.metrics = AgentMetrics()
        self.conversation_context: Optional[ConversationContext] = None
        self.system_context: Optional[str] = None
        self.system_message_param: Optional[str] = None
        self._token_counter = TokenCounter()

    def record_error(self, error: Exception) -> None:
        """Record an error occurrence."""
        self.metrics.errors += 1
        self.metrics.last_error = str(error)
        logger.error(f"Agent error: {error}")

    def record_tool_call(self) -> None:
        """Record a tool call."""
        self.metrics.tool_calls += 1

    def record_message(self) -> None:
        """Record a message processed."""
        self.metrics.total_messages += 1

    def update_token_stats(self, component: str, content: str) -> None:
        """Update token statistics for a component.

        Args:
            component: Component name (system, tool, project, memories, summaries, messages)
            content: Content to count tokens for
        """
        if not hasattr(self.metrics.token_stats, component):
            logger.warning(f"Invalid token stat component: {component}")
            return

        tokens = self._token_counter.count_token(content)
        setattr(self.metrics.token_stats, component, tokens)

    def update_context_stats(self, stats: Dict) -> None:
        """Update context window statistics.

        Args:
            stats: Dictionary of context window statistics
        """
        self.metrics.token_stats.context_window = stats

    def update_usage_stats(self, usage: Dict) -> None:
        """Update actual token usage statistics.

        Args:
            usage: Dictionary of actual token usage from API
        """
        self.metrics.token_stats.actual_usage = usage

    def get_token_stats(self) -> Dict:
        """Get current token statistics.

        Returns:
            Dict of token statistics
        """
        return self.metrics.token_stats.model_dump()

    def get_metrics(self) -> Dict:
        """Get current metrics.

        Returns:
            Dict of metrics
        """
        return self.metrics.model_dump()

    def get_metrics_json(self) -> str:
        """Get current metrics.

        Returns:
            string of metrics
        """
        return self.metrics.model_dump_json(indent=4)

    def reset_metrics(self) -> None:
        """Reset metrics to initial state."""
        self.metrics = AgentMetrics()

    def __str__(self) -> str:
        """String representation of agent state."""
        return f"AgentState(initialized={self.has_initialized}, messages={self.metrics.total_messages}, errors={self.metrics.errors})"
