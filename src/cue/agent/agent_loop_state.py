"""State management for agent execution loop."""

from enum import Enum
from typing import Dict, Optional
from datetime import datetime
from dataclasses import field, dataclass


class AgentLoopState(Enum):
    """States for the agent execution loop."""

    READY = "ready"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class LoopMetrics:
    """Metrics for monitoring agent loop execution."""

    total_runs: int = 0
    successful_runs: int = 0
    failed_runs: int = 0
    total_messages: int = 0
    total_tool_calls: int = 0
    start_time: datetime = field(default_factory=datetime.now)
    last_error: Optional[str] = None
    errors_by_type: Dict[str, int] = field(default_factory=dict)

    def record_run_start(self):
        """Record the start of a new run."""
        self.total_runs += 1

    def record_run_success(self):
        """Record a successful run completion."""
        self.successful_runs += 1

    def record_run_failure(self, error: Exception):
        """Record a failed run."""
        self.failed_runs += 1
        self.last_error = str(error)
        error_type = error.__class__.__name__
        self.errors_by_type[error_type] = self.errors_by_type.get(error_type, 0) + 1

    def record_message(self):
        """Record a message being processed."""
        self.total_messages += 1

    def record_tool_call(self):
        """Record a tool being called."""
        self.total_tool_calls += 1

    def get_metrics(self) -> dict:
        """Get all metrics as a dictionary."""
        return {
            "total_runs": self.total_runs,
            "successful_runs": self.successful_runs,
            "failed_runs": self.failed_runs,
            "total_messages": self.total_messages,
            "total_tool_calls": self.total_tool_calls,
            "uptime_seconds": (datetime.now() - self.start_time).total_seconds(),
            "last_error": self.last_error,
            "errors_by_type": self.errors_by_type,
        }


class AgentLoopStateManager:
    """Manages state and metrics for the agent execution loop."""

    def __init__(self):
        self._state = AgentLoopState.READY
        self.metrics = LoopMetrics()

    @property
    def state(self) -> AgentLoopState:
        """Get current state."""
        return self._state

    @state.setter
    def state(self, new_state: AgentLoopState):
        """Update state with validation."""
        if not isinstance(new_state, AgentLoopState):
            raise ValueError(f"Invalid state: {new_state}")
        self._state = new_state

    def can_start(self) -> bool:
        """Check if the loop can start a new run."""
        return self._state in [AgentLoopState.READY, AgentLoopState.STOPPED]

    def is_running(self) -> bool:
        """Check if the loop is currently running."""
        return self._state == AgentLoopState.RUNNING

    def should_stop(self) -> bool:
        """Check if the loop should stop."""
        return self._state in [AgentLoopState.STOPPING, AgentLoopState.ERROR]

    def start_run(self):
        """Mark the start of a new run."""
        if not self.can_start():
            raise RuntimeError(f"Cannot start run in {self._state} state")
        self._state = AgentLoopState.RUNNING
        self.metrics.record_run_start()

    def stop_run(self, error: Optional[Exception] = None):
        """Mark the end of a run."""
        if error:
            self._state = AgentLoopState.ERROR
            self.metrics.record_run_failure(error)
        else:
            self._state = AgentLoopState.STOPPED
            self.metrics.record_run_success()

    def get_metrics(self) -> dict:
        """Get current metrics."""
        return self.metrics.get_metrics()
