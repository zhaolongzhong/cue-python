"""State management for agent manager."""

from enum import Enum
from typing import Dict, List, Optional
from datetime import datetime
from dataclasses import field, dataclass


class AgentManagerState(Enum):
    """States for the agent manager."""

    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    ERROR = "error"
    STOPPING = "stopping"
    STOPPED = "stopped"
    CLEANING = "cleaning"


@dataclass
class TransferRecord:
    """Record of an agent transfer."""

    from_agent: str
    to_agent: str
    timestamp: datetime
    success: bool
    error: Optional[str] = None


@dataclass
class AgentManagerMetrics:
    """Metrics for monitoring agent manager performance."""

    total_transfers: int = 0
    successful_transfers: int = 0
    failed_transfers: int = 0
    total_runs: int = 0
    active_agents: int = 0
    start_time: datetime = field(default_factory=datetime.now)
    last_error: Optional[str] = None
    errors_by_type: Dict[str, int] = field(default_factory=dict)
    recent_transfers: List[TransferRecord] = field(default_factory=list)

    def record_transfer(self, from_agent: str, to_agent: str, success: bool, error: Optional[str] = None):
        """Record a transfer attempt."""
        self.total_transfers += 1
        if success:
            self.successful_transfers += 1
        else:
            self.failed_transfers += 1

        record = TransferRecord(
            from_agent=from_agent, to_agent=to_agent, timestamp=datetime.now(), success=success, error=error
        )
        self.recent_transfers.append(record)
        # Keep only last 10 transfers
        self.recent_transfers = self.recent_transfers[-10:]

    def record_error(self, error: Exception):
        """Record an error occurrence."""
        self.last_error = str(error)
        error_type = error.__class__.__name__
        self.errors_by_type[error_type] = self.errors_by_type.get(error_type, 0) + 1

    def record_run_start(self):
        """Record the start of a new run."""
        self.total_runs += 1

    def update_active_agents(self, count: int):
        """Update count of active agents."""
        self.active_agents = count

    def get_metrics(self) -> dict:
        """Get all metrics as a dictionary."""
        return {
            "total_transfers": self.total_transfers,
            "successful_transfers": self.successful_transfers,
            "failed_transfers": self.failed_transfers,
            "transfer_success_rate": (
                self.successful_transfers / self.total_transfers * 100 if self.total_transfers > 0 else 0
            ),
            "total_runs": self.total_runs,
            "active_agents": self.active_agents,
            "uptime_seconds": (datetime.now() - self.start_time).total_seconds(),
            "last_error": self.last_error,
            "errors_by_type": self.errors_by_type,
            "recent_transfers": [
                {
                    "from": t.from_agent,
                    "to": t.to_agent,
                    "timestamp": t.timestamp.isoformat(),
                    "success": t.success,
                    "error": t.error,
                }
                for t in self.recent_transfers
            ],
        }


class AgentManagerStateManager:
    """Manages state and metrics for the agent manager."""

    def __init__(self):
        self._state = AgentManagerState.UNINITIALIZED
        self.metrics = AgentManagerMetrics()

    @property
    def state(self) -> AgentManagerState:
        """Get current state."""
        return self._state

    @state.setter
    def state(self, new_state: AgentManagerState):
        """Update state with validation."""
        if not isinstance(new_state, AgentManagerState):
            raise ValueError(f"Invalid state: {new_state}")
        self._state = new_state

    def can_initialize(self) -> bool:
        """Check if manager can be initialized."""
        return self._state in [AgentManagerState.UNINITIALIZED]

    def can_start_run(self) -> bool:
        """Check if manager can start a new run."""
        return self._state == AgentManagerState.READY

    def is_running(self) -> bool:
        """Check if manager is currently running."""
        return self._state == AgentManagerState.RUNNING

    def should_stop(self) -> bool:
        """Check if manager should stop."""
        return self._state in [AgentManagerState.STOPPING, AgentManagerState.ERROR]

    def start_initialization(self):
        """Mark the start of initialization."""
        if not self.can_initialize():
            raise RuntimeError(f"Cannot initialize in {self._state} state")
        self._state = AgentManagerState.INITIALIZING

    def complete_initialization(self, error: Optional[Exception] = None):
        """Mark completion of initialization."""
        if error:
            self._state = AgentManagerState.ERROR
            self.metrics.record_error(error)
        else:
            self._state = AgentManagerState.READY

    def start_run(self):
        """Mark the start of a new run."""
        if not self.can_start_run():
            raise RuntimeError(f"Cannot start run in {self._state} state")
        self._state = AgentManagerState.RUNNING
        self.metrics.record_run_start()

    def stop_run(self, error: Optional[Exception] = None):
        """Mark the end of a run."""
        if error:
            self._state = AgentManagerState.ERROR
            self.metrics.record_error(error)
        else:
            self._state = AgentManagerState.STOPPED

    def get_metrics(self) -> dict:
        """Get current metrics."""
        return self.metrics.get_metrics()
