from enum import Enum
from typing import Any, Optional
from datetime import datetime, timezone

from pydantic import Field, BaseModel

__all__ = [
    "AgentEventType",
    "AgentState",
    "AgentControlType",
    "AgentStateTransition",
    "AgentEventPayload",
    "AgentControlPayload",
    "AgentStatePayload",
]


class AgentEventType(str, Enum):
    STATE_CHANGED = "state_changed"
    TASK_COMPLETED = "task_completed"
    ERROR_OCCURRED = "error_occurred"
    RESOURCE_USAGE_UPDATED = "resource_usage_updated"


class AgentState(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    STOPPED = "stopped"


class AgentControlType(str, Enum):
    RESET = "reset"
    STOP = "stop"
    PAUSE = "pause"
    RESUME = "resume"


# Base class for agent-related payloads
class AgentPayloadBase(BaseModel):
    agent_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    sequence_number: Optional[int] = None


class AgentStateTransition(BaseModel):
    previous_state: AgentState
    new_state: AgentState
    reason: Optional[str] = None


class AgentEventPayload(AgentPayloadBase):
    event_type: AgentEventType
    state_transition: Optional[AgentStateTransition] = None
    data: dict[str, Any] = Field(default_factory=dict)


class AgentControlPayload(AgentPayloadBase):
    control_type: AgentControlType
    parameters: dict[str, Any] = Field(default_factory=dict)

    # Allow recipient to be used as agent_id for compatibility
    def __init__(self, **data):
        # Map recipient to agent_id if agent_id is not provided
        if "agent_id" not in data and "recipient" in data:
            data["agent_id"] = data["recipient"]
        super().__init__(**data)


class AgentStatePayload(AgentPayloadBase):
    state: AgentState
    previous_state: Optional[AgentState] = None
    current_task: Optional[str] = None
    memory_usage: Optional[int] = None
    error_message: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
