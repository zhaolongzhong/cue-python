from typing import Any, Optional
from datetime import datetime

from .types.agent_event import AgentState, AgentControlType, AgentStatePayload, AgentControlPayload
from .types.event_message import EventMessage, EventMessageType


class AgentStateMachine:
    def handle_control_message(self, message: EventMessage) -> EventMessage:
        if not isinstance(message.payload, AgentControlPayload):
            raise ValueError("Invalid payload type")

        # Process control message and generate state transition
        if message.payload.control_type == AgentControlType.RESET:
            return create_agent_state_message(message.payload.agent_id, AgentState.PAUSED, "Paused by user request")

        # Handle other control types...
        return create_agent_state_message(message.payload.agent_id, AgentState.IDLE)


def create_agent_control_message(
    agent_id: str, control_type: AgentControlType, parameters: Optional[dict[str, Any]] = None
) -> EventMessage:
    return EventMessage(
        type=EventMessageType.AGENT_CONTROL,
        payload=AgentControlPayload(
            agent_id=agent_id, control_type=control_type, parameters=parameters or {}, sequence_number=1
        ),
        websocket_request_id=f"ws-{datetime.utcnow().timestamp()}",
        metadata={"source": "user_client"},
    )


def create_agent_state_message(agent_id: str, state: AgentState, current_task: Optional[str] = None) -> EventMessage:
    return EventMessage(
        type=EventMessageType.AGENT_STATE,
        payload=AgentStatePayload(agent_id=agent_id, state=state, current_task=current_task, sequence_number=1),
        metadata={"timestamp": datetime.utcnow().isoformat()},
    )
