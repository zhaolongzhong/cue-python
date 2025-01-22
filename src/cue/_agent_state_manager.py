import time
import asyncio
import logging
from typing import Any, Dict, Optional
from datetime import datetime, timezone

from .types import AgentState, EventMessage, EventMessageType, AgentStatePayload
from .services.service_manager import ServiceManager

logger = logging.getLogger(__name__)


class AgentStateManager:
    def __init__(self, service_manager: Optional[ServiceManager] = None):
        self._agent_states: Dict[str, AgentState] = {}
        self._service_manager = service_manager
        self._state_locks: Dict[str, asyncio.Lock] = {}
        self._last_timestamps: Dict[str, float] = {}  # Per-agent last timestamp

    async def set_agent_state(
        self, agent_id: str, new_state: AgentState, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Update agent state and broadcast state change"""
        lock = self._state_locks.setdefault(agent_id, asyncio.Lock())

        async with lock:
            old_state = self._agent_states.get(agent_id)
            self._agent_states[agent_id] = new_state

            if self._service_manager:
                if new_state == AgentState.ERROR:
                    await self._broadcast_state_change(agent_id, old_state, new_state, metadata)

    def get_agent_state(self, agent_id: str) -> AgentState:
        """Get current state of an agent"""
        return self._agent_states.get(agent_id, AgentState.IDLE)

    def _get_sequence_number(self, agent_id: str) -> int:
        """Generate monotonically increasing sequence number based on current timestamp"""
        current_time = int(time.time() * 1000)  # Convert to milliseconds

        # Ensure the sequence number is strictly greater than the last one
        last_time = self._last_timestamps.get(agent_id, 0)
        sequence_number = max(current_time, last_time + 1)

        # Store this sequence number
        self._last_timestamps[agent_id] = sequence_number

        return sequence_number

    async def _broadcast_state_change(
        self,
        agent_id: str,
        old_state: Optional[AgentState],
        new_state: AgentState,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Broadcast state change event through service manager"""
        sequence_number = self._get_sequence_number(agent_id)

        event_message = EventMessage(
            type=EventMessageType.AGENT_STATE,
            payload=AgentStatePayload(
                agent_id=agent_id,
                state=new_state,
                timestamp=datetime.now(timezone.utc),
                sequence_number=sequence_number,
                previous_state=old_state,
                metadata=metadata or {},
            ),
        )
        await self._service_manager.send_event_message(event_message)
