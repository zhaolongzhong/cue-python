import time
import asyncio
from unittest.mock import Mock, AsyncMock

import pytest

from cue.types.agent_event import AgentState, AgentStatePayload
from cue.types.event_message import EventMessage, EventMessageType
from cue._agent_state_manager import AgentStateManager
from cue.services.service_manager import ServiceManager


@pytest.fixture
def service_manager():
    """Create a mock service manager"""
    mock_service_manager = Mock(spec=ServiceManager)
    mock_service_manager.send_event_message = AsyncMock()
    return mock_service_manager


@pytest.fixture
def state_manager(service_manager):
    """Create an AgentStateManager instance with mock service manager"""
    return AgentStateManager(service_manager=service_manager)


@pytest.mark.asyncio
async def test_sequence_number_monotonic_increase(state_manager):
    """Test that sequence numbers are monotonically increasing"""
    agent_id = "test-agent"

    # Get multiple sequence numbers in quick succession
    seq1 = state_manager._get_sequence_number(agent_id)
    seq2 = state_manager._get_sequence_number(agent_id)
    seq3 = state_manager._get_sequence_number(agent_id)

    # Verify they are strictly increasing
    assert seq1 < seq2 < seq3

    # Verify they are close to current timestamp
    current_time = int(time.time() * 1000)
    assert abs(current_time - seq1) < 1000  # Within 1 second


@pytest.mark.asyncio
async def test_state_transition_broadcast(state_manager, service_manager):
    """Test that state transitions are properly broadcasted"""
    agent_id = "test-agent"
    old_state = AgentState.IDLE
    new_state = AgentState.RUNNING
    metadata = {"reason": "start_processing"}

    # Perform state transition
    await state_manager._broadcast_state_change(
        agent_id=agent_id, old_state=old_state, new_state=new_state, metadata=metadata
    )

    # Verify service manager was called with correct event message
    service_manager.send_event_message.assert_called_once()
    call_args = service_manager.send_event_message.call_args[0][0]

    assert isinstance(call_args, EventMessage)
    assert call_args.type == EventMessageType.AGENT_STATE
    assert isinstance(call_args.payload, AgentStatePayload)
    assert call_args.payload.agent_id == agent_id
    assert call_args.payload.state == new_state
    assert call_args.payload.previous_state == old_state
    assert call_args.payload.metadata == metadata
    assert isinstance(call_args.payload.sequence_number, int)


@pytest.mark.asyncio
async def test_concurrent_state_transitions(state_manager):
    """Test handling of concurrent state transitions"""
    agent_id = "test-agent"
    num_transitions = 5

    async def perform_transition(i):
        await state_manager._broadcast_state_change(
            agent_id=agent_id, old_state=AgentState.IDLE, new_state=AgentState.RUNNING, metadata={"transition": i}
        )
        return state_manager._last_timestamps[agent_id]

    # Execute multiple transitions concurrently
    tasks = [perform_transition(i) for i in range(num_transitions)]
    timestamps = await asyncio.gather(*tasks)

    # Verify timestamps are strictly increasing
    assert timestamps == sorted(timestamps)
    assert len(set(timestamps)) == num_transitions  # All unique


@pytest.mark.asyncio
async def test_state_transition_with_missing_metadata(state_manager, service_manager):
    """Test state transition handling with missing optional fields"""
    agent_id = "test-agent"

    # Perform state transition without optional fields
    await state_manager._broadcast_state_change(
        agent_id=agent_id, old_state=None, new_state=AgentState.RUNNING, metadata=None
    )

    # Verify call was made with default values
    service_manager.send_event_message.assert_called_once()
    call_args = service_manager.send_event_message.call_args[0][0]

    assert call_args.payload.previous_state is None
    assert call_args.payload.metadata == {}


@pytest.mark.asyncio
async def test_service_manager_error_handling(state_manager, service_manager):
    """Test error handling when service manager fails"""
    service_manager.send_event_message.side_effect = Exception("Network error")

    # Attempt state transition
    with pytest.raises(Exception) as exc_info:
        await state_manager._broadcast_state_change(
            agent_id="test-agent", old_state=AgentState.IDLE, new_state=AgentState.RUNNING
        )

    assert str(exc_info.value) == "Network error"
