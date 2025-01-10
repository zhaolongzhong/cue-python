"""Tests for agent manager state management."""
from datetime import datetime, timedelta

import pytest

from cue.agent.agent_manager_state import (
    TransferRecord,
    AgentManagerState,
    AgentManagerMetrics,
    AgentManagerStateManager,
)


@pytest.fixture
def state_manager():
    return AgentManagerStateManager()

@pytest.fixture
def metrics():
    return AgentManagerMetrics()

def test_initial_state(state_manager):
    """Test initial state is uninitialized."""
    assert state_manager.state == AgentManagerState.UNINITIALIZED
    assert state_manager.can_initialize()
    assert not state_manager.can_start_run()
    assert not state_manager.is_running()

def test_state_transitions(state_manager):
    """Test state transitions work correctly."""
    # Test initialization transition
    state_manager.start_initialization()
    assert state_manager.state == AgentManagerState.INITIALIZING

    # Test completion transition
    state_manager.complete_initialization()
    assert state_manager.state == AgentManagerState.READY
    assert state_manager.can_start_run()

    # Test run transition
    state_manager.start_run()
    assert state_manager.state == AgentManagerState.RUNNING
    assert state_manager.is_running()

    # Test stop transition
    state_manager.stop_run()
    assert state_manager.state == AgentManagerState.STOPPED

def test_error_handling(state_manager):
    """Test error state transitions."""
    # Test error during initialization
    state_manager.start_initialization()
    state_manager.complete_initialization(error=ValueError("test error"))
    assert state_manager.state == AgentManagerState.ERROR

    # Test error during run
    state_manager = AgentManagerStateManager()  # Reset
    state_manager.start_initialization()
    state_manager.complete_initialization()
    state_manager.start_run()
    state_manager.stop_run(error=RuntimeError("run error"))
    assert state_manager.state == AgentManagerState.ERROR

def test_invalid_transitions(state_manager):
    """Test invalid state transitions are caught."""
    # Can't start run from uninitialized
    with pytest.raises(RuntimeError):
        state_manager.start_run()

    # Can't initialize twice
    state_manager.start_initialization()
    with pytest.raises(RuntimeError):
        state_manager.start_initialization()

def test_metrics_recording(metrics):
    """Test metrics recording functionality."""
    # Test transfer recording
    metrics.record_transfer("agent1", "agent2", True)
    assert metrics.total_transfers == 1
    assert metrics.successful_transfers == 1
    assert metrics.failed_transfers == 0

    metrics.record_transfer("agent2", "agent3", False, "transfer failed")
    assert metrics.total_transfers == 2
    assert metrics.successful_transfers == 1
    assert metrics.failed_transfers == 1

    # Test error recording
    error = ValueError("test error")
    metrics.record_error(error)
    assert metrics.last_error == str(error)
    assert metrics.errors_by_type["ValueError"] == 1

    # Test run recording
    metrics.record_run_start()
    assert metrics.total_runs == 1

    # Test active agents update
    metrics.update_active_agents(3)
    assert metrics.active_agents == 3

def test_metrics_summary(metrics):
    """Test metrics summary generation."""
    # Record some test data
    metrics.record_transfer("agent1", "agent2", True)
    metrics.record_transfer("agent2", "agent3", False, "failed")
    metrics.record_error(ValueError("test error"))
    metrics.record_run_start()
    metrics.update_active_agents(2)

    # Get metrics summary
    summary = metrics.get_metrics()

    # Verify summary content
    assert summary["total_transfers"] == 2
    assert summary["successful_transfers"] == 1
    assert summary["failed_transfers"] == 1
    assert summary["transfer_success_rate"] == 50.0
    assert summary["total_runs"] == 1
    assert summary["active_agents"] == 2
    assert summary["last_error"] == "test error"
    assert summary["errors_by_type"]["ValueError"] == 1
    assert len(summary["recent_transfers"]) == 2

def test_recent_transfers_limit(metrics):
    """Test that recent transfers list is limited."""
    # Add more than 10 transfers
    for i in range(15):
        metrics.record_transfer(f"agent{i}", f"agent{i+1}", True)

    # Verify only last 10 are kept
    assert len(metrics.recent_transfers) == 10
    # Verify they are the most recent ones
    assert metrics.recent_transfers[-1].from_agent == "agent14"

def test_transfer_record_creation():
    """Test transfer record creation."""
    record = TransferRecord(
        from_agent="agent1",
        to_agent="agent2",
        timestamp=datetime.now(),
        success=True,
        error=None
    )
    assert record.from_agent == "agent1"
    assert record.to_agent == "agent2"
    assert record.success
    assert record.error is None

def test_metrics_uptime(metrics):
    """Test uptime calculation."""
    # Create metrics with known start time
    start_time = datetime.now() - timedelta(seconds=60)
    metrics.start_time = start_time

    # Get metrics
    summary = metrics.get_metrics()

    # Verify uptime is approximately correct (allowing 1 second variance)
    assert 59 <= summary["uptime_seconds"] <= 61
