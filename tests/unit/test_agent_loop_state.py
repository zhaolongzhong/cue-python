"""Tests for agent loop state management."""

from datetime import datetime

import pytest

from cue.agent.agent_loop_state import LoopMetrics, AgentLoopState, AgentLoopStateManager


def test_loop_metrics_initialization():
    """Test LoopMetrics initialization."""
    metrics = LoopMetrics()
    assert metrics.total_runs == 0
    assert metrics.successful_runs == 0
    assert metrics.failed_runs == 0
    assert metrics.total_messages == 0
    assert metrics.total_tool_calls == 0
    assert isinstance(metrics.start_time, datetime)
    assert metrics.last_error is None
    assert metrics.errors_by_type == {}


def test_loop_metrics_recording():
    """Test recording various metrics."""
    metrics = LoopMetrics()

    # Test run recording
    metrics.record_run_start()
    assert metrics.total_runs == 1

    metrics.record_run_success()
    assert metrics.successful_runs == 1

    test_error = ValueError("test error")
    metrics.record_run_failure(test_error)
    assert metrics.failed_runs == 1
    assert metrics.last_error == str(test_error)
    assert metrics.errors_by_type["ValueError"] == 1

    # Test message and tool call recording
    metrics.record_message()
    assert metrics.total_messages == 1

    metrics.record_tool_call()
    assert metrics.total_tool_calls == 1


def test_metrics_get_metrics():
    """Test getting metrics as dictionary."""
    metrics = LoopMetrics()
    metrics.record_run_start()
    metrics.record_run_success()
    metrics.record_message()
    metrics.record_tool_call()

    metrics_dict = metrics.get_metrics()
    assert metrics_dict["total_runs"] == 1
    assert metrics_dict["successful_runs"] == 1
    assert metrics_dict["total_messages"] == 1
    assert metrics_dict["total_tool_calls"] == 1
    assert isinstance(metrics_dict["uptime_seconds"], float)
    assert metrics_dict["last_error"] is None
    assert metrics_dict["errors_by_type"] == {}


def test_state_manager_initialization():
    """Test AgentLoopStateManager initialization."""
    manager = AgentLoopStateManager()
    assert manager.state == AgentLoopState.READY
    assert isinstance(manager.metrics, LoopMetrics)


def test_state_manager_state_validation():
    """Test state validation."""
    manager = AgentLoopStateManager()

    # Test valid state change
    manager.state = AgentLoopState.RUNNING
    assert manager.state == AgentLoopState.RUNNING

    # Test invalid state change
    with pytest.raises(ValueError):
        manager.state = "invalid_state"


def test_state_manager_state_checks():
    """Test state check methods."""
    manager = AgentLoopStateManager()

    # Test initial state
    assert manager.can_start() is True
    assert manager.is_running() is False
    assert manager.should_stop() is False

    # Test running state
    manager.state = AgentLoopState.RUNNING
    assert manager.can_start() is False
    assert manager.is_running() is True
    assert manager.should_stop() is False

    # Test stopping state
    manager.state = AgentLoopState.STOPPING
    assert manager.can_start() is False
    assert manager.is_running() is False
    assert manager.should_stop() is True


def test_state_manager_run_lifecycle():
    """Test complete run lifecycle."""
    manager = AgentLoopStateManager()

    # Start run
    manager.start_run()
    assert manager.state == AgentLoopState.RUNNING
    assert manager.metrics.total_runs == 1

    # Stop run successfully
    manager.stop_run()
    assert manager.state == AgentLoopState.STOPPED
    assert manager.metrics.successful_runs == 1

    # Start another run and stop with error
    manager.start_run()
    error = Exception("test error")
    manager.stop_run(error)
    assert manager.state == AgentLoopState.ERROR
    assert manager.metrics.failed_runs == 1
    assert manager.metrics.last_error == str(error)


def test_state_manager_invalid_start():
    """Test starting run in invalid state."""
    manager = AgentLoopStateManager()
    manager.state = AgentLoopState.RUNNING

    with pytest.raises(RuntimeError):
        manager.start_run()


def test_state_manager_metrics():
    """Test metrics access through state manager."""
    manager = AgentLoopStateManager()
    manager.start_run()
    manager.metrics.record_message()
    manager.metrics.record_tool_call()
    manager.stop_run()

    metrics = manager.get_metrics()
    assert metrics["total_runs"] == 1
    assert metrics["successful_runs"] == 1
    assert metrics["total_messages"] == 1
    assert metrics["total_tool_calls"] == 1
