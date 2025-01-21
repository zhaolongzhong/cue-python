"""Tests for agent state management."""

from datetime import datetime

from cue.agent import AgentState, TokenStats, AgentMetrics


def test_agent_state_initialization():
    """Test AgentState initialization."""
    state = AgentState()
    assert not state.has_initialized
    assert isinstance(state.metrics, AgentMetrics)
    assert state.system_context is None
    assert state.system_message_param is None


def test_token_stats_defaults():
    """Test TokenStats default values."""
    stats = TokenStats()
    assert stats.system == 0
    assert stats.tool == 0
    assert stats.project == 0
    assert stats.memories == 0
    assert stats.summaries == 0
    assert stats.messages == 0
    assert stats.context_window is None
    assert isinstance(stats.actual_usage, dict)
    assert len(stats.actual_usage) == 0


def test_agent_metrics_initialization():
    """Test AgentMetrics initialization."""
    metrics = AgentMetrics()
    assert isinstance(metrics.token_stats, TokenStats)
    assert isinstance(metrics.start_time, datetime)
    assert metrics.total_messages == 0
    assert metrics.tool_calls == 0
    assert metrics.errors == 0
    assert metrics.last_error is None


def test_record_error():
    """Test error recording."""
    state = AgentState()
    error = Exception("Test error")
    state.record_error(error)
    assert state.metrics.errors == 1
    assert state.metrics.last_error == str(error)


def test_record_tool_call():
    """Test tool call recording."""
    state = AgentState()
    state.record_tool_call()
    assert state.metrics.tool_calls == 1


def test_record_message():
    """Test message recording."""
    state = AgentState()
    state.record_message()
    assert state.metrics.total_messages == 1


def test_update_token_stats():
    """Test token statistics updating."""
    state = AgentState()
    test_content = "Test content"

    # Test valid component
    state.update_token_stats("system", test_content)
    stats = state.get_token_stats()
    assert stats["system"] > 0

    # Test invalid component
    state.update_token_stats("invalid", test_content)
    stats = state.get_token_stats()
    assert "invalid" not in stats


def test_update_context_stats():
    """Test context statistics updating."""
    state = AgentState()
    test_stats = {"total": 100, "used": 50}
    state.update_context_stats(test_stats)
    assert state.metrics.token_stats.context_window == test_stats


def test_update_usage_stats():
    """Test usage statistics updating."""
    state = AgentState()
    test_usage = {"prompt_tokens": 100, "completion_tokens": 50}
    state.update_usage_stats(test_usage)
    assert state.metrics.token_stats.actual_usage == test_usage


def test_get_token_stats():
    """Test getting token statistics."""
    state = AgentState()
    stats = state.get_token_stats()
    assert isinstance(stats, dict)
    assert "system" in stats
    assert "tool" in stats
    assert "project" in stats
    assert "memories" in stats
    assert "summaries" in stats
    assert "messages" in stats


def test_get_metrics():
    """Test getting metrics."""
    state = AgentState()
    metrics = state.get_metrics()
    assert isinstance(metrics, dict)
    assert "token_stats" in metrics
    assert "start_time" in metrics
    assert "total_messages" in metrics
    assert "tool_calls" in metrics
    assert "errors" in metrics


def test_reset_metrics():
    """Test metrics reset."""
    state = AgentState()
    state.record_message()
    state.record_tool_call()
    state.record_error(Exception("Test error"))

    state.reset_metrics()
    assert state.metrics.total_messages == 0
    assert state.metrics.tool_calls == 0
    assert state.metrics.errors == 0
    assert state.metrics.last_error is None


def test_string_representation():
    """Test string representation."""
    state = AgentState()
    state.record_message()
    state.record_error(Exception("Test error"))

    str_rep = str(state)
    assert "initialized=False" in str_rep
    assert "messages=1" in str_rep
    assert "errors=1" in str_rep
