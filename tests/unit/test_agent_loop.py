"""Tests for agent execution loop."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from cue.schemas import RunMetadata
from cue._agent_loop import AgentLoop
from cue.agent.agent_loop_state import AgentLoopState


@pytest.fixture
def agent_loop():
    return AgentLoop()


@pytest.fixture
def mock_agent():
    agent = AsyncMock()
    agent.id = "test_agent"
    agent.config = MagicMock()
    agent.config.is_primary = True
    return agent


@pytest.fixture
def mock_tool_manager():
    return MagicMock()


@pytest.fixture
def run_metadata():
    return RunMetadata(max_turns=5)


@pytest.mark.asyncio
async def test_agent_loop_initialization(agent_loop):
    """Test agent loop initialization."""
    assert agent_loop.state_manager.state == AgentLoopState.READY
    assert not agent_loop.stop_run_event.is_set()
    assert agent_loop.execute_run_task is None


@pytest.mark.asyncio
async def test_agent_loop_metrics(agent_loop):
    """Test agent loop metrics tracking."""
    metrics = agent_loop.get_metrics()
    assert metrics["total_runs"] == 0
    assert metrics["successful_runs"] == 0
    assert metrics["failed_runs"] == 0
    assert metrics["total_messages"] == 0
    assert metrics["total_tool_calls"] == 0


@pytest.mark.asyncio
async def test_agent_loop_basic_run(agent_loop, mock_agent, mock_tool_manager, run_metadata):
    """Test basic agent loop run."""
    # Mock successful response
    mock_response = MagicMock()
    mock_response.get_tool_calls.return_value = []
    mock_response.get_text.return_value = "Test response"
    mock_agent.run.return_value = mock_response

    response = await agent_loop.run(agent=mock_agent, tool_manager=mock_tool_manager, run_metadata=run_metadata)

    assert response == mock_response
    assert agent_loop.state_manager.state == AgentLoopState.STOPPED
    metrics = agent_loop.get_metrics()
    assert metrics["total_runs"] == 1
    assert metrics["successful_runs"] == 1


@pytest.mark.asyncio
async def test_agent_loop_error_handling(agent_loop, mock_agent, mock_tool_manager, run_metadata):
    """Test agent loop error handling."""
    # Mock error during run
    mock_agent.run.side_effect = ValueError("Test error")

    with pytest.raises(ValueError):
        await agent_loop.run(agent=mock_agent, tool_manager=mock_tool_manager, run_metadata=run_metadata)

    assert agent_loop.state_manager.state == AgentLoopState.ERROR
    metrics = agent_loop.get_metrics()
    assert metrics["total_runs"] == 1
    assert metrics["failed_runs"] == 1
    assert "ValueError" in metrics["errors_by_type"]


@pytest.mark.asyncio
async def test_agent_loop_graceful_stop(agent_loop, mock_agent, mock_tool_manager, run_metadata):
    """Test graceful stop of agent loop."""

    # Setup mock for long-running task
    async def mock_long_run(*args, **kwargs):
        await asyncio.sleep(0.1)
        return MagicMock()

    mock_agent.run.side_effect = mock_long_run

    # Start run in background
    agent_loop.execute_run_task = asyncio.create_task(
        agent_loop.run(agent=mock_agent, tool_manager=mock_tool_manager, run_metadata=run_metadata)
    )

    # Stop the loop
    await agent_loop.stop()

    assert agent_loop.state_manager.state == AgentLoopState.STOPPING
    assert agent_loop.stop_run_event.is_set()
    assert agent_loop.execute_run_task is None


@pytest.mark.asyncio
async def test_agent_loop_message_queue(agent_loop, mock_agent, mock_tool_manager, run_metadata):
    """Test processing of queued messages."""
    # Add a message to the queue
    await agent_loop.user_message_queue.put("Test message")

    mock_response = MagicMock()
    mock_response.get_tool_calls.return_value = []
    mock_response.get_text.return_value = "Test response"
    mock_agent.run.return_value = mock_response

    await agent_loop.run(agent=mock_agent, tool_manager=mock_tool_manager, run_metadata=run_metadata)

    # Verify message was processed
    mock_agent.add_message.assert_called_once()
    metrics = agent_loop.get_metrics()
    assert metrics["total_messages"] >= 1


@pytest.mark.asyncio
async def test_agent_loop_tool_calls(agent_loop, mock_agent, mock_tool_manager, run_metadata):
    """Test processing of tool calls."""
    # Mock response with tool calls
    mock_response = MagicMock()
    mock_response.get_tool_calls.return_value = ["test_tool"]
    mock_response.get_text.return_value = "Test response"
    mock_agent.run.return_value = mock_response

    # Mock tool processing
    mock_tool_result = MagicMock()
    mock_tool_result.agent_transfer = None
    mock_tool_result.base64_images = None
    mock_agent.client.process_tools_with_timeout.return_value = mock_tool_result

    await agent_loop.run(agent=mock_agent, tool_manager=mock_tool_manager, run_metadata=run_metadata)

    metrics = agent_loop.get_metrics()
    assert metrics["total_tool_calls"] >= 1
