import asyncio
from unittest.mock import Mock, AsyncMock

import pytest
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionAssistantMessageParam,
)

from cue._agent import Agent
from cue.schemas import (
    Author,
    AgentConfig,
    FeatureFlag,
    RunMetadata,
    AgentTransfer,
    CompletionResponse,
    ToolResponseWrapper,
)
from cue._agent_loop import AgentLoop
from cue.tools._tool import ToolManager


@pytest.fixture
def agent_config() -> AgentConfig:
    """Create test agent configuration."""
    return AgentConfig(
        id="test_agent",
        name="test_agent",
        description="Test agent",
        instruction="You are a test agent",
        model="gpt-4o-mini",
        tools=["edit", "memory"],
        is_primary=True,
        feature_flag=FeatureFlag(enable_services=False, enable_storage=False),
    )


@pytest.fixture
def agent(agent_config: AgentConfig) -> Agent:
    """Create a mock agent for testing."""
    mock_agent = Mock(spec=Agent)
    mock_agent.id = agent_config.id
    mock_agent.config = agent_config
    mock_agent.add_message = AsyncMock()
    mock_agent.add_messages = AsyncMock()
    mock_agent.run = AsyncMock()
    mock_agent.persist_message = AsyncMock()
    mock_agent.client = Mock()
    mock_agent.client.process_tools_with_timeout = AsyncMock()
    return mock_agent


@pytest.fixture
def tool_manager() -> ToolManager:
    """Create a mock tool manager for testing."""
    return Mock(spec=ToolManager)


@pytest.fixture
def run_metadata() -> RunMetadata:
    """Create test run metadata."""
    return RunMetadata(
        current_turn=0,
        max_turns=10,
        enable_turn_debug=False,
    )


@pytest.mark.asyncio
async def test_agent_loop_basic_flow(agent: Agent, tool_manager: ToolManager, run_metadata: RunMetadata):
    """Test basic agent loop flow with text response."""
    # Setup
    agent_loop = AgentLoop()

    # Create mock response
    msg = ChatCompletionAssistantMessageParam(role="assistant", content="Test response")
    chat_completion = ChatCompletion(
        id="test_id",
        model="gpt-4o-mini",
        object="chat.completion",
        choices=[{"message": msg, "finish_reason": "stop", "index": 0}],
        usage={"completion_tokens": 10, "prompt_tokens": 20, "total_tokens": 30},
        created=1234567890,
    )

    mock_response = CompletionResponse(
        msg_id="test_id",
        model="gpt-4o-mini",
        author=Author(name="test_agent", role="assistant"),
        response=chat_completion,
    )
    agent.run.return_value = mock_response
    callback = AsyncMock()

    # Execute
    response = await agent_loop.run(
        agent=agent, tool_manager=tool_manager, run_metadata=run_metadata, callback=callback
    )

    # Verify
    assert response == mock_response
    assert run_metadata.current_turn == 1
    agent.run.assert_awaited_once_with(
        tool_manager=tool_manager,
        run_metadata=run_metadata,
        author=Author(role="user", name=""),
    )
    agent.add_message.assert_awaited_once()
    callback.assert_awaited_once()


@pytest.mark.asyncio
async def test_agent_loop_with_tool_calls(agent: Agent, tool_manager: ToolManager, run_metadata: RunMetadata):
    """Test agent loop with tool calls."""
    # Setup
    agent_loop = AgentLoop()

    # Set up multiple responses for consecutive run calls
    tool_call_msg = ChatCompletionAssistantMessageParam(
        role="assistant",
        content="Using tool",
        tool_calls=[
            {"id": "call_123", "type": "function", "function": {"name": "test_tool", "arguments": '{"param": "value"}'}}
        ],
    )
    tool_call_completion = ChatCompletion(
        id="test_id",
        model="gpt-4o-mini",
        object="chat.completion",
        choices=[{"message": tool_call_msg, "finish_reason": "tool_calls", "index": 0}],
        usage={"completion_tokens": 10, "prompt_tokens": 20, "total_tokens": 30},
        created=1234567890,
    )

    final_msg = ChatCompletionAssistantMessageParam(role="assistant", content="Task completed")
    final_completion = ChatCompletion(
        id="test_id2",
        model="gpt-4o-mini",
        object="chat.completion",
        choices=[{"message": final_msg, "finish_reason": "stop", "index": 0}],
        usage={"completion_tokens": 10, "prompt_tokens": 20, "total_tokens": 30},
        created=1234567890,
    )

    tool_call_response = CompletionResponse(
        msg_id="test_id",
        model="gpt-4o-mini",
        author=Author(name="test_agent", role="assistant"),
        response=tool_call_completion,
    )
    final_response = CompletionResponse(
        msg_id="test_id2",
        model="gpt-4o-mini",
        author=Author(name="test_agent", role="assistant"),
        response=final_completion,
    )

    agent.run.side_effect = [tool_call_response, final_response]

    # Mock tool execution result
    tool_result = ToolResponseWrapper(
        model="gpt-4o-mini", tool_messages=[{"tool_call_id": "call_123", "content": "Tool execution result"}]
    )
    agent.client.process_tools_with_timeout.return_value = tool_result
    callback = AsyncMock()

    # Execute
    response = await agent_loop.run(
        agent=agent, tool_manager=tool_manager, run_metadata=run_metadata, callback=callback
    )

    # Verify
    assert response == final_response
    assert run_metadata.current_turn == 2
    assert agent.run.call_count == 2
    assert agent.client.process_tools_with_timeout.await_count == 1
    assert agent.add_messages.await_count == 1
    assert callback.await_count == 3  # Called for tool call, tool result, and final response


@pytest.mark.asyncio
async def test_agent_loop_with_agent_transfer(agent: Agent, tool_manager: ToolManager, run_metadata: RunMetadata):
    """Test agent loop with agent transfer request."""
    # Setup
    agent_loop = AgentLoop()
    agent.config.is_primary = False  # Set as non-primary agent

    # Mock response that should trigger transfer
    msg = ChatCompletionAssistantMessageParam(role="assistant", content="Transferring to primary")
    chat_completion = ChatCompletion(
        id="test_id",
        model="gpt-4o-mini",
        object="chat.completion",
        choices=[{"message": msg, "finish_reason": "stop", "index": 0}],
        usage={"completion_tokens": 10, "prompt_tokens": 20, "total_tokens": 30},
        created=1234567890,
    )

    response = CompletionResponse(
        msg_id="test_id",
        model="gpt-4o-mini",
        author=Author(name="test_agent", role="assistant"),
        response=chat_completion,
    )
    agent.run.return_value = response

    # Execute
    result = await agent_loop.run(agent=agent, tool_manager=tool_manager, run_metadata=run_metadata)

    # Verify
    assert isinstance(result, AgentTransfer)
    assert result.transfer_to_primary
    assert result.message == "Transferring to primary"
    assert result.run_metadata == run_metadata
    agent.run.assert_awaited_once_with(
        tool_manager=tool_manager,
        run_metadata=run_metadata,
        author=Author(role="user", name=""),
    )


@pytest.mark.asyncio
async def test_agent_loop_stop(agent: Agent, tool_manager: ToolManager, run_metadata: RunMetadata):
    """Test stopping the agent loop."""
    # Setup
    agent_loop = AgentLoop()

    # Mock long-running operation
    async def mock_run(*args, **kwargs):
        await asyncio.sleep(1)
        msg = ChatCompletionAssistantMessageParam(role="assistant", content="Response")
        chat_completion = ChatCompletion(
            id="test_id",
            model="gpt-4o-mini",
            object="chat.completion",
            choices=[{"message": msg, "finish_reason": "stop", "index": 0}],
            usage={"completion_tokens": 10, "prompt_tokens": 20, "total_tokens": 30},
            created=1234567890,
        )
        return CompletionResponse(
            msg_id="test_id",
            model="gpt-4o-mini",
            author=Author(name="test_agent", role="assistant"),
            response=chat_completion,
        )

    agent.run.side_effect = mock_run

    # Start the loop in a task
    agent_loop.execute_run_task = asyncio.create_task(
        agent_loop.run(agent=agent, tool_manager=tool_manager, run_metadata=run_metadata)
    )

    # Stop the loop
    await agent_loop.stop()

    # Verify
    assert agent_loop.execute_run_task is None
    assert not agent_loop.stop_run_event.is_set()


@pytest.mark.asyncio
async def test_agent_loop_turn_limit(agent: Agent, tool_manager: ToolManager):
    """Test agent loop respects turn limits."""
    # Setup
    agent_loop = AgentLoop()
    run_metadata = RunMetadata(
        current_turn=10,  # Start at limit
        max_turns=10,
        enable_turn_debug=False,
    )

    msg = ChatCompletionAssistantMessageParam(role="assistant", content="Final response")
    chat_completion = ChatCompletion(
        id="test_id",
        model="gpt-4o-mini",
        object="chat.completion",
        choices=[{"message": msg, "finish_reason": "stop", "index": 0}],
        usage={"completion_tokens": 10, "prompt_tokens": 20, "total_tokens": 30},
        created=1234567890,
    )

    mock_response = CompletionResponse(
        msg_id="test_id",
        model="gpt-4o-mini",
        author=Author(name="test_agent", role="assistant"),
        response=chat_completion,
    )
    agent.run.return_value = mock_response

    # Execute
    response = await agent_loop.run(agent=agent, tool_manager=tool_manager, run_metadata=run_metadata)

    # Verify
    assert response is None  # Should exit due to turn limit
    agent.run.assert_not_called()  # Should not even start the run


@pytest.mark.asyncio
async def test_agent_loop_error_handling(agent: Agent, tool_manager: ToolManager, run_metadata: RunMetadata):
    """Test agent loop handles errors gracefully."""
    # Setup
    agent_loop = AgentLoop()
    agent.run.side_effect = Exception("Test error")
    callback = AsyncMock()

    # Execute
    response = await agent_loop.run(
        agent=agent, tool_manager=tool_manager, run_metadata=run_metadata, callback=callback
    )

    # Verify
    assert response is None  # Should exit due to error
    agent.run.assert_awaited_once_with(
        tool_manager=tool_manager,
        run_metadata=run_metadata,
        author=Author(role="user", name=""),
    )
    callback.assert_not_called()  # Callback should not be called on error
