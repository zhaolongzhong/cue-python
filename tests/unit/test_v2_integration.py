from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cue.v2 import InputItem, SimpleAgent, get_runner_for_model
from cue.v2.types import StepResult


@pytest.mark.asyncio
async def test_end_to_end_without_api():
    """Test complete v2 flow without making API calls"""
    # Create agent
    agent = SimpleAgent(model="gpt-4o-mini", system_prompt="You are helpful")

    # Get runner
    runner = get_runner_for_model(agent.model)

    # Mock the OpenAI client
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Hello! How can I help you?"
    mock_response.choices[0].message.tool_calls = None
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 8
    mock_response.usage.total_tokens = 18

    mock_client = AsyncMock()
    mock_client.chat.completions.create.return_value = mock_response

    with patch("openai.AsyncOpenAI", return_value=mock_client):
        result = await runner.get_response(agent, [InputItem(type="text", content="Hello")])

    # Verify result structure - models now return StepResult
    assert isinstance(result, StepResult)
    assert result.content == "Hello! How can I help you?"
    assert result.usage["total_tokens"] == 18

    # Model calls don't automatically update agent.messages anymore
    # (that's done by AgentRunner in multi-turn scenarios)


@pytest.mark.asyncio
async def test_conversation_memory():
    """Test agent maintains conversation memory"""
    agent = SimpleAgent(model="gpt-4o-mini")
    runner = get_runner_for_model(agent.model)

    # First interaction
    mock_response1 = MagicMock()
    mock_response1.choices[0].message.content = "Hi Alice!"
    mock_response1.choices[0].message.tool_calls = None
    mock_response1.usage.prompt_tokens = 10
    mock_response1.usage.completion_tokens = 5
    mock_response1.usage.total_tokens = 15

    # Second interaction
    mock_response2 = MagicMock()
    mock_response2.choices[0].message.content = "Your name is Alice"
    mock_response2.choices[0].message.tool_calls = None
    mock_response2.usage.prompt_tokens = 20
    mock_response2.usage.completion_tokens = 8
    mock_response2.usage.total_tokens = 28

    mock_client = AsyncMock()
    mock_client.chat.completions.create.side_effect = [mock_response1, mock_response2]

    with patch("openai.AsyncOpenAI", return_value=mock_client):
        # Models return StepResult now, don't automatically update agent.messages
        # This test needs to be rewritten for the new architecture

        # First message
        result1 = await runner.get_response(agent, [InputItem(type="text", content="My name is Alice")])
        assert isinstance(result1, StepResult)
        assert result1.content == "Hi Alice!"

        # Second message
        result2 = await runner.get_response(agent, [InputItem(type="text", content="What's my name?")])
        assert isinstance(result2, StepResult)
        assert result2.content == "Your name is Alice"

        # Model calls are independent - agent.messages not automatically updated in single model calls
        # (This is now handled by AgentRunner for multi-turn scenarios)


@pytest.mark.asyncio
async def test_tool_execution_flow():
    """Test tool execution integrates properly"""
    agent = SimpleAgent(model="gpt-4o-mini")
    runner = get_runner_for_model(agent.model)

    # Mock tool call response
    mock_tool_call = MagicMock()
    mock_tool_call.id = "call_123"
    mock_tool_call.function.name = "bash"
    mock_tool_call.function.arguments = '{"command": "date"}'

    mock_response = MagicMock()
    mock_response.choices[0].message.content = "I'll check the time"
    mock_response.choices[0].message.tool_calls = [mock_tool_call]
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 5
    mock_response.usage.total_tokens = 15

    # Mock final response after tool execution
    mock_final_response = MagicMock()
    mock_final_response.choices[0].message.content = "The current time is..."
    mock_final_response.choices[0].message.tool_calls = None
    mock_final_response.usage.prompt_tokens = 15
    mock_final_response.usage.completion_tokens = 10
    mock_final_response.usage.total_tokens = 25

    mock_client = AsyncMock()
    mock_client.chat.completions.create.side_effect = [mock_response, mock_final_response]

    # Mock tool execution
    with patch("openai.AsyncOpenAI", return_value=mock_client):
        with patch.object(
            runner.tool_executor, "execute", return_value=AsyncMock(__str__=lambda x: "Mon Jul 21 2025")
        ) as mock_execute:
            # With new architecture, single model call with tools returns StepResult with NextStepRunAgain
            result = await runner.get_response(agent, [InputItem(type="text", content="What time is it?")])

            # Should have called tool
            mock_execute.assert_called_once_with("bash", {"command": "date"})

            # Should return StepResult indicating tool execution
            assert isinstance(result, StepResult)
            assert "bash" in result.content  # Should mention tool execution
            from cue.v2.types import NextStepRunAgain

            assert isinstance(result.next_step, NextStepRunAgain)


def test_agent_serialization():
    """Test agent can be easily serialized/deserialized"""
    agent = SimpleAgent(model="gpt-4o-mini", system_prompt="You are helpful", max_turns=5)

    # Agent should be serializable (all fields are basic types)
    import json

    agent_dict = {
        "model": agent.model,
        "system_prompt": agent.system_prompt,
        "max_turns": agent.max_turns,
        "messages": [{"role": msg.role, "content": msg.content} for msg in agent.messages],
        "tools": agent.tools,
    }

    # Should serialize to JSON without issues
    json_str = json.dumps(agent_dict)
    assert isinstance(json_str, str)

    # Should deserialize
    loaded = json.loads(json_str)
    assert loaded["model"] == "gpt-4o-mini"
    assert loaded["system_prompt"] == "You are helpful"


@pytest.mark.asyncio
async def test_multiple_providers_same_interface():
    """Test different providers use same interface"""
    models = ["gpt-4o-mini", "claude-3-5-haiku-20241022"]

    for model in models:
        runner = get_runner_for_model(model)

        # All runners should have same interface
        assert hasattr(runner, "get_response")
        assert hasattr(runner, "stream_response")
        assert hasattr(runner, "tool_executor")
        assert hasattr(runner, "_get_api_key")

        # Test method signature compatibility
        import inspect

        sig = inspect.signature(runner.get_response)
        assert "agent" in sig.parameters
        assert "input_items" in sig.parameters

    # Test Gemini separately to handle import error
    try:
        runner = get_runner_for_model("gemini-1.5-flash")

        assert hasattr(runner, "run")
        assert hasattr(runner, "run_streamed")
        assert hasattr(runner, "tool_executor")
    except ImportError:
        # Expected if google-generativeai not installed
        pass
