from unittest.mock import MagicMock

import pytest

from cue.v2.types import InputItem, SimpleAgent
from cue.v2.openai_model import OpenAIModel
from cue.v2.anthropic_model import AnthropicModel


@pytest.mark.asyncio
async def test_openai_model_api_key_handling():
    """Test OpenAI runner handles API keys correctly"""
    # Test with explicit API key
    runner = OpenAIModel(api_key="test-key")
    agent = SimpleAgent(model="gpt-4o-mini")

    key = runner._get_api_key(agent, "OPENAI_API_KEY")
    assert key == "test-key"

    # Test with agent API key
    agent_with_key = SimpleAgent(model="gpt-4o-mini", api_key="agent-key")
    key = runner._get_api_key(agent_with_key, "OPENAI_API_KEY")
    assert key == "agent-key"


@pytest.mark.asyncio
async def test_openai_model_message_formatting():
    """Test OpenAI runner formats messages correctly"""
    runner = OpenAIModel()
    agent = SimpleAgent(
        model="gpt-4o-mini",
        system_prompt="You are helpful",
    )
    agent.messages = [MagicMock(role="user", content="Hello"), MagicMock(role="assistant", content="Hi there")]

    input_items = [InputItem(type="text", content="How are you?")]

    messages = runner._build_messages(agent, input_items)

    # Should have system + history + new input
    assert len(messages) >= 3
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "You are helpful"
    assert messages[-1]["role"] == "user"
    assert messages[-1]["content"] == "How are you?"


@pytest.mark.asyncio
async def test_anthropic_model_message_formatting():
    """Test Anthropic runner formats messages correctly (no system in messages)"""
    runner = AnthropicModel()
    agent = SimpleAgent(
        model="claude-3-5-haiku-20241022",
        system_prompt="You are helpful",
    )
    agent.messages = [
        MagicMock(role="system", content="System prompt"),
        MagicMock(role="user", content="Hello"),
        MagicMock(role="assistant", content="Hi there"),
    ]

    input_items = [InputItem(type="text", content="How are you?")]

    messages = runner._build_messages(agent, input_items)

    # Should not include system messages in Anthropic format
    for msg in messages:
        assert msg["role"] != "system"

    # Should have user and assistant messages + new input
    assert messages[-1]["role"] == "user"

    # Content should be in list format with cache control due to prompt caching injection
    content = messages[-1]["content"]
    assert isinstance(content, list)
    assert len(content) == 1
    assert content[0]["type"] == "text"
    assert content[0]["text"] == "How are you?"
    assert "cache_control" in content[0]


@pytest.mark.asyncio
async def test_runner_tool_executor_integration():
    """Test runners properly integrate with ToolExecutor"""
    runner = OpenAIModel()

    # Should have tool executor
    assert hasattr(runner, "tool_executor")
    assert runner.tool_executor is not None

    # Should get tool schemas
    schemas = runner.tool_executor.get_tool_schemas()
    assert len(schemas) >= 1
    assert all(schema.get("type") == "function" for schema in schemas)
