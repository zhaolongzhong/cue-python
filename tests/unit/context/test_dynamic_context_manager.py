from unittest.mock import AsyncMock, MagicMock

import pytest

from cue.utils import TokenCounter
from cue.context import DynamicContextManager
from cue.schemas import FeatureFlag, MessageParam
from cue._agent_summarizer import ContentSummarizer


@pytest.fixture
def token_counter():
    counter = TokenCounter()
    counter.count_messages_tokens = MagicMock(return_value=100)
    counter.count_dict_tokens = MagicMock(return_value=10)
    return counter


@pytest.fixture
def summarizer():
    summarizer = AsyncMock(spec=ContentSummarizer)
    summarizer.summarize = AsyncMock(return_value="Summary of removed messages")
    return summarizer


@pytest.fixture
def context_manager(token_counter, summarizer):
    manager = DynamicContextManager(model="gpt-4", max_tokens=1000, feature_flag=FeatureFlag(), summarizer=summarizer)
    manager.token_counter = token_counter
    return manager


@pytest.mark.asyncio
async def test_add_simple_message(context_manager):
    """Test adding a simple message."""
    message = MessageParam(role="user", content="Test message")

    await context_manager.add_messages([message])

    assert len(context_manager.messages) == 1
    assert context_manager.messages[0]["role"] == "user"
    assert context_manager.messages[0]["content"] == "Test message"


@pytest.mark.asyncio
@pytest.mark.skip
async def test_token_limit_enforcement(context_manager, token_counter):
    """Test that messages are removed when token limit is exceeded."""
    # Configure token counter to simulate exceeding limit
    token_counter.count_messages_tokens.return_value = 1200  # Above max_tokens=1000

    messages = [MessageParam(role="user", content=f"Message {i}") for i in range(5)]

    has_truncated = await context_manager.add_messages(messages)

    assert has_truncated
    assert len(context_manager.summaries) == 1
    assert context_manager.summaries[0] == "Summary of removed messages"


@pytest.mark.asyncio
async def test_tool_sequence_preservation(context_manager):
    """Test that tool call sequences are kept together during removal."""
    tool_call = {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": "call_123", "type": "function", "function": {"name": "test_tool", "arguments": "{}"}}],
    }

    tool_result = {"role": "tool", "content": "Tool result", "tool_call_id": "call_123"}

    # Add messages including a tool sequence
    messages = [
        MessageParam(role="user", content="Before tool"),
        tool_call,
        tool_result,
        MessageParam(role="user", content="After tool"),
    ]

    await context_manager.add_messages(messages)

    # Find indices of tool sequence
    indices = context_manager._find_tool_sequence_indices(1)  # Start at tool call

    assert len(indices) == 2  # Should include both call and result
    assert 1 in indices  # Tool call
    assert 2 in indices  # Tool result


@pytest.mark.asyncio
async def test_empty_message_handling(context_manager):
    """Test handling of empty message lists."""
    has_truncated = await context_manager.add_messages([])

    assert not has_truncated
    assert len(context_manager.messages) == 0
    assert len(context_manager.summaries) == 0


@pytest.mark.asyncio
async def test_message_preparation(context_manager):
    """Test message preparation and conversion."""
    # Test dict message
    dict_msg = {"role": "user", "content": "Dict message"}
    dict_result = context_manager._prepare_message_dict(dict_msg, "msg_1")
    assert dict_result["msg_id"] == "msg_1"
    assert dict_result["content"] == "Dict message"

    # Test MessageParam
    param_msg = MessageParam(role="user", content="Param message")
    param_result = context_manager._prepare_message_dict(param_msg, "msg_2")
    assert param_result["msg_id"] == "msg_2"
    assert param_result["content"] == "Param message"
