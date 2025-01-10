import json
from unittest.mock import Mock, AsyncMock

import pytest
from anthropic.types import ToolUseBlock
from openai.types.chat import ChatCompletion, ChatCompletionMessage, ChatCompletionMessageToolCall
from openai.types.chat.chat_completion_message_tool_call import Function

from cue.tools import ToolManager
from cue.schemas import Author, AgentConfig, FeatureFlag, CompletionRequest, CompletionResponse
from cue.tools._tool import Tool
from cue.llm.llm_model import ChatModel
from cue.llm.llm_client import LLMClient, ToolResult

pytestmark = pytest.mark.unit


@pytest.fixture
def agent_config() -> AgentConfig:
    """Create test agent configuration."""
    return AgentConfig(
        id="test_agent",
        name="test_agent",
        description="Test agent",
        instruction="You are a test agent",
        model=ChatModel.GPT_4O_MINI.id,
        tools=[Tool.Edit, Tool.Memory],
        feature_flag=FeatureFlag(enable_services=False, enable_storage=False),
    )


@pytest.fixture
def llm_client(agent_config: AgentConfig) -> LLMClient:
    """Create LLM client for testing."""
    client = LLMClient(agent_config)
    return client


@pytest.mark.asyncio
async def test_llm_client_initialization(agent_config: AgentConfig):
    """Test LLM client initialization with different models."""
    # Test OpenAI client initialization
    client = LLMClient(agent_config)
    assert client.model == ChatModel.GPT_4O_MINI.id

    # Test Anthropic client initialization
    agent_config.model = ChatModel.CLAUDE_3_OPUS_20240229.id
    client = LLMClient(agent_config)
    assert client.model == ChatModel.CLAUDE_3_OPUS_20240229.id

    # Test invalid model
    agent_config.model = "invalid_model"
    with pytest.raises(ValueError):
        LLMClient(agent_config)


@pytest.mark.asyncio
async def test_send_completion_request(llm_client: LLMClient):
    """Test sending completion request."""
    request = CompletionRequest(
        messages=[{"role": "user", "content": "Test message"}],
        tools=[],
        model=ChatModel.GPT_4O_MINI.id,
    )

    # Mock OpenAI response
    chat_message = ChatCompletionMessage(content="Test response", role="assistant")
    mock_response = ChatCompletion(
        id="test_id",
        model=ChatModel.GPT_4O_MINI.id,
        object="chat.completion",
        choices=[{"finish_reason": "stop", "index": 0, "message": chat_message, "logprobs": None}],
        created=123,
        usage={"total_tokens": 10, "prompt_tokens": 5, "completion_tokens": 5},
    )

    mock_completion_response = CompletionResponse(
        msg_id="test_id",
        model=ChatModel.GPT_4O_MINI.id,
        author=Author(name="test_agent", role="assistant"),
        response=mock_response,
    )

    llm_client.llm_client.send_completion_request = AsyncMock(return_value=mock_completion_response)
    response = await llm_client.send_completion_request(request)

    assert response == mock_completion_response
    assert response.get_text() == "Test response"


@pytest.mark.asyncio
async def test_process_tools_with_timeout_openai(llm_client: LLMClient):
    """Test processing tools with timeout for OpenAI."""
    tool_manager = Mock(spec=ToolManager)
    tool_manager.has_tool = Mock(return_value=True)
    tool_manager.mcp = None

    # Create mock tool call
    tool_call = ChatCompletionMessageToolCall(
        id="test_tool_id",
        type="function",
        function=Function(name="test_tool", arguments=json.dumps({"param": "value"})),
    )

    # Create mock tool result
    mock_result = ToolResult(output="Tool output")
    tool_manager.tools = {"test_tool": AsyncMock(return_value=mock_result)}

    # Process tool calls
    result = await llm_client.process_tools_with_timeout(tool_manager=tool_manager, tool_calls=[tool_call], timeout=5)

    assert result.tool_messages is not None
    assert len(result.tool_messages) == 1
    assert result.tool_messages[0]["tool_call_id"] == "test_tool_id"
    assert result.tool_messages[0]["content"] == "Tool output"


@pytest.mark.asyncio
async def test_process_tools_with_timeout_anthropic(agent_config: AgentConfig):
    """Test processing tools with timeout for Anthropic."""
    agent_config.model = ChatModel.CLAUDE_3_OPUS_20240229.id
    llm_client = LLMClient(agent_config)

    tool_manager = Mock(spec=ToolManager)
    tool_manager.has_tool = Mock(return_value=True)
    tool_manager.mcp = None

    # Create mock tool call
    tool_call = ToolUseBlock(id="test_tool_id", name="test_tool", input={"param": "value"}, type="tool_use")

    # Create mock tool result
    mock_result = ToolResult(output="Tool output")
    tool_manager.tools = {"test_tool": AsyncMock(return_value=mock_result)}

    # Process tool calls
    result = await llm_client.process_tools_with_timeout(tool_manager=tool_manager, tool_calls=[tool_call], timeout=5)

    assert result.tool_result_message is not None
    assert result.tool_result_message["role"] == "user"
    assert isinstance(result.tool_result_message["content"], list)
    assert result.tool_result_message["content"][0]["tool_use_id"] == "test_tool_id"


@pytest.mark.asyncio
async def test_process_tools_with_timeout_error(llm_client: LLMClient):
    """Test processing tools with timeout when tool raises error."""
    tool_manager = Mock(spec=ToolManager)
    tool_manager.has_tool = Mock(return_value=True)
    tool_manager.mcp = None

    # Create mock tool call
    tool_call = ChatCompletionMessageToolCall(
        id="test_tool_id",
        type="function",
        function=Function(name="test_tool", arguments=json.dumps({"param": "value"})),
    )

    # Mock tool to raise exception
    async def mock_tool(**kwargs):
        raise Exception("Test error")

    tool_manager.tools = {"test_tool": mock_tool}

    # Process tool calls
    result = await llm_client.process_tools_with_timeout(tool_manager=tool_manager, tool_calls=[tool_call], timeout=5)

    assert result.tool_messages is not None
    assert len(result.tool_messages) == 1
    assert result.tool_messages[0]["tool_call_id"] == "test_tool_id"
    assert "Test error" in result.tool_messages[0]["content"]


@pytest.mark.asyncio
async def test_process_tools_with_timeout_not_found(llm_client: LLMClient):
    """Test processing tools with timeout when tool not found."""
    tool_manager = Mock(spec=ToolManager)
    tool_manager.has_tool = Mock(return_value=False)
    tool_manager.tools = {}
    tool_manager.mcp = None

    # Create mock tool call
    tool_call = ChatCompletionMessageToolCall(
        id="test_tool_id",
        type="function",
        function=Function(name="nonexistent_tool", arguments=json.dumps({"param": "value"})),
    )

    # Process tool calls
    result = await llm_client.process_tools_with_timeout(tool_manager=tool_manager, tool_calls=[tool_call], timeout=5)

    assert result.tool_messages is not None
    assert len(result.tool_messages) == 1
    assert result.tool_messages[0]["tool_call_id"] == "test_tool_id"
    assert "Tool 'nonexistent_tool' not found" in result.tool_messages[0]["content"]
