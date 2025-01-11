from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cue._agent_summarizer import ContentSummarizer
from cue.schemas.agent_config import AgentConfig
from cue.schemas.completion_request import CompletionRequest
from cue.schemas.completion_response import CompletionResponse


@pytest.fixture
def mock_llm_client():
    with patch("cue._agent_summarizer.LLMClient") as mock:
        client = AsyncMock()
        mock.return_value = client
        yield client


@pytest.fixture
def mock_token_counter():
    with patch("cue._agent_summarizer.TokenCounter") as mock:
        counter = MagicMock()
        mock.return_value = counter
        # Set up common token counts
        counter.count_token.return_value = 10  # System context tokens
        counter.count_dict_tokens.return_value = 20  # Message tokens
        yield counter


@pytest.fixture
def mock_debug_utils():
    with patch("cue._agent_summarizer.DebugUtils") as mock:
        yield mock


@pytest.fixture
def summarizer(mock_llm_client, mock_token_counter):
    config = AgentConfig(model="test-model", api_key="test-key", organization="test-org")
    return ContentSummarizer(config)


@pytest.fixture
def mock_completion_response():
    response = AsyncMock(spec=CompletionResponse)
    response.get_text.return_value = "Summarized content"
    response.model = "test-model"
    response.get_id.return_value = "test-id-123"
    usage = MagicMock()
    usage.model_dump = lambda **kwargs: {"prompt_tokens": 30, "completion_tokens": 10, "total_tokens": 40}
    response.get_usage.return_value = usage
    return response


@pytest.mark.asyncio
async def test_update_context(summarizer):
    """Test context update functionality"""
    test_context = "Test system context"
    summarizer.update_context(test_context)
    assert summarizer.system_context == test_context


@pytest.mark.asyncio
async def test_summarize_messages(summarizer, mock_llm_client, mock_completion_response):
    """Test message summarization"""
    mock_llm_client.send_completion_request.return_value = mock_completion_response

    messages = [{"role": "user", "content": "Test message 1"}, {"role": "assistant", "content": "Test response 1"}]

    result = await summarizer.summarize("test-model", messages)

    assert result == "Summarized content"
    mock_llm_client.send_completion_request.assert_called_once()
    request = mock_llm_client.send_completion_request.call_args[1]["request"]
    assert isinstance(request, CompletionRequest)
    assert request.model == "test-model"


@pytest.mark.asyncio
async def test_summarize_text(summarizer, mock_llm_client, mock_completion_response):
    """Test text summarization"""
    mock_llm_client.send_completion_request.return_value = mock_completion_response

    test_text = "Test content to summarize"
    result = await summarizer.summarize_text(test_text)

    assert result == "Summarized content"
    mock_llm_client.send_completion_request.assert_called_once()


@pytest.mark.asyncio
async def test_summarize_with_custom_instruction(summarizer, mock_llm_client, mock_completion_response):
    """Test summarization with custom instruction"""
    mock_llm_client.send_completion_request.return_value = mock_completion_response

    test_text = "Test content"
    custom_instruction = "Focus on key points"
    result = await summarizer.summarize_text(test_text, instruction=custom_instruction)

    assert result == "Summarized content"
    request = mock_llm_client.send_completion_request.call_args[1]["request"]
    assert custom_instruction in request.messages[0]["content"]


@pytest.mark.asyncio
async def test_summarize_error_handling(summarizer, mock_llm_client):
    """Test error handling during summarization"""
    mock_llm_client.send_completion_request.side_effect = Exception("Test error")

    result = await summarizer.summarize_text("Test content")
    assert result is None


@pytest.mark.asyncio
async def test_token_counting(summarizer, mock_token_counter, mock_llm_client, mock_completion_response):
    """Test token counting functionality"""
    mock_llm_client.send_completion_request.return_value = mock_completion_response
    summarizer.system_context = "Test context"

    await summarizer.summarize_text("Test content")

    # Verify token counting calls
    mock_token_counter.count_token.assert_called_with(summarizer.system_context)
    mock_token_counter.count_dict_tokens.assert_called_once()


@pytest.mark.asyncio
async def test_metrics_recording(summarizer, mock_llm_client, mock_completion_response, mock_debug_utils):
    """Test metrics recording functionality"""
    mock_llm_client.send_completion_request.return_value = mock_completion_response
    summarizer.system_context = "Test context"

    await summarizer.summarize_text("Test content")

    # Verify debug snapshot was taken
    mock_debug_utils.take_snapshot.assert_called_once()
    metrics = mock_debug_utils.take_snapshot.call_args[1]["messages"][0]

    # Verify metrics structure
    assert "timestamp" in metrics
    assert "model" in metrics
    assert "token_stats" in metrics
    assert "system_context" in metrics
    assert "message" in metrics
    assert "summary" in metrics


@pytest.mark.asyncio
async def test_no_system_context(summarizer, mock_llm_client, mock_completion_response):
    """Test summarization without system context"""
    mock_llm_client.send_completion_request.return_value = mock_completion_response

    result = await summarizer.summarize_text("Test content")

    assert result == "Summarized content"
    request = mock_llm_client.send_completion_request.call_args[1]["request"]
    assert request.system_prompt_suffix is None


@pytest.mark.asyncio
async def test_temperature_setting(summarizer, mock_llm_client, mock_completion_response):
    """Test temperature setting in requests"""
    mock_llm_client.send_completion_request.return_value = mock_completion_response

    await summarizer.summarize_text("Test content")

    request = mock_llm_client.send_completion_request.call_args[1]["request"]
    assert request.temperature == 0.4  # Check default temperature


@pytest.mark.asyncio
async def test_content_wrapping(summarizer, mock_llm_client, mock_completion_response):
    """Test content wrapping in XML tags"""
    mock_llm_client.send_completion_request.return_value = mock_completion_response

    test_content = "Test content"
    await summarizer.summarize_text(test_content)

    request = mock_llm_client.send_completion_request.call_args[1]["request"]
    assert f"<content>{test_content}</content>" in request.messages[0]["content"]
