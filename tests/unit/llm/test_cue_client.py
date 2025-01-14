import json
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from cue.schemas import AgentConfig, FeatureFlag, ErrorResponse, CompletionRequest, CompletionResponse
from cue.llm.cue_client import CueClient


class MockSettings:
    def get_base_url(self):
        return "http://mock-url"


@pytest.fixture(autouse=True)
def mock_settings():
    """Mock settings for all tests."""
    with patch("cue.llm.cue_client.get_settings", return_value=MockSettings()):
        yield


@pytest.fixture
def config():
    """Create test agent configuration."""
    return AgentConfig(
        id="test_agent",
        name="test_agent",
        description="Test agent",
        instruction="You are a test agent",
        model="gpt-4",
        api_key="test_key",
        tools=[],
        feature_flag=FeatureFlag(enable_services=False, enable_storage=False),
    )


@pytest.fixture
def client(config, mock_settings):
    """Create a CueClient for testing."""
    return CueClient(config=config)


def test_extract_json_dict(client):
    """Test JSON extraction from various string formats."""
    # Test with code fence
    json_str = """```json
    {
        "name": "test",
        "value": 123
    }
    ```"""
    is_json, result = client.extract_json_dict(json_str)
    assert is_json
    assert result == {"name": "test", "value": 123}

    # Test without code fence
    json_str = '{"name": "test", "value": 123}'
    is_json, result = client.extract_json_dict(json_str)
    assert is_json
    assert result == {"name": "test", "value": 123}

    # Test with invalid JSON
    json_str = "not a json"
    is_json, result = client.extract_json_dict(json_str)
    assert not is_json
    assert result is None


def test_convert_tool_call(client):
    """Test conversion of JSON response to tool call format."""
    # Test conversion with valid tool dict
    response_data = {"choices": [{"message": {"content": "test"}}]}
    tool_dict = {"name": "test_tool", "arguments": {"param": "value"}}

    with patch("cue.llm.cue_client.generate_id", return_value="call_test"):
        result = client.convert_tool_call(response_data, tool_dict)

        assert "tool_calls" in result["choices"][0]["message"]
        tool_call = result["choices"][0]["message"]["tool_calls"][0]
        assert tool_call["type"] == "function"
        assert tool_call["id"] == "call_test"
        assert tool_call["function"]["name"] == "test_tool"
        assert json.loads(tool_call["function"]["arguments"]) == {"param": "value"}


def test_replace_tool_call_ids(client):
    """Test replacement of tool call IDs and cleanup of tool names."""
    response_data = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {"id": "old_id", "type": "function", "function": {"name": "test.tool", "arguments": "{}"}}
                    ]
                }
            }
        ]
    }

    with patch("cue.llm.cue_client.generate_id", return_value="call_test"):
        client.replace_tool_call_ids(response_data, "gpt-4")

        tool_call = response_data["choices"][0]["message"]["tool_calls"][0]
        assert tool_call["id"] == "call_test"  # New ID format
        assert tool_call["function"]["name"] == "testtool"  # Dots removed


@pytest.mark.asyncio
async def test_completion_request_with_tool_calls_o1(config):
    """Test completion request handling with tool calls for o1 models."""
    config.model = "gpt-4-o1"
    client = CueClient(config=config)

    mock_response = httpx.Response(
        status_code=200,
        json={
            "choices": [
                {
                    "message": {
                        "content": """```json
                    {
                        "name": "test_tool",
                        "arguments": {"param": "value"}
                    }
                    ```"""
                    }
                }
            ]
        },
    )

    request = CompletionRequest(messages=[{"role": "user", "content": "test"}])
    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client_class.return_value.__aenter__.return_value = mock_client
        response = await client.send_completion_request(request)

    assert isinstance(response, CompletionResponse)
    assert "tool_calls" in response.response["choices"][0]["message"]


@pytest.mark.asyncio
async def test_completion_request_with_tool_calls_regular(client):
    """Test completion request handling with tool calls for regular models."""
    mock_response = httpx.Response(
        status_code=200,
        json={
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {"id": "test_id", "type": "function", "function": {"name": "test.tool", "arguments": "{}"}}
                        ]
                    }
                }
            ]
        },
    )

    request = CompletionRequest(messages=[{"role": "user", "content": "test"}])
    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client_class.return_value.__aenter__.return_value = mock_client
        response = await client.send_completion_request(request)

    assert isinstance(response, CompletionResponse)
    tool_call = response.response["choices"][0]["message"]["tool_calls"][0]
    assert tool_call["id"].startswith("call_")
    assert tool_call["function"]["name"] == "testtool"


@pytest.mark.asyncio
async def test_completion_request_error_handling(client):
    """Test error handling in completion requests."""
    request = CompletionRequest(messages=[{"role": "user", "content": "test"}])
    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=httpx.RequestError("Connection failed"))
        mock_client_class.return_value.__aenter__.return_value = mock_client
        response = await client.send_completion_request(request)

    assert isinstance(response, CompletionResponse)
    assert isinstance(response.error, ErrorResponse)
    assert "Request failed" in response.error.message


@pytest.mark.asyncio
async def test_completion_request_invalid_status(client):
    """Test handling of invalid status codes in completion requests."""
    mock_response = httpx.Response(status_code=400, json={"error": "Bad request"})
    request = CompletionRequest(messages=[{"role": "user", "content": "test"}])

    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client_class.return_value.__aenter__.return_value = mock_client
        response = await client.send_completion_request(request)

    assert isinstance(response, CompletionResponse)
    assert isinstance(response.error, ErrorResponse)
    assert "400" in response.error.code
