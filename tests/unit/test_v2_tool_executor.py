from unittest.mock import AsyncMock, patch

import pytest

from cue.v2.tool_executor import ToolExecutor, SimpleToolResult, execute_tool


@pytest.fixture
def tool_executor():
    """Create a ToolExecutor for testing"""
    return ToolExecutor()


def test_tool_executor_initialization(tool_executor):
    """Test ToolExecutor initializes with tools"""
    assert "bash" in tool_executor.tools
    assert "edit" in tool_executor.tools
    assert tool_executor.has_tool("bash")
    assert not tool_executor.has_tool("nonexistent")


def test_get_tool_schemas(tool_executor):
    """Test tool schemas are generated correctly"""
    schemas = tool_executor.get_tool_schemas()

    assert len(schemas) >= 2  # At least bash and edit

    # Check schema structure
    for schema in schemas:
        assert "type" in schema
        assert schema["type"] == "function"
        assert "function" in schema
        assert "name" in schema["function"]
        assert "description" in schema["function"]
        assert "parameters" in schema["function"]


@pytest.mark.asyncio
async def test_execute_nonexistent_tool(tool_executor):
    """Test executing a non-existent tool returns error"""
    result = await tool_executor.execute("nonexistent", {})

    assert not result.success
    assert "not found" in result.error
    assert result.output == ""


@pytest.mark.asyncio
async def test_execute_tool_with_exception(tool_executor):
    """Test tool execution handles exceptions gracefully"""
    # Mock the entire tool to raise an exception
    mock_tool = AsyncMock(side_effect=Exception("Test error"))
    tool_executor.tools["test_tool"] = mock_tool

    result = await tool_executor.execute("test_tool", {"param": "value"})

    # The exception should be caught and handled
    assert not result.success
    assert "Tool execution failed" in result.error
    assert "Test error" in result.error


def test_simple_tool_result():
    """Test SimpleToolResult creation and string representation"""
    # Success result
    result = SimpleToolResult(output="Success output", success=True)
    assert result.success
    assert str(result) == "Success output"

    # Error result
    error_result = SimpleToolResult(error="Error occurred", success=False)
    assert not error_result.success
    assert str(error_result) == "Error occurred"

    # Empty result
    empty_result = SimpleToolResult()
    assert empty_result.success
    assert str(empty_result) == "Tool completed"


@pytest.mark.asyncio
async def test_execute_tool_convenience_function():
    """Test the convenience execute_tool function"""
    # This will use a real ToolExecutor, so we'll test with a mock
    with patch("cue.v2.tool_executor.ToolExecutor") as mock_executor_class:
        mock_executor = AsyncMock()
        mock_executor.execute.return_value = SimpleToolResult(output="Test output", success=True)
        mock_executor_class.return_value = mock_executor

        result = await execute_tool("bash", command="echo test")

        assert result == "Test output"
        mock_executor.execute.assert_called_once_with("bash", {"command": "echo test"})
