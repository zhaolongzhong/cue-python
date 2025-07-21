#!/usr/bin/env python3
"""
Test AgentRunner - Multi-turn conversation orchestration

Tests the AgentRunner class that coordinates conversations
using ModelBase for individual model responses.
"""

from unittest.mock import AsyncMock

import pytest

from cue.v2.types import InputItem, RunResult, StepResult, SimpleAgent, NextStepFinalOutput
from cue.v2.model_base import ModelBase
from cue.v2.agent_runner import AgentRunner
from cue.v2.streaming_hooks import StreamEvent


class MockModel(ModelBase):
    """Mock model for testing AgentRunner"""

    def __init__(self, api_key=None):
        super().__init__(api_key)
        self.get_response_mock = AsyncMock()
        self.stream_response_mock = AsyncMock()

    async def get_response(self, agent, input_items):
        return await self.get_response_mock(agent, input_items)

    async def stream_response(self, agent, input_items, hooks=None):
        async for event in self.stream_response_mock(agent, input_items, hooks):
            yield event


class TestAgentRunner:
    """Test AgentRunner functionality"""

    @pytest.fixture
    def mock_model(self):
        """Create mock model"""
        return MockModel()

    @pytest.fixture
    def agent_runner(self, mock_model):
        """Create AgentRunner with mock model"""
        return AgentRunner(mock_model)

    @pytest.fixture
    def agent(self):
        """Create test agent"""
        return SimpleAgent(model="test-model", system_prompt="Test agent")

    @pytest.mark.asyncio
    async def test_agent_runner_initialization(self, mock_model):
        """Test AgentRunner initializes correctly"""
        runner = AgentRunner(mock_model)
        assert runner.model is mock_model

    @pytest.mark.asyncio
    async def test_get_response_delegates_to_model(self, agent_runner, mock_model, agent):
        """Test get_response delegates to underlying model"""
        # Setup - mock returns StepResult, runner returns RunResult
        expected_step = StepResult(content="Test response", usage={"tokens": 10}, next_step=NextStepFinalOutput())
        mock_model.get_response_mock.return_value = expected_step

        input_items = [InputItem(type="text", content="Hello")]

        # Execute - use new method name
        result = await agent_runner.run(agent, input_items)

        # Verify - result should be RunResult with steps
        assert isinstance(result, RunResult)
        assert result.content == "Test response"
        assert len(result.steps) == 1
        assert result.steps[0] == expected_step
        mock_model.get_response_mock.assert_called_once_with(agent, input_items)

    @pytest.mark.asyncio
    async def test_cache_token_accumulation(self, agent_runner, mock_model, agent):
        """Test that cache tokens are properly accumulated across turns"""
        # Setup - mock returns StepResult with cache tokens
        expected_step = StepResult(
            content="Test response",
            usage={
                "input_tokens": 50,
                "output_tokens": 25,
                "total_tokens": 75,
                "cache_creation_input_tokens": 1000,
                "cache_read_input_tokens": 500,
            },
            next_step=NextStepFinalOutput(),
        )
        mock_model.get_response_mock.return_value = expected_step

        input_items = [InputItem(type="text", content="Hello")]

        # Execute
        result = await agent_runner.run(agent, input_items)

        # Verify cache tokens are accumulated
        assert isinstance(result, RunResult)
        assert result.usage["cache_creation_input_tokens"] == 1000
        assert result.usage["cache_read_input_tokens"] == 500
        assert result.usage["input_tokens"] == 50
        assert result.usage["output_tokens"] == 25
        assert result.usage["total_tokens"] == 75

    @pytest.mark.asyncio
    async def test_stream_response_delegates_to_model(self, agent_runner, mock_model, agent):
        """Test stream_response delegates to underlying model"""
        # Setup
        expected_events = [
            StreamEvent(type="text", content="Hello"),
            StreamEvent(type="text", content=" world"),
            StreamEvent(type="agent_done", content="Hello world"),
        ]

        async def mock_stream(*args, **kwargs):
            for event in expected_events:
                yield event

        mock_model.stream_response_mock = mock_stream

        input_items = [InputItem(type="text", content="Hello")]

        # Execute - use new method name
        events = []
        async for event in agent_runner.run_streamed(agent, input_items):
            events.append(event)

        # Verify
        assert len(events) == 3
        assert events[0].content == "Hello"
        assert events[1].content == " world"
        assert events[2].content == "Hello world"
        assert events[2].type == "agent_done"

    def test_agent_runner_has_future_methods_documented(self, agent_runner):
        """Test AgentRunner has documentation for future methods"""
        # Verify the class has proper documentation for extensibility
        doc = AgentRunner.__doc__
        assert "multi-turn" in doc.lower()
        assert "conversation" in doc.lower()
        assert "future" in doc.lower() or "enhance" in doc.lower()

    def test_agent_runner_imports_correctly(self):
        """Test AgentRunner can be imported from separate module"""
        from cue.v2 import AgentRunner as AgentRunnerFromInit
        from cue.v2.agent_runner import AgentRunner

        # Both imports should work and be the same class
        assert AgentRunner is AgentRunnerFromInit


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
