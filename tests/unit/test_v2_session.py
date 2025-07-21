#!/usr/bin/env python3
"""
Test Session implementations and AgentRunner with Session support

Tests both InMemorySession and the session-based AgentRunner functionality.
"""

from unittest.mock import AsyncMock

import pytest

from cue.v2.types import InputItem, RunResult, StepResult, SimpleAgent, NextStepFinalOutput
from cue.v2.model_base import ModelBase
from cue.v2.agent_runner import AgentRunner
from cue.v2.memory_session import InMemorySession
from cue.v2.streaming_hooks import StreamEvent


class MockModel(ModelBase):
    """Mock model for testing"""

    def __init__(self, api_key=None):
        super().__init__(api_key)
        self.get_response_mock = AsyncMock()
        self.stream_response_mock = AsyncMock()

    async def get_response(self, agent, input_items):
        return await self.get_response_mock(agent, input_items)

    async def stream_response(self, agent, input_items, hooks=None):
        async for event in self.stream_response_mock(agent, input_items, hooks):
            yield event


class TestInMemorySession:
    """Test InMemorySession functionality"""

    @pytest.fixture
    def session(self):
        """Create empty InMemorySession"""
        return InMemorySession()

    @pytest.fixture
    def session_with_id(self):
        """Create InMemorySession with specific ID"""
        return InMemorySession(session_id="test-session-123")

    @pytest.fixture
    def sample_items(self):
        """Create sample input items"""
        return [
            InputItem(type="text", content="Hello"),
            InputItem(type="text", content="How are you?"),
            InputItem(type="text", content="I'm fine, thanks!"),
        ]

    @pytest.mark.asyncio
    async def test_session_initialization(self, session, session_with_id):
        """Test session initializes correctly"""
        # Default session should have UUID
        assert session.session_id
        assert len(session.session_id) > 0

        # Custom session should use provided ID
        assert session_with_id.session_id == "test-session-123"

        # Both should start empty
        assert len(session) == 0
        assert len(session_with_id) == 0

    @pytest.mark.asyncio
    async def test_add_and_get_items(self, session, sample_items):
        """Test adding and retrieving items"""
        # Start empty
        items = await session.get_items()
        assert items == []

        # Add items
        await session.add_items(sample_items[:2])
        assert len(session) == 2

        # Get all items
        items = await session.get_items()
        assert len(items) == 2
        assert items[0].content == "Hello"
        assert items[1].content == "How are you?"

        # Add more items
        await session.add_items([sample_items[2]])
        assert len(session) == 3

        items = await session.get_items()
        assert len(items) == 3
        assert items[2].content == "I'm fine, thanks!"

    @pytest.mark.asyncio
    async def test_get_items_with_limit(self, session, sample_items):
        """Test retrieving items with limit"""
        await session.add_items(sample_items)

        # Get latest 2 items
        items = await session.get_items(limit=2)
        assert len(items) == 2
        assert items[0].content == "How are you?"
        assert items[1].content == "I'm fine, thanks!"

        # Get latest 1 item
        items = await session.get_items(limit=1)
        assert len(items) == 1
        assert items[0].content == "I'm fine, thanks!"

        # Get more than available
        items = await session.get_items(limit=10)
        assert len(items) == 3

        # Limit of 0 or negative
        items = await session.get_items(limit=0)
        assert items == []

        items = await session.get_items(limit=-1)
        assert items == []

    @pytest.mark.asyncio
    async def test_pop_item(self, session, sample_items):
        """Test popping items from session"""
        # Pop from empty session
        item = await session.pop_item()
        assert item is None

        # Add items and pop
        await session.add_items(sample_items)
        assert len(session) == 3

        # Pop latest item
        item = await session.pop_item()
        assert item.content == "I'm fine, thanks!"
        assert len(session) == 2

        # Pop another
        item = await session.pop_item()
        assert item.content == "How are you?"
        assert len(session) == 1

        # Pop last
        item = await session.pop_item()
        assert item.content == "Hello"
        assert len(session) == 0

        # Pop from empty again
        item = await session.pop_item()
        assert item is None

    @pytest.mark.asyncio
    async def test_clear_session(self, session, sample_items):
        """Test clearing session"""
        await session.add_items(sample_items)
        assert len(session) == 3

        await session.clear_session()
        assert len(session) == 0

        items = await session.get_items()
        assert items == []

    def test_session_repr(self, session):
        """Test string representation"""
        repr_str = repr(session)
        assert "InMemorySession" in repr_str
        assert session.session_id in repr_str
        assert "items=0" in repr_str


class TestAgentRunnerWithSession:
    """Test AgentRunner with Session support"""

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

    @pytest.fixture
    def session(self):
        """Create test session"""
        return InMemorySession(session_id="test-session")

    @pytest.mark.asyncio
    async def test_get_response_with_session_new_input(self, agent_runner, mock_model, agent, session):
        """Test get_response with session and new input"""
        # Setup mock - mock returns StepResult, runner returns RunResult
        expected_step = StepResult(content="Hello there!", usage={"tokens": 5}, next_step=NextStepFinalOutput())
        mock_model.get_response_mock.return_value = expected_step

        # Execute with new input - use new method name
        result = await agent_runner.run(agent, session, "Hello")

        # Verify result - should be RunResult with steps
        assert isinstance(result, RunResult)
        assert result.content == "Hello there!"
        assert len(result.steps) == 1
        assert result.steps[0] == expected_step

        # Verify session was updated
        items = await session.get_items()
        assert len(items) == 2  # user input + assistant response
        assert items[0].content == "Hello"
        assert items[1].content == "Hello there!"

        # Verify mock was called with correct input
        mock_model.get_response_mock.assert_called_once()
        call_args = mock_model.get_response_mock.call_args
        assert call_args[0][0] == agent
        assert len(call_args[0][1]) == 1  # Just the user input
        assert call_args[0][1][0].content == "Hello"

    @pytest.mark.asyncio
    async def test_get_response_with_session_existing_history(self, agent_runner, mock_model, agent, session):
        """Test get_response with session containing history"""
        # Add existing conversation to session
        await session.add_items(
            [
                InputItem(type="text", content="Previous user message"),
                InputItem(type="text", content="Previous assistant response"),
            ]
        )

        # Setup mock - mock returns StepResult, runner returns RunResult
        expected_step = StepResult(content="New response", usage={"tokens": 3}, next_step=NextStepFinalOutput())
        mock_model.get_response_mock.return_value = expected_step

        # Execute with new input
        result = await agent_runner.run(agent, session, "New message")

        # Verify result - should be RunResult with steps
        assert isinstance(result, RunResult)
        assert result.content == "New response"
        assert len(result.steps) == 1
        assert result.steps[0] == expected_step

        # Verify session was updated
        items = await session.get_items()
        assert len(items) == 4  # previous history + new user + new assistant
        assert items[2].content == "New message"
        assert items[3].content == "New response"

        # Verify mock was called with full history
        mock_model.get_response_mock.assert_called_once()
        call_args = mock_model.get_response_mock.call_args
        input_items = call_args[0][1]
        assert len(input_items) == 3  # previous history + new input
        assert input_items[0].content == "Previous user message"
        assert input_items[1].content == "Previous assistant response"
        assert input_items[2].content == "New message"

    @pytest.mark.asyncio
    async def test_get_response_with_session_no_input(self, agent_runner, mock_model, agent, session):
        """Test get_response with session but no new input"""
        # Add existing conversation to session
        await session.add_items(
            [
                InputItem(type="text", content="Existing message"),
            ]
        )

        # Setup mock - mock returns StepResult, runner returns RunResult
        expected_step = StepResult(content="Response to existing", usage={"tokens": 4}, next_step=NextStepFinalOutput())
        mock_model.get_response_mock.return_value = expected_step

        # Execute without new input
        result = await agent_runner.run(agent, session)

        # Verify result - should be RunResult with steps
        assert isinstance(result, RunResult)
        assert result.content == "Response to existing"
        assert len(result.steps) == 1
        assert result.steps[0] == expected_step

        # Verify session was updated with only assistant response
        items = await session.get_items()
        assert len(items) == 2
        assert items[0].content == "Existing message"
        assert items[1].content == "Response to existing"

    @pytest.mark.asyncio
    async def test_stream_response_with_session(self, agent_runner, mock_model, agent, session):
        """Test stream_response with session"""
        # Setup mock streaming
        expected_events = [
            StreamEvent(type="text", content="Hello"),
            StreamEvent(type="text", content=" there"),
            StreamEvent(type="text", content="!"),
            StreamEvent(type="agent_done", content="Hello there!"),
        ]

        async def mock_stream(*args, **kwargs):
            for event in expected_events:
                yield event

        mock_model.stream_response_mock = mock_stream

        # Execute
        events = []
        async for event in agent_runner.run_streamed(agent, session, "Hi"):
            events.append(event)

        # Verify events - with new multi-turn streaming, we get step boundaries too
        assert len(events) == 6  # step_start + 4 content events + conversation_done
        assert events[0].type == "step_start"
        assert events[1].content == "Hello"
        assert events[2].content == " there"
        assert events[3].content == "!"
        assert events[4].content == "Hello there!"
        assert events[5].type == "conversation_done"
        assert events[5].content == "Hello there!"

        # Verify session was updated
        items = await session.get_items()
        assert len(items) == 2  # user input + accumulated assistant response
        assert items[0].content == "Hi"
        assert items[1].content == "Hello there!"  # accumulated text content

    @pytest.mark.asyncio
    async def test_backward_compatibility_with_input_items(self, agent_runner, mock_model, agent):
        """Test that AgentRunner still works with List[InputItem] for backward compatibility"""
        # Setup - mock returns StepResult, runner returns RunResult
        expected_step = StepResult(content="Legacy response", usage={"tokens": 2}, next_step=NextStepFinalOutput())
        mock_model.get_response_mock.return_value = expected_step

        input_items = [InputItem(type="text", content="Legacy input")]

        # Execute with legacy API
        result = await agent_runner.run(agent, input_items)

        # Verify - result should be RunResult with steps
        assert isinstance(result, RunResult)
        assert result.content == "Legacy response"
        assert len(result.steps) == 1
        assert result.steps[0] == expected_step
        mock_model.get_response_mock.assert_called_once_with(agent, input_items)

    @pytest.mark.asyncio
    async def test_backward_compatibility_streaming(self, agent_runner, mock_model, agent):
        """Test that streaming works with List[InputItem] for backward compatibility"""
        # Setup mock streaming
        expected_events = [
            StreamEvent(type="text", content="Legacy"),
            StreamEvent(type="text", content=" stream"),
        ]

        async def mock_stream(*args, **kwargs):
            for event in expected_events:
                yield event

        mock_model.stream_response_mock = mock_stream

        input_items = [InputItem(type="text", content="Legacy input")]

        # Execute
        events = []
        async for event in agent_runner.run_streamed(agent, input_items):
            events.append(event)

        # Verify
        assert len(events) == 2
        assert events[0].content == "Legacy"
        assert events[1].content == " stream"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
