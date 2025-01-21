from unittest.mock import AsyncMock

import pytest

from cue.tools._tool import Tool
from cue.agent.agent_state import AgentState
from cue.types.agent_config import AgentConfig
from cue.context.context_manager import ContextManager


@pytest.fixture
def basic_config():
    """Create a basic AgentConfig for testing"""
    return AgentConfig(
        id="test-agent",
        name="Test Agent",
        client_id="test-client",
        model="gpt-4",
        project_context_path="/tmp",
        tools=[Tool.Memory, Tool.Edit],
    )


@pytest.fixture
def context_manager(basic_config):
    """Create a ContextManager instance with basic config"""
    state = AgentState()
    return ContextManager(config=basic_config, state=state)


def test_context_manager_initialization(context_manager):
    """Test that ContextManager initializes with all required components"""
    assert context_manager.config.id == "test-agent"
    assert context_manager.system_context_manager is not None
    assert context_manager.memory_manager is not None
    assert context_manager.project_context_manager is not None
    assert context_manager.task_context_manager is not None
    assert context_manager.context_window_manager is not None


def test_generate_description(context_manager):
    """Test description generation with tools"""
    description = context_manager.generate_description()
    assert "test-agent" in description
    assert "memory" in description
    assert "edit" in description


def test_generate_description_no_tools():
    """Test description generation without tools"""
    config = AgentConfig(
        id="test-agent",
        name="Test Agent",
        client_id="test-client",
        model="gpt-4",
        tools=[],
        project_context_path="/tmp",
    )
    state = AgentState()
    manager = ContextManager(config=config, state=state)
    description = manager.generate_description()
    assert description is None


def test_update_other_agents_info(context_manager):
    """Test updating information about other agents"""
    other_agents = {"agent1": {"name": "First Agent"}, "agent2": {"name": "Second Agent"}}
    context_manager.update_other_agents_info(other_agents)
    assert "agent1" in context_manager.other_agents
    assert "First Agent" in context_manager.other_agents_info
    assert "Second Agent" in context_manager.other_agents_info


@pytest.mark.asyncio
async def test_update_context(context_manager):
    """Test the context update process"""
    # Mock the async methods using patch
    from unittest.mock import patch

    async def mock_coro():
        pass

    with (
        patch.object(context_manager.system_context_manager, "update_base_context", new=AsyncMock()) as mock_base,
        patch.object(context_manager.project_context_manager, "update_context", new=AsyncMock()) as mock_project,
        patch.object(context_manager.task_context_manager, "load_from_remote", new=AsyncMock()) as mock_task,
    ):
        await context_manager.update_context()

        # Verify all update methods were called
        mock_base.assert_called_once()
        mock_project.assert_called_once()
        mock_task.assert_called_once()


def test_reset_state(context_manager):
    """Test resetting the context manager state"""
    new_state = AgentState()
    context_manager.reset(new_state)

    assert context_manager.state == new_state
    assert context_manager.system_context_manager is not None
    assert context_manager.memory_manager is not None
    assert context_manager.context_window_manager is not None


@pytest.mark.asyncio
async def test_update_recent_memories_no_storage(context_manager):
    """Test memory update when storage is disabled"""
    context_manager.config.feature_flag.enable_storage = False
    result = await context_manager._update_recent_memories()
    assert result is None
