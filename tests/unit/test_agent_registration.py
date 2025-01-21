from unittest.mock import Mock, AsyncMock

import pytest

from cue.types import AgentConfig, FeatureFlag
from cue._agent import Agent
from cue.tools._tool import Tool, ToolManager
from cue.llm.llm_model import ChatModel
from cue._agent_manager import AgentManager
from cue.services.service_manager import ServiceManager


@pytest.fixture
def agent_configs() -> dict[str, AgentConfig]:
    """Create test agent configurations."""
    return {
        "main": AgentConfig(
            id="main",
            name="main",
            is_primary=True,
            description="Primary agent for testing",
            instruction="You are the primary test agent",
            model=ChatModel.GPT_4O_MINI.id,
            tools=[Tool.Edit, Tool.Memory],
            feature_flag=FeatureFlag(enable_services=False),
        ),
        "helper": AgentConfig(
            id="helper",
            name="helper",
            description="Helper agent for testing",
            instruction="You are a helper test agent",
            model=ChatModel.GPT_4O_MINI.id,
            tools=[Tool.Edit],
        ),
    }


@pytest.fixture
def mock_tool_manager() -> Mock:
    """Create a mock tool manager."""
    manager = Mock(spec=ToolManager)
    manager.initialize = AsyncMock()
    manager.clean_up = AsyncMock()
    manager.get_tool_definitions = Mock(return_value=[{"name": "edit", "description": "Edit tool for testing"}])
    return manager


@pytest.fixture
def mock_service_manager():
    service_manager = AsyncMock(spec=ServiceManager)
    assistants = AsyncMock()
    assistants.get_system_context = AsyncMock(return_value="Test system context")
    service_manager.assistants = assistants
    service_manager.messages = AsyncMock()
    service_manager.message_storage_service = Mock()
    return service_manager


@pytest.fixture
def agent_manager():
    """Create a test agent manager."""
    return AgentManager()


@pytest.mark.asyncio
async def test_register_agent(agent_manager, agent_configs):
    """Test basic agent registration."""
    main_config = agent_configs["main"]
    agent = agent_manager.register_agent(main_config)

    assert isinstance(agent, Agent)
    assert agent.config.id == "main"
    assert agent.config.is_primary
    assert agent_manager.primary_agent == agent
    assert len(agent.config.tools) == 2
    assert Tool.Edit in agent.config.tools
    assert Tool.Memory in agent.config.tools

    await agent_manager.clean_up()


@pytest.mark.asyncio
async def test_register_duplicate_agent(agent_manager, agent_configs):
    """Test registering an agent with duplicate ID."""
    main_config = agent_configs["main"]
    first_agent = agent_manager.register_agent(main_config)
    second_agent = agent_manager.register_agent(main_config)

    assert first_agent == second_agent
    assert len(agent_manager._agents) == 1

    await agent_manager.clean_up()


@pytest.mark.asyncio
async def test_primary_agent_selection(agent_manager, agent_configs, mock_tool_manager, mock_service_manager):
    """Test primary agent selection and other agents info update."""
    main_config = agent_configs["main"]
    helper_config = agent_configs["helper"]

    # Register agents
    main_agent = agent_manager.register_agent(main_config)
    helper_agent = agent_manager.register_agent(helper_config)
    await main_agent._initialize(tool_manager=mock_tool_manager, service_manager=mock_service_manager)
    await helper_agent._initialize(tool_manager=mock_tool_manager, service_manager=mock_service_manager)

    # Test primary agent is set
    assert agent_manager.primary_agent == main_agent

    # Update other agents info
    agent_manager._update_other_agents_info()

    # Check primary agent knows about helper
    assert len(main_agent.context_manager.other_agents) == 1
    assert "helper" in main_agent.context_manager.other_agents

    # Check helper agent knows about primary
    assert isinstance(helper_agent.context_manager.other_agents, dict)
    assert "main" in helper_agent.context_manager.other_agents

    await agent_manager.clean_up()


@pytest.mark.asyncio
async def test_agent_cleanup(agent_manager, agent_configs):
    """Test agent cleanup process."""
    main_config = agent_configs["main"]
    helper_config = agent_configs["helper"]

    # Register agents
    agent_manager.register_agent(main_config)
    agent_manager.register_agent(helper_config)
    assert len(agent_manager._agents) == 2

    # Clean up
    await agent_manager.clean_up()
    assert len(agent_manager._agents) == 0
    assert agent_manager.primary_agent is None
