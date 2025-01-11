from typing import Dict

import pytest

from cue._agent import Agent
from cue.schemas import AgentConfig, FeatureFlag
from cue.tools._tool import Tool
from cue.llm.llm_model import ChatModel
from cue._agent_manager import AgentManager


@pytest.fixture
def agent_configs() -> Dict[str, AgentConfig]:
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
async def test_get_agent(agent_manager, agent_configs):
    """Test retrieving registered agents."""
    main_config = agent_configs["main"]
    helper_config = agent_configs["helper"]

    main_agent = agent_manager.register_agent(main_config)
    helper_agent = agent_manager.register_agent(helper_config)

    # Test get by ID
    assert agent_manager.get_agent("main") == main_agent
    assert agent_manager.get_agent("helper") == helper_agent

    # Test get non-existent agent
    with pytest.raises(Exception, match="Agent 'unknown' not found"):
        agent_manager.get_agent("unknown")

    await agent_manager.clean_up()


@pytest.mark.asyncio
async def test_list_agents(agent_manager, agent_configs):
    """Test listing registered agents."""
    main_config = agent_configs["main"]
    helper_config = agent_configs["helper"]

    agent_manager.register_agent(main_config)
    agent_manager.register_agent(helper_config)

    # List all agents
    agents = agent_manager.list_agents()
    assert len(agents) == 2
    assert any(a["id"] == "main" for a in agents)
    assert any(a["id"] == "helper" for a in agents)

    # List with exclusion
    filtered_agents = agent_manager.list_agents(exclude=["main"])
    assert len(filtered_agents) == 1
    assert filtered_agents[0]["id"] == "helper"

    await agent_manager.clean_up()


@pytest.mark.asyncio
async def test_primary_agent_selection(agent_manager, agent_configs):
    """Test primary agent selection and other agents info update."""
    main_config = agent_configs["main"]
    helper_config = agent_configs["helper"]

    # Register agents
    main_agent = agent_manager.register_agent(main_config)
    helper_agent = agent_manager.register_agent(helper_config)

    # Test primary agent is set
    assert agent_manager.primary_agent == main_agent

    # Update other agents info
    agent_manager._update_other_agents_info()

    # Check primary agent knows about helper
    assert len(main_agent.other_agents) == 1
    assert "helper" in main_agent.other_agents

    # Check helper agent knows about primary
    assert isinstance(helper_agent.other_agents, dict)
    assert "main" in helper_agent.other_agents

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
