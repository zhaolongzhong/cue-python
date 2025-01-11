"""Tests for agent manager."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cue.schemas import AgentConfig, RunMetadata, AgentTransfer
from cue.llm.llm_model import ChatModel
from cue._agent_manager import AgentManager
from cue.agent.agent_manager_state import AgentManagerState

pytestmark = pytest.mark.unit


@pytest.fixture
def agent_manager():
    return AgentManager()


@pytest.fixture
def mock_agent_config():
    """Create a valid AgentConfig for testing."""
    return AgentConfig(
        id="test_agent",
        name="Test Agent",
        model=ChatModel.CLAUDE_3_5_HAIKU_20241022.model_id,
        api_key="test_key",
        is_primary=True,
        tools=[],
    )


@pytest.fixture
def run_metadata():
    return RunMetadata(max_turns=5)


@pytest.mark.asyncio
async def test_agent_manager_initialization(agent_manager, mock_agent_config):
    """Test agent manager initialization."""
    # Mock LLMClient
    with patch("cue._agent.LLMClient") as mock_llm_client:
        mock_llm_client_instance = MagicMock()
        mock_llm_client.return_value = mock_llm_client_instance

        # Register an agent
        agent = agent_manager.register_agent(mock_agent_config)
        assert agent_manager.primary_agent == agent

        with patch("cue._agent_manager.ServiceManager", new_callable=AsyncMock) as mock_service_manager:
            mock_service_manager.create.return_value = AsyncMock()
            await agent_manager.initialize()

            # Check state transitions
            assert agent_manager.state_manager.state == AgentManagerState.READY
            metrics = agent_manager.get_metrics()
            assert metrics["active_agents"] == 1


@pytest.mark.asyncio
async def test_agent_registration(agent_manager, mock_agent_config):
    """Test agent registration with metrics."""
    # Mock LLMClient
    with patch("cue._agent.LLMClient") as mock_llm_client:
        mock_llm_client_instance = MagicMock()
        mock_llm_client.return_value = mock_llm_client_instance

        agent = agent_manager.register_agent(mock_agent_config)
        assert agent.config.id == "test_agent"
        assert agent_manager.primary_agent == agent

        metrics = agent_manager.get_metrics()
        assert metrics["active_agents"] == 1

        # Register same agent again
        agent2 = agent_manager.register_agent(mock_agent_config)
        assert agent2 == agent  # Should return existing agent
        assert metrics["active_agents"] == 1


@pytest.mark.asyncio
async def test_agent_transfer(agent_manager, mock_agent_config, run_metadata):
    """Test agent transfer with metrics."""
    # Mock LLMClient
    with patch("cue._agent.LLMClient") as mock_llm_client:
        mock_llm_client_instance = MagicMock()
        mock_llm_client.return_value = mock_llm_client_instance

        # Register primary agent
        primary_agent = agent_manager.register_agent(mock_agent_config)

        # Register secondary agent
        secondary_config = AgentConfig(
            id="secondary_agent",
            name="Secondary Agent",
            model=ChatModel.CLAUDE_3_5_HAIKU_20241022.model_id,
            api_key="test_key",
            is_primary=False,
            tools=[],
        )
        secondary_agent = agent_manager.register_agent(secondary_config)

        # Initialize
        with patch("cue._agent_manager.ServiceManager", new_callable=AsyncMock) as mock_service_manager:
            mock_service_manager.create.return_value = AsyncMock()
            await agent_manager.initialize()

            # Create transfer
            transfer = AgentTransfer(to_agent_id=secondary_config.id, message="test transfer", max_messages=5)

            # Set active agent
            agent_manager.active_agent = primary_agent

            # Mock build_context_for_next_agent
            primary_agent.build_context_for_next_agent = MagicMock(return_value="test context")

            # Handle transfer
            await agent_manager._handle_transfer(transfer)

            # Verify transfer was recorded
            metrics = agent_manager.get_metrics()
            assert metrics["total_transfers"] == 1
            assert metrics["successful_transfers"] == 1
            assert len(metrics["recent_transfers"]) == 1

            # Verify active agent changed
            assert agent_manager.active_agent == secondary_agent


@pytest.mark.asyncio
async def test_failed_transfer(agent_manager, mock_agent_config):
    """Test failed transfer handling."""
    # Mock LLMClient
    with patch("cue._agent.LLMClient") as mock_llm_client:
        mock_llm_client_instance = MagicMock()
        mock_llm_client.return_value = mock_llm_client_instance

        # Register only primary agent
        primary_agent = agent_manager.register_agent(mock_agent_config)
        agent_manager.active_agent = primary_agent

        # Create transfer to non-existent agent
        transfer = AgentTransfer(to_agent_id="non_existent", message="test transfer", max_messages=5)

        # Mock build_context_for_next_agent
        primary_agent.build_context_for_next_agent = MagicMock(return_value="test context")

        # Handle transfer
        await agent_manager._handle_transfer(transfer)

        # Verify failed transfer was recorded
        metrics = agent_manager.get_metrics()
        assert metrics["total_transfers"] == 1
        assert metrics["failed_transfers"] == 1
        assert metrics["successful_transfers"] == 0

        # Verify active agent unchanged
        assert agent_manager.active_agent == primary_agent


@pytest.mark.asyncio
async def test_run_execution(agent_manager, mock_agent_config, run_metadata):
    """Test run execution with state management."""
    # Mock LLMClient
    with patch("cue._agent.LLMClient") as mock_llm_client:
        mock_llm_client_instance = MagicMock()
        mock_llm_client.return_value = mock_llm_client_instance

        # Register and setup agent
        _agent = agent_manager.register_agent(mock_agent_config)

        with patch("cue._agent_manager.ServiceManager", new_callable=AsyncMock) as mock_service_manager:
            mock_service_manager.create.return_value = AsyncMock()
            await agent_manager.initialize()

            # Mock agent response
            mock_response = MagicMock()
            mock_response.get_tool_calls.return_value = []
            mock_response.get_text.return_value = "Test response"

            # Mock agent_loop run
            with patch("cue._agent_manager.AgentLoop.run", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = mock_response

                # Start run
                _response = await agent_manager.start_run(
                    active_agent_id="test_agent", message="test message", run_metadata=run_metadata
                )

                # Verify state transitions and metrics
                metrics = agent_manager.get_metrics()
                assert metrics["total_runs"] == 1
                assert agent_manager.state_manager.state == AgentManagerState.STOPPED


@pytest.mark.asyncio
async def test_error_handling(agent_manager, mock_agent_config, run_metadata):
    """Test error handling during run."""
    # Mock LLMClient
    with patch("cue._agent.LLMClient") as mock_llm_client:
        mock_llm_client_instance = MagicMock()
        mock_llm_client.return_value = mock_llm_client_instance

        # Register and setup agent
        _agent = agent_manager.register_agent(mock_agent_config)

        with patch("cue._agent_manager.ServiceManager", new_callable=AsyncMock) as mock_service_manager:
            mock_service_manager.create.return_value = AsyncMock()
            await agent_manager.initialize()

            # Mock agent_loop to raise error
            error = ValueError("test error")
            with patch("cue._agent_manager.AgentLoop.run", new_callable=AsyncMock) as mock_run:
                mock_run.side_effect = error

                # Start run
                with pytest.raises(ValueError):
                    await agent_manager.start_run(
                        active_agent_id="test_agent", message="test message", run_metadata=run_metadata
                    )

                # Verify error was recorded
                metrics = agent_manager.get_metrics()
                assert metrics["last_error"] == str(error)
                assert metrics["errors_by_type"]["ValueError"] == 1
                assert agent_manager.state_manager.state == AgentManagerState.ERROR


@pytest.mark.asyncio
async def test_cleanup(agent_manager, mock_agent_config):
    """Test cleanup with metrics."""
    # Mock LLMClient
    with patch("cue._agent.LLMClient") as mock_llm_client:
        mock_llm_client_instance = MagicMock()
        mock_llm_client.return_value = mock_llm_client_instance

        # Register agents
        agent1 = agent_manager.register_agent(mock_agent_config)

        secondary_config = AgentConfig(
            id="secondary_agent",
            name="Secondary Agent",
            model=ChatModel.CLAUDE_3_5_HAIKU_20241022.model_id,
            api_key="test_key",
            is_primary=False,
            tools=[],
        )
        agent2 = agent_manager.register_agent(secondary_config)

        # Initialize
        with patch("cue._agent_manager.ServiceManager", new_callable=AsyncMock) as mock_service_manager:
            mock_service_manager.create.return_value = AsyncMock()
            await agent_manager.initialize()

            # Add cleanup method to agents
            agent1.clean_up = AsyncMock()
            agent2.clean_up = AsyncMock()

            # Cleanup
            await agent_manager.clean_up()

            # Verify state and metrics
            assert len(agent_manager._agents) == 0
            assert agent_manager.state_manager.state == AgentManagerState.UNINITIALIZED
            metrics = agent_manager.get_metrics()
            assert metrics["active_agents"] == 0
