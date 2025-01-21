from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cue.utils.token_counter import TokenCounter
from cue.services.service_manager import ServiceManager
from cue.context.system_context_manager import SystemContextManager


@pytest.fixture
def mock_service_manager():
    service_manager = AsyncMock(spec=ServiceManager)
    assistants = AsyncMock()
    assistants.get_system_context = AsyncMock(return_value="Test system context")
    service_manager.assistants = assistants
    return service_manager


@pytest.fixture
def mock_token_counter():
    with patch("cue.context.system_context_manager.TokenCounter") as mock:
        counter_instance = MagicMock(spec=TokenCounter)
        counter_instance.count_token.return_value = 10
        mock.return_value = counter_instance
        yield counter_instance


@pytest.fixture
def metrics():
    return {
        "project": {"prev": "", "curr": "", "updated": False},
        "memories": {"prev": "", "curr": "", "updated": False},
        "summaries": {"prev": "", "curr": "", "updated": False},
    }


@pytest.fixture
def token_stats():
    return {"project": 0, "memories": 0, "summaries": 0, "context_updated": False}


@pytest.fixture
def system_context_manager(metrics, token_stats, mock_token_counter, mock_service_manager):
    return SystemContextManager(metrics, token_stats, service_manager=mock_service_manager)


@pytest.mark.asyncio
async def test_update_base_context(system_context_manager, mock_service_manager):
    """Test updating base context from service"""
    await system_context_manager.update_base_context()

    assert "Test system context" in system_context_manager.system_context_base
    assert "Time Awareness" in system_context_manager.system_context_base
    mock_service_manager.assistants.get_system_context.assert_called_once()


def test_build_system_context_new(system_context_manager):
    """Test building system context with new content"""
    system_context_manager.system_context_base = "Base context"
    context = system_context_manager.build_system_context("Project context", "Memory context", "Summary context")

    assert "Base context" in context
    assert "Project context" in context
    assert "Memory context" in context
    assert "Summary context" in context
    assert system_context_manager.metrics["context_updated"] is True
    assert system_context_manager.token_stats["context_updated"] is True


def test_build_system_context_unchanged(system_context_manager):
    """Test building system context with unchanged content"""
    system_context_manager.system_context_base = "Base context"

    # First build
    system_context_manager.build_system_context("Project context", "Memory context", "Summary context")

    # Second build with same content
    system_context_manager.build_system_context("Project context", "Memory context", "Summary context")

    assert system_context_manager.metrics["context_updated"] is False
    assert system_context_manager.token_stats["context_updated"] is False


def test_update_stats(system_context_manager):
    """Test updating stats for a context component"""
    value, tokens = system_context_manager.update_stats("test_key", "Test content", "test_context")

    assert value == "Test content"
    assert tokens == 10
    assert system_context_manager.metrics["test_key"]["curr"] == "Test content"
    assert system_context_manager.metrics["test_key"]["updated"] is True
    assert system_context_manager.token_stats["test_context"] == 10


def test_update_stats_unchanged(system_context_manager):
    """Test updating stats with unchanged content"""
    # First update
    system_context_manager.update_stats("test_key", "Test content", "test_context")

    # Second update with same content
    value, tokens = system_context_manager.update_stats("test_key", "Test content", "test_context")

    assert system_context_manager.metrics["test_key"]["updated"] is False
    assert system_context_manager.metrics["test_key"]["prev"] == "Test content"


def test_update_stats_empty(system_context_manager):
    """Test updating stats with empty content"""
    value, tokens = system_context_manager.update_stats("test_key", "", "test_context")

    assert value == ""
    assert tokens == 0
    assert "test_key" not in system_context_manager.metrics
