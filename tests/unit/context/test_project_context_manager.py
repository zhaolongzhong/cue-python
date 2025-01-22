import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cue.utils.token_counter import TokenCounter
from cue.services.service_manager import ServiceManager
from cue.context.project_context_manager import ProjectContextManager


@pytest.fixture
def mock_service_manager():
    service_manager = AsyncMock(spec=ServiceManager)
    assistants = AsyncMock()
    assistants.get_project_context = AsyncMock(return_value="Test project context")
    service_manager.assistants = assistants
    return service_manager


@pytest.fixture
def mock_token_counter():
    with patch("cue.context.project_context_manager.TokenCounter") as mock:
        counter_instance = MagicMock(spec=TokenCounter)
        counter_instance.count_token.return_value = 10
        mock.return_value = counter_instance
        yield counter_instance


@pytest.fixture
def project_context_manager(session_context, mock_service_manager):
    return ProjectContextManager(
        session_context=session_context,
        path="/test/path/context.txt",
        service_manager=mock_service_manager,
    )


logger = logging.getLogger(__name__)


@pytest.mark.asyncio
async def test_update_context_from_service(project_context_manager, mock_service_manager):
    """Test updating context from service manager"""
    await project_context_manager.update_context()

    assert (
        project_context_manager.get_project_context() == "Test project context"
    )  # https://platform.openai.com/tokenizer
    assert project_context_manager.get_params()["content"].endswith("<project_context_token>3</project_context_token>")
    mock_service_manager.assistants.get_project_context.assert_called_once()


@pytest.mark.asyncio
async def test_update_context_from_file(project_context_manager, tmp_path):
    """Test updating context from file"""
    project_context_manager.service_manager = None
    test_content = "Test file content"
    test_file = tmp_path / "context.txt"
    test_file.write_text(test_content)

    project_context_manager.path = str(test_file)
    await project_context_manager.update_context()

    assert project_context_manager.get_project_context() == test_content
    assert project_context_manager.get_params()["content"].endswith("<project_context_token>3</project_context_token>")


@pytest.mark.asyncio
async def test_update_context_no_path(session_context):
    """Test updating context with no path"""
    manager = ProjectContextManager(session_context=session_context, path=None)
    await manager.update_context()

    assert manager.get_project_context() is None
    assert manager.get_params() is None


@pytest.mark.asyncio
async def test_update_context_nonexistent_file(project_context_manager):
    """Test updating context with nonexistent file"""
    project_context_manager.path = "/nonexistent/path/context.txt"
    project_context_manager.service_manager = None
    await project_context_manager.update_context()

    assert project_context_manager.get_project_context() is None
    assert project_context_manager.get_params() is None


@pytest.mark.asyncio
async def test_context_overwrite_empty(project_context_manager, tmp_path):
    """Test handling of context being overwritten with empty content"""
    project_context_manager.service_manager = None
    # First write content
    test_file = tmp_path / "context.txt"
    test_file.write_text("Initial content")

    project_context_manager.path = str(test_file)
    await project_context_manager.update_context()

    # Then overwrite with empty
    test_file.write_text("")
    await project_context_manager.update_context()

    params = project_context_manager.get_params()
    assert "has been overwritten with empty" in params["content"]
    assert "<pre_project_context>Initial content</pre_project_context>" in params["content"]
