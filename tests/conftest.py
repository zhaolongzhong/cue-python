import logging
from unittest.mock import Mock

import pytest

from cue.utils.logs import setup_logging
from cue.tools._tool import ToolManager
from cue.llm.llm_model import ChatModel
from cue._session_context import SessionContext

# Global configuration dictionary
test_config = {"default_model": ChatModel.GPT_4O_MINI}


@pytest.fixture(scope="session")
def default_chat_model() -> str:
    """Fixture to provide the default ChatModel for tests."""
    return test_config["default_model"].id


@pytest.fixture(scope="session", autouse=True)
def configure_logging_fixture():
    setup_logging()
    yield
    # Teardown: Close all handlers to release file resources
    logger = logging.getLogger()
    handlers = logger.handlers[:]
    for handler in handlers:
        handler.close()
        logger.removeHandler(handler)


# Function to change the default model (can be called from tests if needed)
def set_default_model(model: str) -> None:
    """Set the default model for tests."""
    test_config["default_model"] = model


# @pytest.fixture(scope="session")
# def event_loop() -> Iterator[asyncio.AbstractEventLoop]:
#     loop = asyncio.new_event_loop()
#     asyncio.set_event_loop(loop)
#     yield loop
#     loop.close()


@pytest.fixture
def session_context():
    return SessionContext(assistant_id="asst_123", conversation_id="conv_123")


@pytest.fixture
def tool_manager() -> ToolManager:
    """Create a mock tool manager for testing."""
    return Mock(spec=ToolManager)
