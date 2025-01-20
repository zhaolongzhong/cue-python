from pathlib import Path

from dotenv import load_dotenv

from .llm import ChatModel
from .tools import Tool
from .types import AgentConfig, FeatureFlag, RunMetadata, CompletionResponse
from ._agent import Agent
from ._client import AsyncCueClient
from ._version import __title__, __version__
from .utils.logs import setup_logging as _setup_logging
from ._agent_manager import AgentManager

env_path = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

__all__ = [
    "__version__",
    "__title__",
    "Agent",
    "AgentConfig",
    "AgentManager",
    "AsyncCueClient",
    "ChatModel",
    "CompletionResponse",
    "FeatureFlag",
    "RunMetadata",
    "Tool",
]

_setup_logging()
