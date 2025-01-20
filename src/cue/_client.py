from typing import Optional

from .llm import ChatModel
from .types import AgentConfig, RunMetadata, CompletionResponse
from .config import get_settings
from .utils.logs import _logger, setup_logging
from .tools._tool import Tool
from ._agent_manager import AgentManager
from ._agent_provider import AgentProvider

setup_logging()


class AsyncCueClient:
    def __init__(self, config_file_path: Optional[str] = None):
        self.logger = _logger
        self.agent_manager = AgentManager()
        self.agents: dict[str, AgentConfig] = {}
        self.active_agent_id: Optional[str] = None
        self.run_metadata = RunMetadata()
        self.agent_provider = AgentProvider(
            config_file=config_file_path if config_file_path else get_settings().AGENTS_CONFIG_FILE
        )

    def _create_default_config(self) -> AgentConfig:
        return AgentConfig(
            id="default_agent",
            model=ChatModel.GPT_4O_MINI,
            temperature=0.8,
            max_tokens=2000,
            conversation_id="",
            tools=[
                Tool.Edit,
                Tool.Bash,
            ],
        )

    async def initialize(self, configs: Optional[list[AgentConfig]] = None):
        """Initialize the client with multiple agents."""
        self.logger.info("Initializing AsyncCueClient")

        active_agent_id = None
        if not configs:
            configs_dict = self.agent_provider.get_configs()
            configs = configs_dict.values()
            active_agent_id = self.agent_provider.get_primary_agent().id
        else:
            active_agent_id = self.agent_provider.find_primary_agent_id(configs)
            if not active_agent_id:
                active_agent_id = configs[0].id
        self.active_agent_id = active_agent_id

        for config in configs:
            agent_id = config.id
            self.agents[agent_id] = self.agent_manager.register_agent(config)

        if not self.agents:
            # Fallback to default configuration if no agents are configured
            default_config = self._create_default_config()
            self.agents[default_config.id] = self.agent_manager.register_agent(default_config)
            self.active_agent_id = default_config.id

        await self.agent_manager.initialize()
        self.logger.info(f"Initialized with {len(self.agents)} agents. Active agent: {self.active_agent_id}")

    async def send_message(self, message: str, agent_id: Optional[str] = None) -> str:
        """Send a message to a specific agent or the active agent."""
        target_agent_id = agent_id or self.active_agent_id
        if not target_agent_id or target_agent_id not in self.agents:
            raise ValueError(f"Invalid agent ID: {target_agent_id}")

        self.logger.debug(f"Sending message to agent {target_agent_id}: {message}")
        self.run_metadata.user_messages.append(message)

        response = await self.agent_manager.start_run(target_agent_id, message, self.run_metadata)

        if isinstance(response, CompletionResponse):
            return response.get_text()
        return str(response)

    def get_agent_ids(self) -> list[str]:
        """Get a list of all available agent IDs."""
        return list(self.agents.keys())

    async def cleanup(self):
        """Clean up resources used by the client."""
        self.logger.info("Cleaning up AsyncCueClient")
        await self.agent_manager.clean_up()
