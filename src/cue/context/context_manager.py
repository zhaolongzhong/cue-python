import logging
from typing import Optional

from ..tools._tool import Tool, ToolManager
from ..tools.memory import MemoryTool
from .._session_context import SessionContext
from .._agent_summarizer import ContentSummarizer, create_summarizer
from ..agent.agent_state import AgentState
from ..types.agent_config import AgentConfig
from ..types.message_param import MessageParam
from ..memory.memory_manager import DynamicMemoryManager
from ..system_message_builder import SystemMessageBuilder
from ..services.service_manager import ServiceManager
from ..context.task_context_manager import TaskContextManager
from ..context.context_window_manager import ContextWindowManager
from ..context.system_context_manager import SystemContextManager
from ..context.project_context_manager import ProjectContextManager

logger = logging.getLogger(__name__)


class ContextManager:
    def __init__(
        self,
        session_context: SessionContext,
        config: AgentConfig,
        state: AgentState,
        other_agents: dict[str, dict[str, str]] = None,
        tool_manager: Optional[ToolManager] = None,
        service_manager: Optional[ServiceManager] = None,
    ):
        self.session_context = session_context
        self.tool_manager = tool_manager
        self.service_manager = service_manager
        self.summarizer: ContentSummarizer = create_summarizer(config)
        self.other_agents = other_agents
        self.config = config
        self.state = state
        self.description: Optional[str] = None
        self.initialize()

    def initialize(self):
        self.system_context_manager = SystemContextManager(
            session_context=self.session_context,
            metrics=self.state.get_metrics(),
            token_stats=self.state.get_token_stats(),
            service_manager=self.service_manager,
        )
        self.memory_manager = DynamicMemoryManager(max_tokens=1000)
        self.project_context_manager = ProjectContextManager(
            session_context=self.session_context,
            path=self.config.project_context_path,
            service_manager=self.service_manager,
        )
        self.task_context_manager = TaskContextManager(
            session_context=self.session_context,
            service_manager=self.service_manager,
        )
        self.context_window_manager = ContextWindowManager(
            model=self.config.model,
            max_tokens=self.config.max_context_tokens,
            feature_flag=self.config.feature_flag,
            summarizer=self.summarizer,
        )
        self.system_context: Optional[str] = None
        self.system_message_builder = SystemMessageBuilder(
            session_context=self.session_context,
            config=self.config,
        )
        if self.service_manager:
            self.message_storage_service = self.service_manager.message_storage_service

    def reset(self, state: AgentState):
        self.state = state
        self.initialize()

    def update_config(self, config: AgentConfig):
        """Update the agent configuration."""
        self.config = config

    def update_state(self, state: AgentState):
        """Update the agent state."""
        self.state = state

    def get_system_message(self) -> MessageParam:
        return self.system_message_builder.build()

    def build_system_context(self) -> str:
        """Build short time static system context"""
        return self.system_context_manager.build_system_context(
            project_context=self.project_context_manager.get_project_context(),
            task_context=self.task_context_manager.get_formatted_task_context(),
            memories=self.memory_manager.recent_memories,
            summaries=self.context_window_manager.get_summaries(),
        )

    def generate_description(self) -> str:
        """Return the description about the agent."""
        description = ""
        if self.config.description:
            description = self.config.description
        if not self.config.tools:
            return
        tool_names = [tool.value for tool in self.config.tools]
        if not tool_names:
            return description
        description += f" Agent {self.config.id} is able to use these tools: {', '.join(tool_names)}"
        return description

    def update_other_agents_info(self, other_agents: dict[str, dict[str, str]]) -> None:
        """Update information about other available agents.
        Args:
            other_agents: Dictionary mapping agent IDs to their info dictionaries.
                        Each info dict should contain at least a 'name' key.
        """
        # Format the info into a readable string for system messages
        self.other_agents = other_agents
        if other_agents:
            agents_list = [f"{agent_id} ({info['name']})" for agent_id, info in other_agents.items()]
            self.other_agents_info = "Available agents: " + ", ".join(agents_list)
            self.system_message_builder.set_other_agents_info(self.other_agents_info)

    async def update_context(self) -> None:
        logger.debug("Update context ...")
        await self.system_context_manager.update_base_context()
        logger.debug("Update recent memories ...")
        await self._update_recent_memories()
        logger.debug("Update project context ...")
        await self.project_context_manager.update_context()
        await self.task_context_manager.load_from_remote()
        self.summarizer.update_context(self.system_context)

    async def _update_recent_memories(self) -> Optional[str]:
        """Should be called whenever there is a memory update"""
        if not self.config.feature_flag.enable_storage or Tool.Memory not in self.config.tools:
            return None
        try:
            memory_tool: MemoryTool = self.tool_manager.tools[Tool.Memory.value]
            memory_dict = await memory_tool.get_recent_memories(assistant_id=self.session_context.assistant_id, limit=5)
            self.memory_manager.add_memories(memory_dict)
            self.memory_manager.update_recent_memories()
        except Exception as e:
            self.state.record_error(e, from_tag="update_recent_memories")
            logger.error(f"Ran into error while trying to get recent memories: {e}")
            return None
