"""Base agent implementation."""

import os
import logging
from typing import Dict, List, Union, Optional
from pathlib import Path

from pydantic import BaseModel

from .llm import LLMClient
from .agent import AgentState
from .tools import Tool, MemoryTool, ToolManager
from .utils import DebugUtils, TokenCounter, record_usage, record_usage_details
from .context import SystemContextManager, DynamicContextManager, ProjectContextManager
from .schemas import (
    Author,
    AgentConfig,
    RunMetadata,
    MessageParam,
    CompletionRequest,
    CompletionResponse,
    ConversationContext,
    ToolResponseWrapper,
    ToolCallToolUseBlock,
)
from .services import ServiceManager
from .context.message import MessageManager
from ._agent_summarizer import ContentSummarizer
from .memory.memory_manager import DynamicMemoryManager
from .system_message_builder import SystemMessageBuilder

logger = logging.getLogger(__name__)


class Agent:
    def __init__(self, config: AgentConfig):
        self.id = config.id
        self.config = config
        self.state = AgentState()
        self.tool_manager: Optional[ToolManager] = None
        self.summarizer = ContentSummarizer(
            AgentConfig(
                model="claude-3-5-haiku-20241022" if "claude" in config.model else "gpt-4o-mini",
                api_key=config.api_key,
            )
        )
        self.system_context_manager = SystemContextManager(
            metrics=self.state.get_metrics(), token_stats=self.state.get_token_stats()
        )
        self.project_context_manager = ProjectContextManager(path=self.config.project_context_path)
        self.memory_manager = DynamicMemoryManager(max_tokens=1000)
        self.context = DynamicContextManager(
            model=self.config.model,
            max_tokens=config.max_context_tokens,
            feature_flag=self.config.feature_flag,
            summarizer=self.summarizer,
        )
        self.client: LLMClient = LLMClient(self.config)
        self.metadata: Optional[RunMetadata] = None
        self.description = self._generate_description()
        self.other_agents_info = ""
        self.tool_json = None
        self.system_message_builder = SystemMessageBuilder(self.id, self.config)
        self.setup_feedback()
        self.token_counter = TokenCounter()
        self.service_manager: Optional[ServiceManager] = None
        self.message_manager: MessageManager = MessageManager()

        # Move references from state to class properties for now
        self.conversation_context: Optional[ConversationContext] = None
        self.system_context: Optional[str] = None
        self.system_message_param: Optional[str] = None

    def set_service_manager(self, service_manager: ServiceManager):
        self.service_manager = service_manager
        self.system_context_manager.set_service_manager(service_manager)
        self.message_manager.set_service_manager(service_manager)
        self.project_context_manager.set_service_manager(service_manager)

    def get_system_message(self) -> MessageParam:
        self.system_message_builder.set_conversation_context(self.conversation_context)
        self.system_message_builder.set_other_agents_info(self.other_agents_info)
        return self.system_message_builder.build()

    async def _update_recent_memories(self) -> Optional[str]:
        """Should be called whenever there is a memory update"""
        if not self.config.feature_flag.enable_storage or Tool.Memory not in self.config.tools:
            return None
        try:
            memory_tool: MemoryTool = self.tool_manager.tools[Tool.Memory.value]
            memory_dict = await memory_tool.get_recent_memories(limit=5)
            self.memory_manager.add_memories(memory_dict)
            self.memory_manager.update_recent_memories()
        except Exception as e:
            self.state.record_error(e)
            logger.error(f"Ran into error while trying to get recent memories: {e}")
            return None

    def _generate_description(self) -> str:
        """Return the description about the agent."""
        description = ""
        if self.config.description:
            description = self.config.description
        if not self.config.tools:
            return
        tool_names = [tool.value for tool in self.config.tools]
        if not tool_names:
            return description
        description += f" Agent {self.id} is able to use these tools: {', '.join(tool_names)}"
        return description

    async def _initialize(self, tool_manager: ToolManager):
        if self.state.has_initialized:
            return

        logger.debug(f"initialize ... \n{self.config.model_dump_json(indent=4)}")
        if not self.tool_manager:
            self.tool_manager = tool_manager

        await tool_manager.initialize()

        self.update_tool_json()
        self.system_message_param = self.get_system_message()
        try:
            await self.update_context()
            if self.config.feature_flag.enable_storage:
                messages = await self.message_manager.get_messages_asc(limit=10)
                if messages:
                    logger.debug(f"initial messages: {len(messages)}")
                    self.context.clear_messages()
                    await self.add_messages(messages)
        except Exception as e:
            self.state.record_error(e)
            logger.error(f"Ran into error when initialize: {e}")

        self.summarizer.update_context(self.system_context)
        self.state.has_initialized = True

    def update_tool_json(self):
        if self.config.tools:
            tools = self.config.tools.copy()

            self.tool_json = self.tool_manager.get_tool_definitions(self.config.model, tools)
            if self.config.enable_mcp:
                mcp_tools = self.tool_manager.get_mcp_tools(model=self.config.model)
                if mcp_tools:
                    self.tool_json.extend(mcp_tools)
            self.state.update_token_stats("tool", str(self.tool_json))

    async def clean_up(self):
        if self.tool_manager:
            await self.tool_manager.clean_up()

    async def add_message(self, message: Union[CompletionResponse, ToolResponseWrapper, MessageParam]):
        messages = await self.add_messages([message])
        if messages:
            return messages[0]

    async def add_messages(self, messages: List[Union[CompletionResponse, ToolResponseWrapper, MessageParam]]) -> list:
        try:
            if self.config.feature_flag.enable_storage:
                messages_with_id = []
                for message in messages.copy():
                    update_message = message
                    if not message.msg_id:
                        update_message = await self.persist_message(message)
                    else:
                        logger.debug(f"Message is already persisted: {message.msg_id}")
                    messages_with_id.append(update_message)
                messages = messages_with_id
            has_truncated_history = await self.context.add_messages(messages)
            if has_truncated_history:
                """Only update context when there are truncated messages to make the most of prompt caching"""
                try:
                    logger.debug("We have truncated messages, update context")
                    await self.update_context()
                except Exception as e:
                    self.state.record_error(e)
                    logger.debug(f"Ran into error when update context: {e}")
            self.state.update_context_stats(self.context.get_context_stats())
        except Exception as e:
            self.state.record_error(e)
            logger.error(f"Ran into error when add messages: {e}")
        return messages

    async def persist_message(
        self, message: Union[CompletionResponse, ToolResponseWrapper, MessageParam]
    ) -> Union[CompletionResponse, ToolResponseWrapper, MessageParam]:
        if not self.config.feature_flag.enable_storage:
            return message
        try:
            message_with_id = await self.message_manager.persist_message(message)
            return message_with_id
        except Exception as e:
            self.state.record_error(e)
            logger.error(f"Ran into error when persist message: {e}")
        return message

    def snapshot(self) -> str:
        """Take a snapshot of current message list and save to a file"""
        return DebugUtils.take_snapshot(self.context.messages)

    def setup_feedback(self) -> str:
        """Setup system feedback"""
        if not self.config.feedback_path:
            self.config.feedback_path = Path(
                os.path.abspath(os.path.join(os.path.dirname(__file__), "../../logs/feedbacks"))
            )
            self.config.feedback_path.mkdir(parents=True, exist_ok=True)

    async def update_context(self) -> None:
        logger.debug("update_context ...")
        await self.system_context_manager.update_base_context()
        await self._update_recent_memories()
        await self.project_context_manager.update_context()

    def build_system_context(self) -> str:
        """Build short time static system context"""
        return self.system_context_manager.build_system_context(
            project_context=self.project_context_manager.get_project_context(),
            memories=self.memory_manager.recent_memories,
            summaries=self.context.get_summaries(),
        )

    async def build_message_params(self) -> list[dict]:
        """
        Build a list of message parameter dictionaries for the completion API call.

        The method constructs the message parameters in order from most static to least static data
        to optimize prompt caching efficiency. The construction follows this sequence:
        1. Project context (static)
        2. Relevant memories (semi-static)
        3. Summary of removed messages (if any)
        4. Current message list (dynamic)
        """
        # Get message list
        messages = self.context.get_messages()
        logger.debug(f"{self.id} run message param size: {len(messages)}")
        message_params = [
            msg.model_dump() if hasattr(msg, "model_dump") else msg.dict() if hasattr(msg, "dict") else msg
            for msg in messages
        ]

        return message_params

    async def run(
        self,
        tool_manager: ToolManager,
        run_metadata: RunMetadata,
        author: Optional[Author] = None,
    ):
        try:
            self.metadata = run_metadata
            await self._initialize(tool_manager)
            message_params = await self.build_message_params()
            messages_str = str(message_params)
            self.state.update_token_stats("messages", messages_str)
            self.state.record_message()
            return await self.send_messages(messages=message_params, run_metadata=run_metadata, author=author)
        except Exception as e:
            self.state.record_error(e)
            logger.error(f"Ran into error during run: {e}")
            raise

    async def send_messages(
        self,
        messages: List[Union[BaseModel, Dict]],
        run_metadata: Optional[RunMetadata] = None,
        author: Optional[Author] = None,
    ) -> Union[CompletionResponse, ToolCallToolUseBlock]:
        if not self.metadata:
            self.metadata = run_metadata

        messages_dict = [
            msg.model_dump(exclude_none=True, exclude_unset=True) if isinstance(msg, BaseModel) else msg
            for msg in messages
        ]

        system_message_content = self.get_system_message().content
        self.state.update_token_stats("system", system_message_content)
        system_context = self.build_system_context()
        self.metadata.token_stats = self.state.get_token_stats()
        self.metadata.metrics = self.state.get_metrics()

        completion_request = CompletionRequest(
            author=Author(name=self.id, role="assistant") if not author else author,
            model=self.config.model,
            messages=messages_dict,
            metadata=self.metadata,
            tool_json=self.tool_json,
            system_prompt_suffix=system_message_content,
            system_context=system_context,
        )

        response = await self.client.send_completion_request(completion_request)
        usage_dict = record_usage(response)
        self.state.update_usage_stats(usage_dict)
        record_usage_details(self.state.get_token_stats())
        logger.debug(f"metrics: {self.state.get_metrics_json()}")
        logger.debug(f"{self.id} response: {response}")
        return response

    def build_context_for_next_agent(self, max_messages: int = 6) -> str:
        """
        Build context to be passed to the next agent, excluding the current transfer command and its result.

        Args:
            max_messages: Number of previous messages to include (0-12).
                        If 0, returns empty string as no history should be included.

        Returns:
            str: Formatted message history, excluding current transfer command and result
        """
        if max_messages == 0:
            return ""

        history = self.context.get_messages()

        # Since the last two messages are the transfer command and its result,
        # we exclude them and then take up to max_messages from the remaining history
        history_without_transfer = history[:-2] if len(history) >= 2 else []

        # Calculate how many messages to take
        start_idx = max(0, len(history_without_transfer) - max_messages)
        messages = history_without_transfer[start_idx:]

        messages_content = ",".join(str(msg) for msg in messages)

        return messages_content

    def handle_overwrite_model(self):
        override_model = self.service_manager.get_overwrite_model() if self.service_manager else None
        if override_model and override_model != self.client.model:
            self.config = self.config.model_copy()
            self.config.model = override_model
            self.client = LLMClient(self.config)
            self.update_tool_json()

            self.context = DynamicContextManager(
                model=self.config.model,
                max_tokens=self.config.max_context_tokens,
                feature_flag=self.config.feature_flag,
                summarizer=self.summarizer,
            )
