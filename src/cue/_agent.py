import logging
from typing import Any, Union, Callable, Optional

from pydantic import BaseModel

from .llm import LLMClient
from .agent import AgentState
from .tools import ToolManager
from .types import (
    Author,
    AgentConfig,
    RunMetadata,
    MessageParam,
    CompletionRequest,
    CompletionResponse,
    ToolResponseWrapper,
    ToolCallToolUseBlock,
)
from .utils import DebugUtils, TokenCounter, record_usage, record_usage_details
from .services import ServiceManager
from ._session_context import SessionContext
from .context.context_manager import ContextManager
from .services.message_storage_service import MessageStorageService

logger = logging.getLogger(__name__)


class Agent:
    def __init__(self, config: AgentConfig):
        self.id = config.id
        self.config = config
        self.tools = None
        self.other_agents = {}

        self.session_context = SessionContext(assistant_id=self.id)
        self.tool_manager: Optional[ToolManager] = None
        self.service_manager: Optional[ServiceManager] = None
        self.message_storage_service: Optional[MessageStorageService] = None

        self.client: LLMClient = LLMClient(self.config)
        self.metadata: Optional[RunMetadata] = None
        self.state = AgentState()
        self.context_manager: Optional[ContextManager] = None
        self.token_counter = TokenCounter()
        self.conversation_id: Optional[str] = None
        self.streaming_callback: Optional[Callable[[CompletionResponse], Any]] = None

    def update_other_agents_info(self, other_agents: dict[str, dict[str, str]]) -> None:
        self.other_agents = other_agents

    def set_streaming_callback(self, callback: Optional[Callable[[CompletionResponse], Any]]) -> None:
        """Set callback for streaming responses (used for Claude Code)."""
        self.streaming_callback = callback

    def _get_system_message(self) -> MessageParam:
        return self.context_manager.get_system_message()

    async def initialize(self, tool_manager: ToolManager, service_manager: Optional[ServiceManager] = None):
        if self.state.has_initialized:
            return

        logger.debug(f"initialize ... \n{self.config.model_dump_json(indent=4)}")

        self.service_manager = service_manager
        if not self.tool_manager:
            self.tool_manager = tool_manager
            await tool_manager.initialize()
            self._update_tools()

        if self.service_manager:
            conversation = await service_manager.get_conversation(self.id)
            if conversation:
                self.session_context.update(conversation_id=conversation.id)
            else:
                logger.warning(f"No conversation found for agent: {self.id}")
        else:
            logger.warning(f"No service manager found for agent: {self.id}")

        self.context_manager = ContextManager(
            session_context=self.session_context,
            config=self.config,
            state=self.state,
            other_agents=self.other_agents,
            tool_manager=tool_manager,
            service_manager=self.service_manager,
        )
        try:
            await self.update_context()
            if self.config.feature_flag.enable_storage and self.message_storage_service:
                messages = await self.message_storage_service.get_messages_asc(
                    conversation_id=self.conversation_id, limit=10
                )
                if messages:
                    logger.debug(f"initial messages: {len(messages)}")
                    self.context_manager.context_window_manager.clear_messages()
                    await self.add_messages(messages)
        except Exception as e:
            self.state.record_error(e, from_tag="initialize")
            logger.error(f"Ran into error when initialize: {e}")

        self.state.has_initialized = True

    def _update_tools(self):
        if self.config.tools:
            tools = self.config.tools.copy()

            self.tools = self.tool_manager.get_tool_definitions(self.config.model, tools)
            if self.config.enable_mcp:
                mcp_tools = self.tool_manager.get_mcp_tools(model=self.config.model)
                if mcp_tools:
                    self.tools.extend(mcp_tools)
            self.state.update_token_stats("tool", str(self.tools))

    async def update_context(self) -> None:
        await self.context_manager.update_context()

    def build_system_context(self) -> str:
        """Build short time static system context"""
        return self.context_manager.build_system_context()

    async def clean_up(self):
        if self.tool_manager:
            await self.tool_manager.clean_up()

    async def add_message(
        self, message: Union[CompletionResponse, ToolResponseWrapper, MessageParam]
    ) -> Union[CompletionResponse, ToolResponseWrapper, MessageParam]:
        messages = await self.add_messages([message])
        if messages:
            return messages[0]

    async def add_messages(self, messages: list[Union[CompletionResponse, ToolResponseWrapper, MessageParam]]) -> list:
        try:
            # if self.config.feature_flag.enable_storage:
            #     messages_with_id = []
            #     for message in messages.copy():
            #         update_message = message
            #         if not message.msg_id:
            #             update_message = await self.persist_message(message)
            #         else:
            #             logger.debug(f"Message is already persisted: {message.msg_id}")
            #         messages_with_id.append(update_message)
            #     messages = messages_with_id
            has_truncated_history = await self.context_manager.context_window_manager.add_messages(messages)
            if has_truncated_history:
                """Only update context when there are truncated messages to make the most of prompt caching"""
                try:
                    logger.debug("We have truncated messages, update context")
                    await self.update_context()
                except Exception as e:
                    self.state.record_error(e, from_tag="add_messages")
                    logger.debug(f"Ran into error when update context: {e}")
            self.state.update_context_stats(self.context_manager.context_window_manager.get_context_stats())
        except Exception as e:
            self.state.record_error(e, from_tag="add_messages")
            logger.error(f"Ran into error when add messages: {e}")
        return messages

    async def persist_message(
        self, message: Union[CompletionResponse, ToolResponseWrapper, MessageParam]
    ) -> Union[CompletionResponse, ToolResponseWrapper, MessageParam]:
        if not self.config.feature_flag.enable_storage or not self.message_storage_service:
            return message
        try:
            message_with_id = await self.message_storage_service.persist_message(
                conversation_id=self.conversation_id,
                message=message,
            )
            return message_with_id
        except Exception as e:
            self.state.record_error(e)
            logger.error(f"Ran into error when persist message: {e}")
        return message

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
        messages = self.context_manager.context_window_manager.get_messages()
        logger.debug(f"{self.id} run message param size: {len(messages)}")
        message_params = [
            msg.model_dump() if hasattr(msg, "model_dump") else msg.dict() if hasattr(msg, "dict") else msg
            for msg in messages
        ]

        return message_params

    async def run(
        self,
        run_metadata: RunMetadata,
        author: Optional[Author] = None,
    ) -> Union[CompletionResponse, ToolCallToolUseBlock]:
        try:
            self.metadata = run_metadata
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
        messages: list[Union[BaseModel, dict]],
        run_metadata: Optional[RunMetadata] = None,
        author: Optional[Author] = None,
    ) -> Union[CompletionResponse, ToolCallToolUseBlock]:
        if not self.metadata:
            self.metadata = run_metadata

        messages_dict = [
            msg.model_dump(exclude_none=True, exclude_unset=True) if isinstance(msg, BaseModel) else msg
            for msg in messages
        ]

        system_message_content = self._get_system_message().content
        self.state.update_token_stats("system", system_message_content)
        system_context = self.build_system_context()
        self.metadata.token_stats = self.state.get_token_stats()
        self.metadata.metrics = self.state.get_metrics()

        completion_request = CompletionRequest(
            author=Author(name=self.id, role="assistant") if not author else author,
            model=self.config.model,
            messages=messages_dict,
            metadata=self.metadata,
            tools=self.tools,
            system_prompt_suffix=system_message_content,
            system_context=system_context,
        )

        if self.config.streaming:
            # Handle streaming responses with callback
            response = None
            async for streaming_response in self.client.send_streaming_completion_request(completion_request):
                response = streaming_response
                # Call streaming callback if available
                if self.streaming_callback:
                    await self.streaming_callback(streaming_response)
            if response is None:
                raise RuntimeError("No response received from Claude Code streaming")
        else:
            response = await self.client.send_completion_request(completion_request)
        usage_dict = record_usage(response)
        self.state.update_usage_stats(usage_dict)
        record_usage_details(self.state.get_token_stats())
        logger.debug(f"metrics: {self.state.get_metrics_json()}")
        logger.debug(f"{self.id} response: {response}")
        return response

    def build_context_for_next_agent(self, max_messages: int = 6) -> str:
        return self.context_manager.context_window_manager.build_context_for_next_agent(max_messages=max_messages)

    async def handle_overwrite_config(self):
        if not self.service_manager:
            return
        override_config: AgentConfig = await self.service_manager.get_agent_config(assistant_id=self.id)
        if not override_config:
            return
        self.config = self.config.model_copy()
        if override_config.model != self.client.model or (
            override_config.max_turns > 0 and override_config.max_turns != self.config.max_turns
        ):
            self.config.model = override_config.model
            self.config.max_turns = override_config.max_turns
            self.client = LLMClient(self.config)
            self._update_tools()
            self.context_manager.reset(self.state)

    async def reset_state(self):
        logger.debug(f"Resetting state for agent {self.id}")
        self.state = AgentState()
        await self.handle_overwrite_config()

    def snapshot(self) -> str:
        """Take a snapshot of current message list and save to a file"""
        return DebugUtils.take_snapshot(self.context_manager.context_window_manager.messages)
