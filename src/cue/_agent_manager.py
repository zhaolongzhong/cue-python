import asyncio
import logging
from typing import Any, Dict, List, Union, Callable, Optional

from .utils import DebugUtils, console_utils
from ._agent import Agent
from .schemas import (
    AgentConfig,
    RunMetadata,
    MessageParam,
    AgentTransfer,
    CompletionResponse,
    ConversationContext,
)
from .services import ServiceManager
from ._agent_loop import AgentLoop
from .tools._tool import ToolManager, MCPServerManager
from .schemas.feature_flag import FeatureFlag
from .schemas.event_message import (
    EventMessage,
    MessagePayload,
    EventMessageType,
)
from .agent.agent_manager_state import AgentManagerState, AgentManagerStateManager

logger = logging.getLogger(__name__)


class AgentManager:
    def __init__(
        self,
        prompt_callback=None,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ):
        logger.info("AgentManager initialized")
        self.prompt_callback = prompt_callback
        self.loop = loop or asyncio.get_event_loop()
        self.agent_loop = AgentLoop()
        self._agents: Dict[str, Agent] = {}
        self.active_agent: Optional[Agent] = None
        self.primary_agent: Optional[Agent] = None
        self.service_manager: Optional[ServiceManager] = None
        self.tool_manager: Optional[ToolManager] = None
        self.run_metadata: Optional[RunMetadata] = None
        self.console_utils = console_utils
        self.user_message_queue: asyncio.Queue[str] = asyncio.Queue()
        self.execute_run_task: Optional[asyncio.Task] = None
        self.stop_run_event: asyncio.Event = asyncio.Event()
        self.mcp: MCPServerManager = None
        self.state_manager = AgentManagerStateManager()

    async def on_message_received(self, message: EventMessage) -> None:
        """Handle incoming messages from services."""
        logger.debug(f"on_message_received: {message}")
        if not self.active_agent:
            logger.warning("No active agent to handle message")
            return

        if message.type == EventMessageType.USER:
            await self.user_message_queue.put(message.payload.text)
        else:
            await self.active_agent.add_message(
                MessageParam(
                    role="assistant" if message.type == EventMessageType.ASSISTANT else "user",
                    content=message.payload.text,
                    name=message.payload.name,
                )
            )

    async def initialize(
        self, run_metadata: Optional[RunMetadata] = RunMetadata(), mcp: Optional[MCPServerManager] = None
    ):
        """Initialize the agent manager with proper state management."""
        try:
            if not self.state_manager.can_initialize():
                raise RuntimeError(f"Cannot initialize in {self.state_manager.state} state")

            self.state_manager.start_initialization()
            logger.debug(f"initialize mode: {run_metadata.mode}")
            self.run_metadata = run_metadata
            self.mcp = mcp

            if self.primary_agent:
                feature_flag = self.primary_agent.config.feature_flag
            else:
                feature_flag = FeatureFlag()

            if feature_flag.enable_services or run_metadata.mode in "client":
                self.service_manager = await ServiceManager.create(
                    run_metadata=self.run_metadata,
                    feature_flag=feature_flag,
                    on_message_received=self.on_message_received,
                    agent=self.primary_agent.config,
                )
                await self.service_manager.connect()

            self._update_other_agents_info()
            await self.initialize_run()

            self.state_manager.complete_initialization()
            self.state_manager.metrics.update_active_agents(len(self._agents))
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            self.state_manager.complete_initialization(error=e)
            raise

    async def clean_up(self):
        """Clean up resources with state tracking."""
        self.state_manager.state = AgentManagerState.CLEANING
        try:
            if self.service_manager:
                await self.service_manager.close()

            cleanup_tasks = []
            for agent_id in list(self._agents.keys()):
                try:
                    if hasattr(self._agents[agent_id], "clean_up"):
                        cleanup_tasks.append(asyncio.create_task(self._agents[agent_id].clean_up()))
                except Exception as e:
                    logger.error(f"Error cleaning up agent {agent_id}: {e}")
                    self.state_manager.metrics.record_error(e)

            if cleanup_tasks:
                await asyncio.gather(*cleanup_tasks, return_exceptions=True)

            self._agents.clear()
            self.primary_agent = None
            self.active_agent = None
            self.state_manager.metrics.update_active_agents(0)
            logger.info("All agents cleaned up and removed.")
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            self.state_manager.metrics.record_error(e)
            raise
        finally:
            self.state_manager.state = AgentManagerState.UNINITIALIZED

    async def start_run(
        self,
        active_agent_id: str,
        message: str,
        run_metadata: RunMetadata,
        callback: Optional[Callable[[Union[CompletionResponse, MessageParam]], Any]] = None,
    ) -> Optional[Union[CompletionResponse, MessageParam]]:
        """Start a run with state management and metrics."""
        try:
            if not self.state_manager.can_start_run():
                raise RuntimeError(f"Cannot start run in {self.state_manager.state} state")

            self.state_manager.start_run()

            if run_metadata:
                self.run_metadata = run_metadata
            self.run_metadata.current_turn = 0

            if active_agent_id not in self._agents:
                error = ValueError(f"Agent {active_agent_id} not found")
                self.state_manager.stop_run(error=error)  # Record this initialization error
                raise error

            self.active_agent = self._agents[active_agent_id]
            self.active_agent.set_service_manager(self.service_manager)

            if not callback:
                callback = self.handle_response

            if message:
                user_message = MessageParam(role="user", content=message, model=self.active_agent.config.model)
                message = await self.active_agent.add_message(user_message)
                await callback(message)

            logger.debug(f"run - queued message for agent {active_agent_id}: {message}")

            if self.run_metadata.mode == "runner":
                if not self.execute_run_task or self.execute_run_task.done():
                    self.execute_run_task = asyncio.create_task(self._execute_run(callback))
                return None

            try:
                return await self._execute_run(callback)
            except Exception as e:
                logger.error(f"Failed to start run: {e}")
                # Don't record error here since it's already recorded in _execute_run
                raise
        except Exception as e:
            logger.error(f"Failed to start run: {e}")
            if not isinstance(e, ValueError):  # Only record if it's not already recorded
                self.state_manager.stop_run(error=e)
            raise

    async def _execute_run(
        self,
        callback: Optional[Callable[[CompletionResponse], Any]] = None,
    ) -> Optional[CompletionResponse]:
        """Execute run with improved error handling and metrics."""
        logger.debug(f"execute_run loop started. run_metadata: {self.run_metadata.model_dump_json(indent=4)}")
        try:
            while True:
                if self.state_manager.should_stop():
                    logger.info("Run stopping due to state change")
                    break

                response = await self.agent_loop.run(
                    agent=self.active_agent,
                    run_metadata=self.run_metadata,
                    callback=callback,
                    prompt_callback=self.prompt_callback,
                    tool_manager=self.tool_manager,
                )

                if isinstance(response, AgentTransfer):
                    if response.run_metadata:
                        logger.debug(
                            f"handle transfer to {response.to_agent_id}, run metadata: {self.run_metadata.model_dump_json(indent=4)}"
                        )
                    await self._handle_transfer(response)
                    continue

                self.state_manager.stop_run()
                return response
        except Exception as e:
            logger.error(f"Run execution error: {e}")
            self.state_manager.stop_run(error=e)
            raise

    async def _handle_transfer(self, agent_transfer: AgentTransfer) -> None:
        """Handle agent transfer with metrics and validation."""
        try:
            from_agent_id = self.active_agent.id
            to_agent_id = agent_transfer.to_agent_id

            if agent_transfer.transfer_to_primary:
                to_agent_id = self.primary_agent.id
                agent_transfer.to_agent_id = to_agent_id

            if to_agent_id not in self._agents:
                error_msg = (
                    f"Target agent '{to_agent_id}' not found. Available agents: {', '.join(self._agents.keys())}"
                )
                logger.error(error_msg)
                self.state_manager.metrics.record_transfer(
                    from_agent=from_agent_id, to_agent=to_agent_id, success=False, error=error_msg
                )
                await self.active_agent.add_message(MessageParam(role="user", content=error_msg))
                return

            messages = []
            agent_transfer.context = self.active_agent.build_context_for_next_agent(
                max_messages=agent_transfer.max_messages
            )

            if agent_transfer.context:
                context_message = MessageParam(
                    role="assistant",
                    content=f"Here is context from {from_agent_id} <background>{agent_transfer.context}</background>",
                )
                messages.append(context_message)

            transfer_message = MessageParam(role="assistant", content=agent_transfer.message, name=from_agent_id)
            messages.append(transfer_message)

            self.active_agent = self._agents[to_agent_id]
            await self.active_agent.add_messages(messages)

            DebugUtils.take_snapshot(
                messages=messages,
                suffix=f"transfer_to_{self.active_agent.id}_{self.active_agent.config.model}",
                with_timestamp=True,
                subfolder="transfer",
            )

            self.active_agent.conversation_context = ConversationContext(participants=[from_agent_id, to_agent_id])

            self.state_manager.metrics.record_transfer(from_agent=from_agent_id, to_agent=to_agent_id, success=True)
        except Exception as e:
            logger.error(f"Transfer error: {e}")
            self.state_manager.metrics.record_transfer(
                from_agent=from_agent_id, to_agent=to_agent_id, success=False, error=str(e)
            )
            raise

    def register_agent(self, config: AgentConfig) -> Agent:
        """Register agent with metrics tracking."""
        if config.id in self._agents:
            logger.warning(f"Agent with id {config.id} already exists, returning existing agent.")
            return self._agents[config.id]

        agent = Agent(config=config)
        self._agents[agent.config.id] = agent
        logger.info(
            f"register_agent {agent.config.id} (name: {config.id}), tool: {config.tools} available agents: {list(self._agents.keys())}"
        )

        if config.is_primary:
            self.primary_agent = agent

        self.state_manager.metrics.update_active_agents(len(self._agents))
        return agent

    def get_metrics(self) -> dict:
        """Get current metrics."""
        return self.state_manager.get_metrics()

    def _update_other_agents_info(self):
        """Update each agent with information about other available agents."""
        for agent in self._agents.values():
            other_agents = {
                id: {"name": other.config.name} for id, other in self._agents.items() if id != agent.config.id
            }
            agent.update_other_agents_info(other_agents)

    def get_agent(self, identifier: str) -> Optional[Agent]:
        if identifier in self._agents:
            return self._agents[identifier]
        for agent in self._agents.values():
            if agent.config.id == identifier:
                return agent

    def list_agents(self, exclude: List[str] = []) -> List[dict[str, str]]:
        return [
            {"id": agent.id, "description": agent.description}
            for agent in self._agents.values()
            if agent.id not in exclude
        ]

    async def initialize_run(self):
        """Initialize run state."""
        self.stop_run_event.clear()
        if not self.run_metadata:
            self.run_metadata = RunMetadata()

    async def handle_response(self, response: Union[CompletionResponse, MessageParam]) -> None:
        """Handle agent response during run."""
        if not response:
            return

        # Update to handle both MessageParam and CompletionResponse
        if self.service_manager:
            text_content = response.content if isinstance(response, MessageParam) else response.text
            if text_content:
                event_message = EventMessage(
                    type=EventMessageType.ASSISTANT,
                    payload=MessagePayload(text=text_content),
                )
                await self.service_manager.send_message(event_message)
