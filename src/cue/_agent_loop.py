import asyncio
import logging
from typing import Any, Union, Callable, Optional

from .utils import DebugUtils
from ._agent import Agent
from .schemas import (
    Author,
    ErrorType,
    RunMetadata,
    MessageParam,
    AgentTransfer,
    CompletionResponse,
    ToolResponseWrapper,
)
from .tools._tool import ToolManager
from .services.monitoring_client import MonitoringClient

logger = logging.getLogger(__name__)


class AgentLoop:
    def __init__(self):
        self.stop_run_event: asyncio.Event = asyncio.Event()
        self.user_message_queue: asyncio.Queue[str] = asyncio.Queue()
        self.execute_run_task: Optional[asyncio.Task] = None

    async def run(
        self,
        agent: Agent,
        tool_manager: ToolManager,
        run_metadata: RunMetadata,
        callback: Optional[Callable[[CompletionResponse], Any]] = None,
        prompt_callback: Optional[Callable] = None,
        monitoring: Optional[MonitoringClient] = None,
    ) -> Optional[Union[CompletionResponse, AgentTransfer]]:
        """
        Run the agent execution loop.

        Args:
            agent: The agent to run
            run_metadata: Metadata for the run
            callback: Callback for processing responses
            prompt_callback: Callback for user prompts
        """
        logger.debug(f"Agent run loop started. {agent.id}")
        response = None
        author = Author(role="user", name="")

        while True:
            if self.stop_run_event.is_set():
                logger.info("Stop signal received. Exiting execute_run loop.")
                break

            # Process queued messages
            while not self.user_message_queue.empty():
                try:
                    new_message = self.user_message_queue.get_nowait()
                    logger.debug(f"Received new user message during run: {new_message}")
                    new_user_message = MessageParam(role="user", content=new_message)
                    message = await agent.add_message(new_user_message)
                    if callback and message:
                        await callback(response)
                except asyncio.QueueEmpty:
                    break

            run_metadata.current_turn += 1
            if not await self._should_continue_run(run_metadata, prompt_callback):
                break

            try:
                response: CompletionResponse = await agent.run(
                    tool_manager=tool_manager,
                    run_metadata=run_metadata,
                    author=author,
                )
            except asyncio.CancelledError:
                logger.info("agent.run was cancelled.")
                break
            except Exception as e:
                logger.exception(f"Error during agent run: {e}")
                if monitoring:
                    await monitoring.report_exception(
                        e, error_type=ErrorType.SYSTEM, additional_context={"component": "_agent_loop"}
                    )
                break

            author = Author(role="assistant", name="")
            if not isinstance(response, CompletionResponse):
                if response.error:
                    logger.error(response.error.model_dump())
                    message = await agent.add_message(response)
                    if callback and message:
                        await callback(response)
                    continue
                else:
                    raise Exception(f"Unexpected response: {response}")

            tool_calls = response.get_tool_calls()
            if not tool_calls:
                message = await agent.add_message(response)
                if callback and message:
                    await callback(message)
                if agent.config.is_primary:
                    DebugUtils.log_chat({"assistant": response.get_text()}, "agent_loop")
                    return response
                else:
                    # Handle transfer to primary agent
                    logger.info("Auto switch to primary agent")
                    transfer = AgentTransfer(
                        message=response.get_text(),
                        transfer_to_primary=True,
                        run_metadata=run_metadata,
                    )
                    return transfer

            text_content = response.get_text()
            if text_content:
                DebugUtils.log_chat({"assistant": text_content}, "agent_loop")

            if agent.config.feature_flag.enable_storage:
                # persist tool use message and update msg id
                persisted_message = await agent.persist_message(response)
                if persisted_message:
                    response.msg_id = persisted_message.msg_id

            if callback and response:
                await callback(response)

            tool_result = await agent.client.process_tools_with_timeout(
                tool_manager=tool_manager,
                tool_calls=tool_calls,
                timeout=60,
                author=response.author,
            )

            if isinstance(tool_result, ToolResponseWrapper):
                # Add tool call and tool result pair
                messages = await agent.add_messages([response, tool_result])
                if callback and messages:
                    await callback(messages[-1])
                # Handle explicit transfer request
                agent_transfer = tool_result.agent_transfer
                if agent_transfer:
                    agent_transfer.run_metadata = run_metadata
                    return tool_result.agent_transfer

                if tool_result.base64_images:
                    logger.debug("Add base64_image result to message params")
                    tool_result_content = {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{tool_result.base64_images[0]}"},
                    }
                    contents = [
                        {"type": "text", "text": "Please check previous query info related to this image"},
                        tool_result_content,
                    ]
                    message_param = MessageParam(role="user", content=contents, model=tool_result.model)
                    await agent.add_message(message_param)
            else:
                raise Exception(f"Unexpected response: {tool_result}")

        logger.info("Exiting execute_run loop.")
        return response

    async def stop(self):
        """Signal the execute_run loop to stop gracefully."""
        if self.execute_run_task and not self.execute_run_task.done():
            logger.info("Stopping the ongoing run...")
            self.stop_run_event.set()
            try:
                await self.execute_run_task
            except asyncio.CancelledError:
                logger.info("stop execute_run_task was cancelled.")
            finally:
                self.execute_run_task = None
                self.stop_run_event.clear()
                logger.info("Run has been stopped.")
        else:
            logger.info("No active run to stop.")

    async def _should_continue_run(self, run_metadata: RunMetadata, prompt_callback: Optional[Callable] = None) -> bool:
        """Determines if the loop should continue based on turn count and user input."""
        logger.debug(f"Maximum turn {run_metadata.max_turns}, current: {run_metadata.current_turn - 1}")
        if run_metadata.enable_turn_debug and prompt_callback:
            user_response = await prompt_callback(
                f"Maximum turn {run_metadata.max_turns}, current: {run_metadata.current_turn - 1}. Debug. Continue? (y/n, press Enter to continue): "
            )
            if user_response.lower() not in ["y", "yes", ""]:
                logger.warning("Stopped by user.")
                return False

        if run_metadata.current_turn > run_metadata.max_turns:
            if prompt_callback:
                run_metadata.max_turns += 10
                user_response = await prompt_callback(
                    f"Increase maximum turn to {run_metadata.max_turns}, continue? (y/n, press Enter to continue): "
                )
                if user_response.lower() not in ["y", "yes", ""]:
                    logger.warning("Stopped by user.")
                    return False
            else:
                return False

        return True
