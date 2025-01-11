import asyncio
import logging
from typing import Any, Union, Callable, Optional

from .utils import DebugUtils
from ._agent import Agent
from .schemas import Author, RunMetadata, MessageParam, AgentTransfer, CompletionResponse, ToolResponseWrapper
from .tools._tool import ToolManager

logger = logging.getLogger(__name__)


class AgentLoop:
    def __init__(self):
        self.stop_run_event: asyncio.Event = asyncio.Event()
        self.user_message_queue: asyncio.Queue[str] = asyncio.Queue()
        self.execute_run_task: Optional[asyncio.Task] = None
        self.user_control_enabled: bool = True  # Flag to control user interaction features

    async def run(
        self,
        agent: Agent,
        tool_manager: ToolManager,
        run_metadata: RunMetadata,
        callback: Optional[Callable[[CompletionResponse], Any]] = None,
        prompt_callback: Optional[Callable] = None,
    ) -> Optional[Union[CompletionResponse, AgentTransfer]]:
        """
        Run the agent execution loop.

        Args:
            agent: The agent to run
            tool_manager: Tool manager instance
            run_metadata: Metadata for the run
            callback: Callback for processing responses
            prompt_callback: Callback for user prompts

        The loop supports:
        - Dynamic user message injection via user_message_queue
        - Graceful stopping via stop_run_event
        - Turn limit control with user approval
        - User control enabling/disabling via user_control_enabled flag
        """
        logger.debug(f"Agent run loop started. {agent.id}")
        response = None
        author = Author(role="user", name="")

        while True:
            if self.stop_run_event.is_set():
                logger.info("Stop signal received. Exiting execute_run loop.")
                break

            # Process queued messages with timeout to allow for external cancellation
            try:
                while True:
                    try:
                        # Use get_nowait() with a small sleep to allow for cancellation
                        new_message = self.user_message_queue.get_nowait()
                        logger.debug(f"Received new user message during run: {new_message}")

                        # Special handling for control messages
                        if new_message.lower() == "stop":
                            logger.info("Stop command received from user")
                            self.stop_run_event.set()
                            break

                        new_user_message = MessageParam(role="user", content=new_message)
                        message = await agent.add_message(new_user_message)
                        if callback and message:
                            await callback(message)

                    except asyncio.QueueEmpty:
                        # No more messages to process
                        break

                    # Small sleep to allow for external cancellation
                    await asyncio.sleep(0.1)

                    if self.stop_run_event.is_set():
                        break

            except asyncio.CancelledError:
                logger.info("Message processing was cancelled")
                raise

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
        """
        Determines if the loop should continue based on turn count and user input.
        
        Features:
        - Turn limit checking
        - Debug mode prompts
        - User approval for continuation
        - Max turn extension with user approval
        - User control can be disabled via user_control_enabled flag
        """
        current_turn = run_metadata.current_turn - 1
        max_turns = run_metadata.max_turns
        logger.debug(f"Maximum turn {max_turns}, current: {current_turn}")

        # Debug mode check
        if self.user_control_enabled and run_metadata.enable_turn_debug and prompt_callback:
            prompt = (
                f"Turn {current_turn}/{max_turns}\n"
                f"Options:\n"
                f"- Press Enter to continue\n"
                f"- Type 'n' to stop\n"
                f"- Type '+N' to increase max turns by N (e.g. +10)\n"
                f"- Type 'stop' to terminate execution\n"
                f"> "
            )
            user_response = await prompt_callback(prompt)

            # Handle user response
            if user_response.lower() == "stop":
                logger.warning("Stop command received in debug mode")
                self.stop_run_event.set()
                return False
            elif user_response.startswith("+"):
                try:
                    increase = int(user_response[1:])
                    run_metadata.max_turns += increase
                    logger.info(f"Increased max turns by {increase} to {run_metadata.max_turns}")
                    return True
                except ValueError:
                    logger.warning(f"Invalid turn increase value: {user_response}")
            elif user_response.lower() in ["n", "no"]:
                logger.warning("Stopped by user in debug mode")
                return False

        # Max turn check with user control
        if run_metadata.current_turn > max_turns:
            if self.user_control_enabled and prompt_callback:
                prompt = (
                    f"Maximum turn {max_turns} reached.\n"
                    f"Options:\n"
                    f"- Press Enter to add 10 more turns\n"
                    f"- Type 'n' to stop\n"
                    f"- Type '+N' to increase by N turns\n"
                    f"- Type 'stop' to terminate execution\n"
                    f"> "
                )
                user_response = await prompt_callback(prompt)

                # Handle user response
                if user_response.lower() == "stop":
                    logger.warning("Stop command received at max turns")
                    self.stop_run_event.set()
                    return False
                elif user_response.startswith("+"):
                    try:
                        increase = int(user_response[1:])
                        run_metadata.max_turns += increase
                        logger.info(f"Increased max turns by {increase} to {run_metadata.max_turns}")
                        return True
                    except ValueError:
                        logger.warning(f"Invalid turn increase value: {user_response}")
                        run_metadata.max_turns += 10  # Default increase
                elif user_response.lower() in ["n", "no"]:
                    logger.warning("Stopped by user at max turns")
                    return False
                else:
                    run_metadata.max_turns += 10  # Default increase
                    logger.info(f"Increased max turns by 10 to {run_metadata.max_turns}")
            else:
                return False

        return True
