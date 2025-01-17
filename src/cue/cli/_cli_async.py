import os
import sys
import json
import signal
import asyncio
import logging
import argparse
from typing import Optional
from pathlib import Path
from functools import partial
from concurrent.futures import ThreadPoolExecutor

from rich.text import Text

from ..utils import DebugUtils, console_utils
from ..config import get_settings
from ..schemas import RunMetadata, CompletionResponse
from ..utils.logs import setup_logging
from ._cli_command import CliCommand, parse_command
from .._agent_manager import AgentManager
from .._agent_provider import AgentProvider
from ..tools.mcp_manager import MCPServerManager

setup_logging()
logger = logging.getLogger(__name__)


class CLI:
    def __init__(self, args):
        self.logger = logger
        self.console_utils = console_utils
        self.enable_services = False
        self.enable_debug_turn = args.enable_debug_turn
        self.mode = self.determine_mode(args)
        self.runner_id = args.runner_id if hasattr(args, "runner_id") else "default"
        self.run_metadata: Optional[RunMetadata] = None
        self.config_path = args.config_path if hasattr(args, "config_path") else None

    def determine_mode(self, args) -> str:
        """
        Determine the operating mode based on command-line arguments.

        Args:
            args: Parsed command-line arguments.

        Returns:
            str: Mode of operation: 'cli', 'runner', 'client'.
        """
        if args.client:
            return "client"
        elif args.runner:
            # Log runner mode with ID if present
            if args.runner_id:
                self.logger.info(f"Running in runner mode with ID: {args.runner_id}")
            else:
                self.logger.info("Running in runner mode with default ID")
            return "runner"
        else:
            return "cli"

    def setup_runner_environment(self):
        """Set up environment variables for runner mode"""
        if self.runner_id:
            os.environ["CUE_RUNNER_ID"] = self.runner_id

            # Set up runner space paths
            runner_space = Path(f"/tmp/cue_runners/{self.runner_id}")
            runner_space.mkdir(parents=True, exist_ok=True)

            # Set control and service file paths
            control_file = runner_space / "control.json"
            service_file = runner_space / "service.json"

            os.environ["CUE_CONTROL_FILE"] = str(control_file)
            os.environ["CUE_SERVICE_FILE"] = str(service_file)

            self.logger.debug(f"Set runner environment - ID: {self.runner_id}")
            self.logger.debug(f"Control file: {control_file}")
            self.logger.debug(f"Service file: {service_file}")

    async def _config_agents(self):
        """Configure all agents."""
        self.agents = {
            config.id: self.agent_manager.register_agent(config)
            for config in self.agent_provider.get_configs().values()
        }

    async def setup(self, mcp: MCPServerManager):
        """Asynchronous initialization."""
        logger.debug(f"setup run mode: {self.mode}")
        # Get the current running loop
        self.loop = asyncio.get_running_loop()

        self.run_metadata = RunMetadata(
            runner_name=self.runner_id,
            mode=self.mode,
            enable_turn_debug=self.enable_debug_turn,
            enable_services=self.enable_services,
        )

        # Create executor only if needed (not in runner mode)
        if self.run_metadata.mode != "runner":
            self.executor = ThreadPoolExecutor(thread_name_prefix="cli_executor")

        # Initialize agent manager
        self.agent_manager = AgentManager(prompt_callback=self.handle_prompt, loop=self.loop)

        if self.run_metadata.mode != "client":
            # Use custom config path if provided, otherwise use default from settings
            config_file = self.config_path or get_settings().AGENTS_CONFIG_FILE
            if self.config_path:
                logger.info(f"Using custom config file: {self.config_path}")

            # Initialize other components
            self.agent_provider = AgentProvider(config_file)
            self.primary_agent_config = self.agent_provider.get_primary_agent()

            # Configure agents
            await self._config_agents()

            # Set up primary agent
            self.agent_manager.primary_agent = self.agents[self.primary_agent_config.id]
            self.agent_manager.active_agent = self.agent_manager.primary_agent

        # Initialize the agent manager
        await self.agent_manager.initialize(self.run_metadata, mcp)

        # Trigger bootstrap sequence for primary agent in CLI mode
        # if self.mode == "cli" and self.agent_manager.primary_agent:
        #     bootstrap_metadata = RunMetadata(
        #         enable_turn_debug=False,
        #     )
        #     # Send bootstrap message to trigger bootstrap sequence
        #     await self.run_loop(
        #         self.agent_manager.primary_agent.id,
        #         "This is bootstrap AUTO-EXECUTE ON STARTUP. Please access relevant context, review our previous session's work, and prepare a summary to begin our current session.",
        #         bootstrap_metadata,
        #     )

        #     # Print separator for visual clarity
        #     self.console_utils.console.print("\n" + "=" * 50 + "\n")

    def _sync_get_input(self, prompt: str) -> str:
        """Synchronous input function."""
        return self.console_utils.console.input(prompt)

    async def _get_user_input_async(self, prompt: str) -> str:
        """Get user input asynchronously using the executor."""
        try:
            input_func = partial(self._sync_get_input, prompt)
            return await self.loop.run_in_executor(self.executor, input_func)
        except Exception as e:
            logger.error(f"Error getting user input: {e}", exc_info=True)
            raise

    async def handle_prompt(self, prompt: str) -> str:
        """Handles prompts from the AgentManager by asking the user via the CLI."""
        return await self._get_user_input_async(prompt)

    async def run(self, mcp: MCPServerManager):
        """Main run loop for the CLI."""
        try:
            await self.setup(mcp=mcp)

            self.logger.debug("Running the CLI. Commands: 'exit'/'quit' to exit, 'snapshot'/'-s' to save context")

            if self.run_metadata.mode == "runner":
                # In runner mode, just keep the agent alive and process websocket messages
                self.logger.info("Running in runner mode - waiting for websocket messages...")
                try:
                    # Keep the runner alive until interrupted
                    while True:
                        await asyncio.sleep(1)
                except asyncio.CancelledError:
                    self.logger.info("Runner received cancellation signal")
                    raise
                return

            while True:
                try:
                    user_input = await self._get_user_input_async("")

                    # Parse command
                    command, message = parse_command(user_input)

                    if not command and not message or message and message.lower() in ["y", "yes", ""]:
                        continue

                    # Handle empty or too short messages
                    if not command and (not message or len(message) < 3):
                        logger.debug(f"command: {command}, message: {message}")
                        self.console_utils.print_error_msg("Message must be at least 3 characters long.")
                        continue

                    # Handle commands
                    if command == CliCommand.HELP:
                        from ._cli_command import print_help

                        print_help(self.console_utils.console, message)
                        continue

                    if command == CliCommand.EXIT:
                        self.logger.info("Exit command received. Shutting down.")
                        break
                    if command == CliCommand.STOP_RUN:
                        self.logger.info("Stop command received. Stopping current run.")
                        await self.agent_manager.stop_run()
                        continue

                    if command == CliCommand.SNAPSHOT:
                        try:
                            snapshot_path = self.agent_manager.active_agent.snapshot()
                            success_msg = Text("[System]: ", style="system")
                            success_msg.append(f"Snapshot saved to {snapshot_path}", style="success")
                            self.console_utils.console.print(success_msg)

                            if message and len(message) >= 3:
                                user_input = message
                            else:
                                continue
                        except Exception as e:
                            error_msg = Text("[System]: ", style="system")
                            error_msg.append(f"Failed to save snapshot: {str(e)}", style="error")
                            self.console_utils.console.print(error_msg)
                            continue

                    # Process user input
                    self.logger.debug(f"Processing user input: {user_input}")
                    self.run_metadata.user_messages.append(f"{user_input}")
                    DebugUtils.log_chat({"user": user_input})

                    if self.run_metadata.mode == "client":
                        await self.send_message(user_input)
                    else:
                        if not self.agent_manager.active_agent:
                            raise Exception("No active agent found")
                        active_agent_id = self.agent_manager.active_agent.id
                        await self.run_loop(active_agent_id, user_input, self.run_metadata)

                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    self.logger.error(f"Error in main loop: {e}", exc_info=True)
                    self.console_utils.print_error_msg(f"Error: {str(e)}")

        except Exception as e:
            self.logger.exception(f"An error occurred: {e}")
            raise
        finally:
            await self.clean_up()

    async def send_message(self, user_input) -> None:
        try:
            await self.agent_manager.broadcast_user_message(user_input)
        except Exception as e:
            self.console_utils.print_error_msg(f"Error sending message: {str(e)}")

    async def run_loop(self, agent_id: str, user_input: str, run_metadata: RunMetadata) -> Optional[CompletionResponse]:
        try:
            response = await self.agent_manager.start_run(agent_id, user_input, run_metadata)
            return response
        except Exception as e:
            self.console_utils.print_error_msg(f"Error during run: {str(e)}")

    async def clean_up(self):
        """Clean up resources."""
        self.logger.info("Cleaning up the agent manager...")
        try:
            await self.agent_manager.clean_up()
            if hasattr(self, "executor"):
                self.executor.shutdown(wait=True)
            self.logger.debug("Cleanup complete.")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}", exc_info=True)


def _parse_args():
    parser = argparse.ArgumentParser(description="Cue CLI: An interactive asynchronous client.")
    parser.add_argument("-v", "--version", action="store_true", help="Print the version of the Cue client.")
    parser.add_argument("-r", "--run", action="store_true", help="Run the interactive CLI.")
    parser.add_argument("-d", "--enable_debug_turn", action="store_true", help="Pause for each run loop turn.")
    parser.add_argument(
        "-client", "--client", action="store_true", help="Run as a second remote client with websocket."
    )
    parser.add_argument(
        "-runner",
        "--runner",
        action="store_true",
        help="Run in runner mode, communicating via websocket without local interface.",
    )
    parser.add_argument("--runner-id", type=str, metavar="ID", help="Specify runner ID for the runner mode")
    parser.add_argument(
        "-c",
        "--config_path",
        type=str,
        help="Path to a custom cue.config.json file",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (overrides environment variable).",
    )
    parser.add_argument("--log-file", type=str, help="Path to a file to log messages (overrides environment settings).")
    return parser.parse_args()


async def async_main():
    args = _parse_args()

    if len(sys.argv) == 1:
        print("No arguments provided. Use -h for help.")
        return 1

    if args.version:
        from .. import __version__

        print(f"version {__version__}")
        return 0

    # Configure logging
    if args.log_level:
        logger.setLevel(getattr(logging, args.log_level.upper(), logging.DEBUG))
        logging.getLogger("httpx").setLevel(logging.WARN)

    if args.log_file:
        file_handler = logging.FileHandler(args.log_file, encoding="utf-8")
        formatter = logging.Formatter(
            "[%(asctime)s - %(name)s:%(lineno)d - %(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG if logger.level <= logging.DEBUG else logger.level)
        logger.addHandler(file_handler)

    mcp = None
    cli = None
    try:
        # Initialize MCP
        mcp = MCPServerManager()
        await mcp.connect()
        cli = CLI(args=args)

        # Create manager task but don't await it yet
        manager_task = asyncio.create_task(mcp.run())

        try:
            await cli.run(mcp=mcp)

        except asyncio.CancelledError:
            logger.info("Main loop cancelled, initiating cleanup")
            # Request clean shutdown
            mcp.request_shutdown()
            # Wait for manager task to complete
            await manager_task
            await cli.clean_up
            raise

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt detected, initiating cleanup")
        if mcp:
            mcp.request_shutdown()
            if "manager_task" in locals():
                await manager_task
        return 130

    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return 1

    finally:
        if mcp:
            try:
                # Give cleanup a chance to complete
                await asyncio.shield(mcp.disconnect())
                logger.info("Cleanup completed")
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")
        if cli:
            await cli.clean_up


def main():
    """Entry point with proper signal handling"""
    if sys.platform != "win32":
        # Set up signal handlers for graceful shutdown
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        def handle_signal(sig):
            loop.stop()
            tasks = [t for t in asyncio.all_tasks(loop) if t is not asyncio.current_task(loop)]
            for task in tasks:
                task.cancel()

            loop.remove_signal_handler(signal.SIGTERM)
            loop.remove_signal_handler(signal.SIGINT)

        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, lambda s=sig: handle_signal(s))

    try:
        exit_code = asyncio.run(async_main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt in main")
        sys.exit(130)
    except Exception:
        logger.exception("Fatal error in main")
        sys.exit(1)


if __name__ == "__main__":
    main()
