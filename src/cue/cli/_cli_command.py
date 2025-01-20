import logging
from enum import Enum
from typing import Optional
from dataclasses import dataclass

from rich.text import Text
from rich.console import Console

logger = logging.getLogger(__name__)


@dataclass
class CommandInfo:
    description: str
    usage: str
    examples: list[str]


class CliCommand(Enum):
    STOP_RUN = "stop"
    EXIT = "exit"
    QUIT = "quit"
    SNAPSHOT = "snapshot"
    SNAPSHOT_SHORT = "-s"
    HELP = "help"
    HELP_SHORT = "-h"


# Command documentation
COMMAND_DOCS: dict[CliCommand, CommandInfo] = {
    CliCommand.STOP_RUN: CommandInfo(description="Stop the current run", usage="stop", examples=["stop"]),
    CliCommand.EXIT: CommandInfo(description="Exit the CLI", usage="exit or quit", examples=["exit", "quit"]),
    CliCommand.SNAPSHOT: CommandInfo(
        description="Take a snapshot of current conversation context",
        usage="snapshot [message] or -s [message]",
        examples=["snapshot", "-s", "snapshot let's continue our chat", "-s what's next?"],
    ),
    CliCommand.HELP: CommandInfo(
        description="Show available commands",
        usage="help or -h [command]",
        examples=["help", "-h", "help snapshot", "-h -s"],
    ),
}


def print_help(console: Console, command: Optional[str] = None) -> None:
    """
    Print help information for commands

    Args:
        console: Rich console instance for formatted output
        command: Optional specific command to show help for
    """

    def print_command_info(cmd: CliCommand, info: CommandInfo) -> None:
        message = Text()
        message.append(f"\n{cmd.value}", style="bold cyan")
        if cmd == CliCommand.SNAPSHOT:
            message.append(f", {CliCommand.SNAPSHOT_SHORT.value}", style="bold cyan")
        elif cmd == CliCommand.HELP:
            message.append(f", {CliCommand.HELP_SHORT.value}", style="bold cyan")
        message.append("\n  Description: ", style="bold")
        message.append(f"{info.description}\n")
        message.append("  Usage: ", style="bold")
        message.append(f"{info.usage}\n")
        message.append("  Examples:\n", style="bold")
        for example in info.examples:
            message.append(f"    {example}\n")
        console.print(message)

    if command:
        # Show help for specific command
        cmd_lower = command.lower()
        for cmd, info in COMMAND_DOCS.items():
            if cmd_lower in [cmd.value, getattr(cmd, "value_short", None)]:
                print_command_info(cmd, info)
                return
        console.print(f"Unknown command: {command}", style="bold red")
    else:
        # Show general help
        message = Text("\nAvailable Commands:", style="bold green")
        console.print(message)
        for cmd, info in COMMAND_DOCS.items():
            print_command_info(cmd, info)


def parse_command(user_input: str) -> tuple[Optional[CliCommand], Optional[str]]:
    """
    Parse user input to separate commands from regular messages

    Returns:
        Tuple of (Command if present, remaining message content)
    """
    input_lower = user_input.strip().lower()

    # Check for help commands
    if input_lower in [CliCommand.HELP.value, CliCommand.HELP_SHORT.value]:
        return (CliCommand.HELP, None)

    if input_lower.startswith(f"{CliCommand.HELP.value} ") or input_lower.startswith(f"{CliCommand.HELP_SHORT.value} "):
        remaining = user_input.split(" ", 1)[1]
        return (CliCommand.HELP, remaining)

    # Check for basic commands
    if input_lower in [CliCommand.EXIT.value, CliCommand.QUIT.value]:
        return (CliCommand.EXIT, None)

    # Check for stop commands
    if input_lower in [CliCommand.STOP_RUN.value]:
        return (CliCommand.STOP_RUN, None)

    # Check for snapshot commands
    if input_lower == CliCommand.SNAPSHOT.value or input_lower == CliCommand.SNAPSHOT_SHORT.value:
        return (CliCommand.SNAPSHOT, None)

    # Check for commands with additional text
    if input_lower.startswith(f"{CliCommand.SNAPSHOT_SHORT.value} ") or input_lower.startswith(
        f"{CliCommand.SNAPSHOT.value} "
    ):
        remaining = user_input.split(" ", 1)[1]
        return (CliCommand.SNAPSHOT, remaining)

    return (None, user_input)
