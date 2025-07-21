import sys
import logging
import builtins
from typing import Optional
from threading import Lock

from rich.text import Text
from rich.style import Style
from rich.theme import Theme
from rich.console import Console
from rich.logging import RichHandler

default_theme = Theme(
    {
        "debug": Style(color="grey54"),
        "info": Style(color="cyan"),
        "warning": Style(color="magenta"),
        "error": Style(color="red", bold=True),
        "assistant_bold": Style(color="green", bold=True),
        "assistant": Style(color="yellow", bold=False),
        "user": Style(color="blue", bold=True),
        "system": Style(color="magenta", bold=True),
        "success": Style(color="cyan", bold=True),
    }
)

console = Console(theme=default_theme, markup=True, stderr=True)
original_print = builtins.print


class PromptManager:
    """Manages the user prompt to prevent duplication."""

    def __init__(self, console: Console):
        self.prompt_active = False
        self.lock = Lock()
        self.console = console

    def print_prompt(self):
        with self.lock:
            if not self.prompt_active:
                text = Text()
                # text.append("$ >> ", style="user")
                self.console.print(text, end="")
                self.prompt_active = True
                # Debugging Statement (Optional)
                # original_print("[DEBUG] Prompt printed.", file=sys.stderr)

    def reset_prompt(self):
        with self.lock:
            self.prompt_active = False
            # Debugging Statement (Optional)
            # original_print("[DEBUG] Prompt reset.", file=sys.stderr)


# Initialize PromptManager
prompt_manager = PromptManager(console=console)


def print_prompt():
    """Consistently print the user prompt."""
    prompt_manager.print_prompt()


def reset_prompt():
    """Reset the prompt state before printing a new message."""
    prompt_manager.reset_prompt()


def custom_print(*args, **kwargs):
    """Custom print function to manage prompts."""
    # Clear the current line
    original_print("\r\033[K", end="", file=sys.stdout)

    # Reset prompt state as we're about to print a new message
    reset_prompt()

    # Determine if the print call is meant to print the prompt itself
    if args == ("$ >> ",) or args == ("$ >> ", ""):
        return

    # Print the actual message
    if args and args[0] != "$ >> ":
        original_print(*args, **kwargs)

    # Print the prompt after the message if the print call ends with a newline
    if kwargs.get("end", "\n") == "\n":
        print_prompt()


# Override the built-in print function
builtins.print = custom_print


class ConsoleUtils:
    """Utility class for managing console messages."""

    def __init__(self, console: Console, prompt_manager: PromptManager):
        self.console = console
        self.prompt_manager = prompt_manager

    def print_msg(self, message: str, role: str = "assistant") -> None:
        """Print a standard message from an agent."""
        self.prompt_manager.reset_prompt()
        # Clear the current line
        sys.stdout.write("\r\033[K")
        sys.stdout.flush()
        # Create and print the styled message
        text = Text()
        # text.append("$ >> ", style="assistant_bold" if role == "assistant" else "user")
        text.append(str(message), style="assistant" if role == "assistant" else "user")
        self.console.print(text)
        # Print the prompt after the message
        self.prompt_manager.print_prompt()

    def print_error_msg(self, message: str, agent_id: Optional[str] = None) -> None:
        """Print an error message from an agent."""
        self.prompt_manager.reset_prompt()
        # Clear the current line
        sys.stdout.write("\r\033[K")
        sys.stdout.flush()
        # Create and print the styled error message
        text = Text()
        if not agent_id:
            agent_id = "$ >> "
        text.append(f"[{agent_id}]: ", style="assistant")
        text.append(str(message), style="error")
        self.console.print(text)
        # Print the prompt after the error message
        self.prompt_manager.print_prompt()


class SimpleFormatter(logging.Formatter):
    """Custom formatter to include short log levels."""

    def format(self, record):
        record.shortlevel = record.levelname[0]
        return super().format(record)


class CustomRichHandler(RichHandler):
    """Custom RichHandler to integrate with PromptManager."""

    def __init__(self, prompt_manager: PromptManager, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.formatter = SimpleFormatter(
            "[%(asctime)s][%(shortlevel)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S.%f"
        )
        self.prompt_manager = prompt_manager

    def emit(self, record):
        try:
            # Apply theme style based on log level
            message = self.format(record)
            style = record.levelname.lower()

            # Clear the current line
            original_print("\r\033[K", end="", file=sys.stderr)

            # Reset prompt state before printing the log
            self.prompt_manager.reset_prompt()

            # Print the styled log message
            self.console.print(message, style=style)

            # Print the prompt after the log message
            self.prompt_manager.print_prompt()
        except Exception:
            self.handleError(record)


# Initialize CustomRichHandler with PromptManager
rich_handler = CustomRichHandler(
    prompt_manager=prompt_manager,
    console=console,
    show_level=False,
    show_time=False,
    show_path=False,
    markup=False,
    rich_tracebacks=True,
)

# Initialize ConsoleUtils with PromptManager
console_utils = ConsoleUtils(console=console, prompt_manager=prompt_manager)

# Initial prompt
print_prompt()
