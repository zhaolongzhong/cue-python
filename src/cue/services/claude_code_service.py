import logging
from typing import Optional, AsyncIterator

from claude_code_sdk import (
    UserMessage,
    ResultMessage,
    AssistantMessage,
    ClaudeCodeOptions,
    query,
)

from ..types import CompletionResponse

logger = logging.getLogger(__name__)


class ClaudeCodeService:
    """Service for integrating Claude Code SDK with Cue."""

    def __init__(self, system_prompt: Optional[str] = None, max_turns: int = 20):
        self.options = ClaudeCodeOptions(
            allowed_tools=[
                "Bash",
                "Edit",
                "Glob",
                "Grep",
                "LS",
                "MultiEdit",
                "NotebookEdit",
                "NotebookRead",
                "Read",
                "Task",
                "TodoWrite",
                "WebFetch",
                "WebSearch",
                "Write",
            ],
            permission_mode="acceptEdits",
            system_prompt=system_prompt,
            max_turns=5,
        )

    async def query(self, prompt: str) -> AsyncIterator[CompletionResponse]:
        """Query Claude Code and yield CompletionResponse objects."""
        try:
            messages = []
            async for message in query(prompt=prompt, options=self.options):
                if isinstance(message, AssistantMessage):
                    messages.append(message)
                elif isinstance(message, UserMessage):
                    messages.append(message)
                elif isinstance(message, ResultMessage):
                    # Handle tool results if needed
                    logger.debug(f"Tool result: {message}")

            # Yield the final combined response
            if messages:
                # Use the last message which contains the final result
                final_message = messages[-1]
                yield CompletionResponse(
                    response=final_message,
                    model="claude-code",
                    usage=None,
                )
        except Exception as e:
            logger.error(f"Error querying Claude Code: {e}")
            raise

    async def stream_query(self, prompt: str) -> AsyncIterator[CompletionResponse]:
        """Stream Claude Code responses - yields each message as it comes."""
        try:
            async for message in query(prompt=prompt, options=self.options):
                print(f"stream_query - stream message: {message}")
                if isinstance(message, AssistantMessage):
                    yield CompletionResponse(
                        response=message,
                        model="claude-code",
                        usage=None,
                    )
                elif isinstance(message, UserMessage):
                    yield CompletionResponse(
                        response=message,
                        model="claude-code",
                        usage=None,
                    )
                elif isinstance(message, ResultMessage):
                    # Handle tool results if needed
                    logger.debug(f"Tool result: {message}")
        except Exception as e:
            logger.error(f"Error streaming Claude Code: {e}")
            raise
