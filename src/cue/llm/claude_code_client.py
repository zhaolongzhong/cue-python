import logging
from typing import AsyncIterator

from ..types import AgentConfig, MessageParam, CompletionRequest, CompletionResponse
from .llm_request import LLMRequest
from ..services.claude_code_service import ClaudeCodeService

logger = logging.getLogger(__name__)


class ClaudeCodeClient(LLMRequest):
    """Client for Claude Code SDK integration."""

    def __init__(self, config: AgentConfig):
        self.config = config
        self.service = ClaudeCodeService(
            system_prompt=config.instruction,
            max_turns=config.max_turns or 20,
        )

    async def send_completion_request(self, request: CompletionRequest) -> CompletionResponse:
        """Send completion request to Claude Code."""
        try:
            # Extract the last user message as the prompt
            prompt = ""
            for message in reversed(request.messages):
                # Handle both dict and MessageParam objects
                role = message.get("role") if isinstance(message, dict) else message.role
                content = message.get("content") if isinstance(message, dict) else message.content

                if role == "user":
                    prompt = content
                    break

            if not prompt:
                raise ValueError("No user message found in request")

            # Get the first response from Claude Code
            async for response in self.service.query(prompt):
                return response

            # If no response, return empty
            return CompletionResponse(
                response=MessageParam(role="assistant", content=""),
                model="claude-code",
                usage=None,
            )

        except Exception as e:
            logger.error(f"Error in Claude Code completion: {e}")
            raise

    async def send_streaming_completion_request(self, request: CompletionRequest) -> AsyncIterator[CompletionResponse]:
        """Send streaming completion request to Claude Code - yields each message as it comes."""
        try:
            # Extract the last user message as the prompt
            prompt = ""
            for message in reversed(request.messages):
                # Handle both dict and MessageParam objects
                role = message.get("role") if isinstance(message, dict) else message.role
                content = message.get("content") if isinstance(message, dict) else message.content

                if role == "user":
                    prompt = content
                    break

            if not prompt:
                raise ValueError("No user message found in request")

            # Stream each response from Claude Code
            async for response in self.service.stream_query(prompt):
                yield response

        except Exception as e:
            logger.error(f"Error in Claude Code streaming completion: {e}")
            raise
