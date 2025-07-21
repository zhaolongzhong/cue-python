import os
from abc import ABC, abstractmethod
from typing import List, Optional, AsyncGenerator

from .types import InputItem, StepResult, SimpleAgent
from .streaming_hooks import StreamEvent, StreamingHooks


class ModelBase(ABC):
    """Base model interface - each provider implements single-turn model response logic

    This is the core abstraction for AI model interactions:
    - Handles single model request/response cycles
    - Provider-specific message formatting and API calls
    - Tool execution within a single model turn
    - Streaming response generation with events

    Each provider (Anthropic, OpenAI, Gemini) implements this interface
    with their specific API requirements and message formats.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key

    @abstractmethod
    async def get_response(self, agent: SimpleAgent, input_items: List[InputItem]) -> StepResult:
        """Get a single response from the model

        This method handles one complete model interaction:
        - Formats messages for the specific provider
        - Makes API call with tools if available
        - Executes any tool calls returned by the model
        - Returns StepResult with content, usage, and next_step decision
        """
        pass

    @abstractmethod
    async def stream_response(
        self, agent: SimpleAgent, input_items: List[InputItem], hooks: Optional[StreamingHooks] = None
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream a single response from the model"""
        pass

    def _get_api_key(self, agent: SimpleAgent, env_var: str) -> str:
        """Get API key from agent, model, or environment"""
        return agent.api_key or self.api_key or os.getenv(env_var)


def get_model_for_name(model: str, api_key: Optional[str] = None) -> ModelBase:
    """Factory function to get appropriate model for model name"""
    model_lower = model.lower()

    if "claude" in model_lower or "anthropic" in model_lower:
        from .anthropic_model import AnthropicModel

        return AnthropicModel(api_key)
    elif "gpt" in model_lower or "openai" in model_lower:
        from .openai_model import OpenAIModel

        return OpenAIModel(api_key)
    elif "gemini" in model_lower:
        from .gemini_model import GeminiModel

        return GeminiModel(api_key)
    else:
        from .openai_model import OpenAIModel

        return OpenAIModel(api_key)


# Backward compatibility alias for now
get_runner_for_model = get_model_for_name
