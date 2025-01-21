import logging
from typing import Optional
from datetime import datetime

from .types import AgentConfig, CompletionRequest
from .utils import DebugUtils, TokenCounter, record_usage
from .llm.llm_client import LLMClient
from .utils.mesage_params_utils import get_text_from_message_params

logger = logging.getLogger(__name__)


class ContentSummarizer:
    def __init__(self, config: AgentConfig):
        self.model = config.model
        self.client: LLMClient = LLMClient(config)
        self.tag = self.__class__.__name__
        self.token_counter = TokenCounter()
        self.system_context: Optional[str] = None

    def update_context(self, system_context: str) -> None:
        self.system_context = system_context

    async def summarize(self, model: str, messages: list[dict], instruction: Optional[str] = None) -> Optional[str]:
        """Summarize message content using LLM with optional custom instruction.

        Args:
            model: The model identifier to use
            messages: List of message dictionaries to summarize
            instruction: Optional custom instruction to append to the default summarization prompt

        Returns:
            Optional[str]: Summarized content or None if summarization fails
        """
        message_content = get_text_from_message_params(
            model=model, messages=messages, content_max_length=200, prepend_message_id=True
        )
        return await self._summarize_content(message_content, instruction)

    async def summarize_text(self, text: str, instruction: Optional[str] = None) -> Optional[str]:
        """Summarize arbitrary text content using LLM with optional custom instruction.

        Args:
            text: The text content to summarize
            instruction: Optional custom instruction to append to the default summarization prompt

        Returns:
            Optional[str]: Summarized content or None if summarization fails
        """
        return await self._summarize_content(text, instruction)

    async def _summarize_content(self, content: str, instruction: Optional[str] = None) -> Optional[str]:
        """Internal method to handle the actual summarization for both messages and text.

        Args:
            content: The content to summarize
            instruction: Optional custom instruction to append to the default summarization prompt

        Returns:
            Optional[str]: Summarized content or None if summarization fails
        """
        # Build the prompt with default instruction and optional custom instruction
        base_instruction = """
Those messages above will be truncated from message list,
please utilize system message and context info to summarize those content by extracting useful info.
This summaries will be added to the beginning of the message list again. Be specific on details
such as filename or path etc, and be concise.
"""
        final_instruction = f"{base_instruction} {instruction}" if instruction else base_instruction

        message = {
            "role": "user",
            "content": f"<content>{content}</content> {final_instruction}",
        }

        if not self.system_context:
            logger.warning("No system context are provided ")
        request = CompletionRequest(
            model=self.model,
            messages=[message],
            temperature=0.4,
            system_prompt_suffix=self.system_context,  # use context as system message
        )

        system_token = self.token_counter.count_token(self.system_context)
        messages_token = self.token_counter.count_dict_tokens(message)
        total_tokens = system_token + messages_token
        token_stats = {
            "system_context": system_token,
            "messages_token": messages_token,
            "total_tokens": total_tokens,
        }

        try:
            completion_response = await self.client.send_completion_request(request=request)
            summary = completion_response.get_text()

            try:
                usage_dict = record_usage(completion_response, subfolder="usage", filename="summarization")
                token_stats["actual_usage"] = usage_dict
                metrics = {
                    "timestamp": datetime.now().isoformat(),
                    "model": self.model,
                    "token_stats": token_stats,
                    "system_context": self.system_context,
                    "message": message,
                    "summary": summary,
                }

                DebugUtils.take_snapshot(
                    messages=[metrics],
                    suffix="summarization",
                    with_timestamp=True,
                    subfolder="summarization",
                )
            except Exception as e:
                logger.error(f"Running into error when record metrics for summarization: {e}")

            return summary
        except Exception as e:
            logger.error(f"{self.tag} error: {str(e)}")
            return None


def create_summarizer(config: AgentConfig) -> ContentSummarizer:
    """
    Create a ContentSummarizer with appropriate model selection based on the agent's config.

    Args:
        config (AgentConfig): The agent's configuration

    Returns:
        ContentSummarizer: Configured summarizer instance
    """
    summarizer_config = AgentConfig(
        model=_get_summarizer_model(config.model),
        api_key=config.api_key,
        openai_api_key=config.openai_api_key,
        anthropic_api_key=config.anthropic_api_key,
        gemini_api_key=config.gemini_api_key,
        use_cue=config.use_cue,
    )

    return ContentSummarizer(summarizer_config)


def _get_summarizer_model(agent_model: str) -> str:
    """
    Determine the appropriate summarizer model based on the agent's model.

    Args:
        agent_model (str): The model name used by the agent

    Returns:
        str: The model name to use for the summarizer
    """
    if "claude" in agent_model.lower():
        return "claude-3-5-haiku-20241022"
    return "gpt-4o-mini"
