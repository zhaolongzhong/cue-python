import os
import json
import logging

import openai
from pydantic import BaseModel
from openai.types.chat.chat_completion import ChatCompletion

from ..utils import DebugUtils, TokenCounter, generate_id
from ..schemas import AgentConfig, ErrorResponse, CompletionRequest, CompletionResponse
from .llm_request import LLMRequest
from .system_prompt import SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class GeminiClient(LLMRequest):
    def __init__(
        self,
        config: AgentConfig,
    ):
        api_key = config.gemini_api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("API key is missing in both config and settings.")
        self.client = openai.AsyncOpenAI(
            api_key=api_key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
        self.config = config
        self.model = config.model
        logger.debug(f"[GeminiClient] initialized with model: {self.model} {self.config.id}")

    async def send_completion_request(self, request: CompletionRequest) -> CompletionResponse:
        self.tools = request.tools
        response = None
        error = None
        try:
            messages = [
                msg.model_dump(exclude_none=True, exclude_unset=True) if isinstance(msg, BaseModel) else msg
                for msg in request.messages
            ]
            DebugUtils.debug_print_messages(messages, tag=f"{self.config.id} send_completion_request")

            system_prompt = (
                f"{SYSTEM_PROMPT}{' ' + request.system_prompt_suffix if request.system_prompt_suffix else ''}"
            )

            system_context_tokens = 0
            if request.system_context:
                system_context = {"role": "assistant", "content": request.system_context.strip()}
                system_context_tokens = TokenCounter.count_token(str(system_context))
                messages.insert(0, system_context)

            system_message = {"role": "system", "content": system_prompt}
            system_message_tokens = TokenCounter.count_token(str(system_message))
            tool_tokens = TokenCounter.count_token(str(request.tools))
            message_tokens = TokenCounter.count_token(str(messages))
            input_tokens = {
                "system_tokens": system_message_tokens,
                "system_context_tokens": system_context_tokens,
                "tool_tokens": tool_tokens,
                "message_tokens": message_tokens,
            }
            logger.debug(
                f"{self.config.model_dump_json(indent=4)} input_tokens: {json.dumps(input_tokens, indent=4)} \nsystem_message: \n{json.dumps(system_message, indent=4)}"
                f"\ntools_json: {json.dumps(request.tools, indent=4)}"
            )
            messages.insert(0, system_message)
            DebugUtils.take_snapshot(messages=messages, suffix=f"{request.model}_pre_request")
            if self.tools:
                response = await self.client.chat.completions.create(
                    messages=messages,
                    model=self.model,
                    max_completion_tokens=request.max_tokens,
                    temperature=request.temperature,
                    response_format=request.response_format,
                    tool_choice=request.tool_choice,
                    tools=self.tools,
                    # parallel_tool_calls=request.parallel_tool_calls, # not support
                )
            else:
                response = await self.client.chat.completions.create(
                    messages=messages,
                    model=self.model,
                    max_completion_tokens=request.max_tokens,
                    temperature=request.temperature,
                    response_format=request.response_format,
                )
            self.replace_tool_call_ids(response, request.model)

        except openai.APIConnectionError as e:
            error = ErrorResponse(message=f"The server could not be reached. {e.__cause__}")
        except openai.RateLimitError as e:
            error = ErrorResponse(
                message=f"A 429 status code was received; we should back off a bit. {e.response}",
                code=str(e.status_code),
            )
        except openai.APIStatusError as e:
            message = f"Another non-200-range status code was received. {e.response}, {e.response.text}"
            DebugUtils.debug_print_messages(messages=messages, tag=f"{self.config.id} send_completion_request")
            error = ErrorResponse(
                message=message,
                code=str(e.status_code),
            )
        except Exception as e:
            error = ErrorResponse(
                message=f"Exception: {e}",
            )
        if error:
            logger.error(error.model_dump())
        return CompletionResponse(author=request.author, response=response, model=self.model, error=error)

    def replace_tool_call_ids(self, response_data: ChatCompletion, model: str) -> None:
        """
        Replace tool call IDs in the response to:
        1) Ensure uniqueness by generating new IDs from the server if duplicates exist.
        2) Shorten IDs to save tokens (length optimization may be adjusted).
        """
        for choice in response_data.choices:
            message = choice.message
            tool_calls = message.tool_calls
            if tool_calls:
                for tool_call in tool_calls:
                    tool_call.id = self.generate_tool_id()
                    if "." in tool_call.function.name:
                        logger.error(f"Received tool name that contains dot: {tool_call}")
                        name = tool_call.function.name.replace(".", "")
                        tool_call.function.name = name

    def generate_tool_id(self) -> str:
        """Generate a short tool call ID for session-scoped uniqueness.

        Uses 4-char random suffix since unique IDs only needed within
        a single session's context window

        Returns:
            String like: 'call_a1b2'
        """
        tool_call_id = generate_id(prefix="call_", length=4)
        return tool_call_id
