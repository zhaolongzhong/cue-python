import os
import re
import json
import logging
from typing import Optional

import openai
from pydantic import BaseModel
from openai.types.chat import ChatCompletionMessageToolCall
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.completion_create_params import Function

from ..types import AgentConfig, ErrorResponse, CompletionRequest, CompletionResponse
from ..utils import DebugUtils, TokenCounter, generate_id
from .llm_request import LLMRequest
from .system_prompt import SYSTEM_PROMPT
from .openai_client_utils import JSON_FORMAT, O1_MODEL_SYSTEM_PROMPT_BASE

logger = logging.getLogger(__name__)


class OpenAIClient(LLMRequest):
    def __init__(
        self,
        config: AgentConfig,
    ):
        api_key = config.openai_api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("API key is missing in both config and settings.")

        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.config = config
        self.model = config.model
        logger.debug(f"[OpenAIClient] initialized with model: {self.model} {self.config.id}")

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
            is_o1_mini = "o1-mini" == request.model
            is_o1 = "o1" == request.model
            if is_o1_mini:
                messages = self.handle_o1_model(messages, request)
                DebugUtils.take_snapshot(messages, suffix=f"{request.model}_pre_request")
                response = await self.client.chat.completions.create(
                    messages=messages,
                    model=self.model,
                    max_completion_tokens=request.max_tokens,
                )
                content = response.choices[0].message.content
                _is_json, json_dict = self.extract_json_dict(content)
                if json_dict:
                    response = self.convert_tool_call(response, json_dict)
            else:
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
                    f"{self.config.model_dump_json(indent=4)} input_tokens: {json.dumps(input_tokens, indent=4)} "
                    "\nsystem_message: \n{json.dumps(system_message, indent=4)}"
                )
                messages.insert(0, system_message)
                DebugUtils.take_snapshot(messages=messages, suffix=f"{request.model}_pre_request")
                if self.tools:
                    if is_o1:
                        response = await self.client.chat.completions.create(
                            messages=messages,
                            model=self.model,
                            max_completion_tokens=request.max_tokens,
                            response_format=request.response_format,
                            tool_choice=request.tool_choice,
                            tools=self.tools,
                        )
                    else:
                        response = await self.client.chat.completions.create(
                            messages=messages,
                            model=self.model,
                            max_completion_tokens=request.max_tokens,
                            temperature=request.temperature,
                            response_format=request.response_format,
                            tool_choice=request.tool_choice,
                            tools=self.tools,
                            parallel_tool_calls=request.parallel_tool_calls,
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

    def handle_o1_model(self, messages: list[dict], request: CompletionRequest) -> list[dict]:
        """
        For o1, filter out system message, combine merge them into a message with assistant role.
        """
        try:
            tools = request.tools
            formatted_tools = "\n".join([json.dumps(tool) for tool in tools])

            messages = [msg for msg in request.messages if msg["role"] != "system"]
            system_message_content = " ".join([msg["content"] for msg in messages if msg["role"] == "system"])
            system_message_content = system_message_content.strip()
            if request.system_context:
                system_message_content += request.system_context

            system_prompt = O1_MODEL_SYSTEM_PROMPT_BASE.format(
                json_format=JSON_FORMAT,
                available_functions=formatted_tools,
                additional_context=(
                    f"{request.system_prompt_suffix}{{' ' + system_message_content if system_message_content else ''}}"
                ),
            )
            system_prompt = f"{SYSTEM_PROMPT} {system_prompt}"
            # logger.debug(f"o1_system_prompt: {system_prompt}")

            system_message = {
                "role": "assistant",
                "content": f"<system_context>{system_prompt}</system_context>",
            }
            # convert tools call and tool message to normal message
            final_messages = [msg for msg in messages if msg["role"] != "system"]
            final_messages.insert(0, system_message)
            logger.debug(f"{request.model} system message: {system_prompt}")
            return final_messages
        except Exception as e:
            logger.error(f"Error handling o1 model: {e}")

    def extract_json_dict(self, string: str) -> tuple[bool, Optional[dict]]:
        """
        Extract and parse a JSON dictionary from a string that may contain other content.
        Handles JSON blocks with or without markdown code fence markers.

        Args:
            string: Input string that may contain JSON

        Returns:
            Tuple[bool, Optional[Dict]]: (success, parsed_dict)
        """
        string = string.strip()
        logger.debug(f"extract_json_dict input string: {string}")

        # Try to find ```json ... ``` block first
        json_block_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
        matches = re.findall(json_block_pattern, string, re.DOTALL)

        if not matches:
            # If no code block found, try to parse the string directly
            json_str = string
        else:
            # Use the first matched block
            json_str = matches[0].strip()

        logger.debug(f"Extracted potential JSON string: {json_str}")

        try:
            # Pre-process the string to handle line breaks properly
            # First, replace actual line breaks in the content with \n
            lines = json_str.splitlines()
            json_str = " ".join(lines)

            # Now the string is all on one line, we can safely parse it
            json_object = json.loads(json_str)
            is_json = isinstance(json_object, dict)
            return (is_json, json_object)
        except ValueError:
            # If first attempt fails, try with more aggressive normalization
            try:
                # Remove any remaining control characters
                json_str = "".join(char for char in json_str if char.isprintable() or char in "\n\r\t")
                json_object = json.loads(json_str)
                is_json = isinstance(json_object, dict)
                logger.debug(f"Valid JSON dictionary after normalization: {is_json}")
                return (is_json, json_object)
            except ValueError as e:
                logger.debug(f"JSON parsing failed after normalization: {e}")
                return (False, None)

    def convert_tool_call(self, chat_completion: ChatCompletion, tool_dict: dict) -> ChatCompletion:
        try:
            original = chat_completion.choices[0].message
            tool_call_id = self.generate_tool_id()
            adjusted_tool_dict = {
                "name": tool_dict.get("name"),
                "arguments": json.dumps(tool_dict.get("arguments", {})),  # Serialize arguments to JSON string
            }

            tool_call = ChatCompletionMessageToolCall(
                id=tool_call_id, function=Function(**adjusted_tool_dict), type="function"
            )

            # Initialize tool_calls if it's None
            if original.tool_calls is None:
                original.tool_calls = [tool_call]
                logger.debug("Initialized tool_calls as an empty array.")

        except Exception as e:
            logger.error(f"Error in convert_tool_call: {e}, tool_dict: {tool_dict}")
        return chat_completion

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
