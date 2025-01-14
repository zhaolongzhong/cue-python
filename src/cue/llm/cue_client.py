import os
import re
import json
import logging

import httpx
from pydantic import BaseModel

from ..utils import DebugUtils, generate_id
from ..config import get_settings
from ..schemas import AgentConfig, ErrorResponse, CompletionRequest, CompletionResponse
from .llm_request import LLMRequest
from .system_prompt import SYSTEM_PROMPT
from .openai_client_utils import JSON_FORMAT, O1_MODEL_SYSTEM_PROMPT_BASE

logger = logging.getLogger(__name__)


class CueClient(LLMRequest):
    def __init__(
        self,
        config: AgentConfig,
    ):
        api_key = config.api_key or os.environ.get("CUE_API_KEY")
        if not api_key:
            raise ValueError("API key is missing in both config and settings.")

        self.api_key = api_key
        self.config = config
        self.model = config.model
        settings = get_settings()
        self.base_url = settings.get_base_url()
        logger.debug(f"[CueClient] initialized with model: {self.model} {self.config.id}")

    async def send_completion_request(self, request: CompletionRequest) -> CompletionResponse:
        self.tool_json = request.tool_json
        response = None
        error = None

        try:
            messages = [
                msg.model_dump(exclude_none=True, exclude_unset=True) if isinstance(msg, BaseModel) else msg
                for msg in request.messages
            ]

            DebugUtils.debug_print_messages(messages, tag=f"{self.config.id} send_completion_request")

            # Prepare the system message and context
            if request.system_context:
                system_context = {"role": "assistant", "content": request.system_context.strip()}
                messages.insert(0, system_context)

            system_prompt = (
                f"{SYSTEM_PROMPT}{' ' + request.system_prompt_suffix if request.system_prompt_suffix else ''}"
            )
            system_message = {"role": "system", "content": system_prompt}
            messages.insert(0, system_message)
            request.messages = messages

            async with httpx.AsyncClient() as client:
                headers = {
                    "X-API-Key": f"{self.api_key}",
                    "Content-Type": "application/json",
                    "accept": "application/json",
                }

                # Prepare request data with tool-related parameters
                request_data = request.model_dump()
                if self.tool_json:
                    if "o1" in request.model:
                        # Special handling for o1 models
                        system_prompt = O1_MODEL_SYSTEM_PROMPT_BASE.format(
                            json_format=JSON_FORMAT,
                            available_functions="\n".join([json.dumps(tool) for tool in self.tool_json]),
                            additional_context=request.system_prompt_suffix or "",
                        )
                        request_data["messages"][0]["content"] = system_prompt

                    # Add tool-related parameters
                    request_data["tools"] = self.tool_json
                    request_data["tool_choice"] = request.tool_choice
                    if request.parallel_tool_calls:
                        request_data["parallel_tool_calls"] = request.parallel_tool_calls

                response = await client.post(f"{self.base_url}/chat/completions", json=request_data, headers=headers)

                if response.status_code != 200:
                    error = ErrorResponse(
                        message=f"API request failed with status {response.status_code}: {response.text}",
                        code=str(response.status_code),
                    )
                    return CompletionResponse(author=request.author, model=self.model, error=error)

                response_data = response.json()
                if "o1" in request.model:
                    # Extract and convert tool calls from JSON response for o1 models
                    content = response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    is_json, json_dict = self.extract_json_dict(content)
                    if json_dict:
                        response_data = self.convert_tool_call(response_data, json_dict)

                # Handle tool calls for non-o1 models and clean up IDs
                if "choices" in response_data and response_data["choices"]:
                    message = response_data["choices"][0].get("message", {})
                    if message.get("tool_calls"):
                        self.replace_tool_call_ids(response_data, request.model)

                return CompletionResponse(**response_data)

        except httpx.RequestError as e:
            error = ErrorResponse(message=f"Request failed: {str(e)}")
        except Exception as e:
            error = ErrorResponse(message=f"Exception: {str(e)}")

        if error:
            logger.error(error.model_dump())
        return CompletionResponse(author=request.author, model=self.model, error=error)

    def extract_json_dict(self, string: str) -> tuple[bool, dict | None]:
        """
        Extract and parse a JSON dictionary from a string that may contain other content.
        Handles JSON blocks with or without markdown code fence markers.
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
            lines = json_str.splitlines()
            json_str = " ".join(lines)

            # Parse JSON
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

    def convert_tool_call(self, response_data: dict, tool_dict: dict) -> dict:
        """Convert JSON response to tool call format for o1 models."""
        try:
            message = response_data.get("choices", [{}])[0].get("message", {})
            tool_call_id = self.generate_tool_id()

            adjusted_tool_dict = {
                "name": tool_dict.get("name"),
                "arguments": json.dumps(tool_dict.get("arguments", {})),
            }

            tool_call = {"id": tool_call_id, "type": "function", "function": adjusted_tool_dict}

            # Add tool call to message
            message["tool_calls"] = [tool_call]
            response_data["choices"][0]["message"] = message

        except Exception as e:
            logger.error(f"Error in convert_tool_call: {e}, tool_dict: {tool_dict}")

        return response_data

    def replace_tool_call_ids(self, response_data: dict, model: str) -> None:
        """
        Replace tool call IDs in the response to ensure uniqueness and proper format.
        Also cleans up any tool names containing dots.
        """
        try:
            if "choices" in response_data and response_data["choices"]:
                message = response_data["choices"][0].get("message", {})
                tool_calls = message.get("tool_calls", [])

                for tool_call in tool_calls:
                    tool_call["id"] = self.generate_tool_id()
                    if "function" in tool_call and "name" in tool_call["function"]:
                        if "." in tool_call["function"]["name"]:
                            logger.error(f"Received tool name that contains dot: {tool_call}")
                            name = tool_call["function"]["name"].replace(".", "")
                            tool_call["function"]["name"] = name
        except Exception as e:
            logger.error(f"Error in replace_tool_call_ids: {e}")

    def generate_tool_id(self) -> str:
        """Generate a short tool call ID for session-scoped uniqueness."""
        return generate_id(prefix="call_", length=4)
