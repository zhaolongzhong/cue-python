from typing import Any, Union, Optional, cast

from pydantic import Field, BaseModel, ConfigDict
from anthropic.types import (
    Message as AnthropicMessage,
    TextBlock,
    MessageParam as AnthropicMessageParam,
    ToolUseBlock,
    TextBlockParam,
    ToolUseBlockParam,
)
from openai.types.chat import ChatCompletion, ChatCompletionMessageToolCall, ChatCompletionAssistantMessageParam

from .error import ErrorResponse

__all__ = ["CompletionResponse", "CompletionUsage", "ToolCallToolUseBlock"]

ToolCallToolUseBlock = Union[ChatCompletionMessageToolCall, ToolUseBlock]


class CompletionUsage(BaseModel):
    input_tokens: int = Field(default=0, alias="prompt_tokens")
    output_tokens: int = Field(default=0, alias="completion_tokens")

    # https://platform.openai.com/docs/guides/prompt-caching/requirements
    total_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0

    # https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching
    cache_creation_input_tokens: Optional[int] = Field(default=0)
    cache_read_input_tokens: Optional[int] = Field(default=0)

    model_config = ConfigDict(
        populate_by_name=True,  # Allows both alias and field name to be used
        extra="ignore",  # Ignores extra fields in the input data
    )


class InvalidResponseTypeError(Exception):
    """Raised when response type is neither AnthropicMessage nor ChatCompletion"""

    pass


class CompletionResponse:
    def __init__(
        self,
        model: str,
        author: Optional[Any] = None,
        response: Optional[Any] = None,
        chat_completion: Optional[ChatCompletion] = None,
        anthropic_message: Optional[AnthropicMessage] = None,
        error: Optional[Any] = None,
        metadata: Optional[Any] = None,
        msg_id: Optional[str] = None,
        **kwargs,
    ):
        self.msg_id = msg_id
        self.author = author
        self.model = model
        self.error = error
        self.metadata = metadata
        if isinstance(chat_completion, dict):
            chat_completion = ChatCompletion(**chat_completion)
        if isinstance(anthropic_message, dict):
            anthropic_message = AnthropicMessage(**anthropic_message)
        self.response = response or chat_completion or anthropic_message

    def get_id(self) -> str:
        if isinstance(self.response, AnthropicMessage):
            return self.response.id
        elif isinstance(self.response, ChatCompletion):
            return self.response.id
        raise InvalidResponseTypeError(
            f"Expected AnthropicMessage or ChatCompletion, got {type(self.response).__name__}"
        )

    def get_text(self) -> Optional[str]:
        if self.response is None:
            if isinstance(self.error, ErrorResponse):
                return self.error.message
            elif self.error:
                return str(self.error)
            else:
                raise Exception("Unexpected response at CompletionResponse")

        if isinstance(self.response, AnthropicMessage):
            return "\n".join(content.text for content in self.response.content if isinstance(content, TextBlock))
        elif isinstance(self.response, ChatCompletion):
            return self.response.choices[0].message.content
        raise InvalidResponseTypeError(
            f"Expected AnthropicMessage or ChatCompletion, got {type(self.response).__name__}"
        )

    def get_tool_calls(self) -> Optional[list[Any]]:
        if isinstance(self.response, AnthropicMessage):
            tool_calls = [
                content_item for content_item in self.response.content if isinstance(content_item, ToolUseBlock)
            ]
            return tool_calls
        elif isinstance(self.response, ChatCompletion):
            return self.response.choices[0].message.tool_calls
        elif isinstance(self.error, ErrorResponse):
            return None
        raise InvalidResponseTypeError(
            f"Expected AnthropicMessage or ChatCompletion, got {type(self.response).__name__}"
        )

    def get_tool_calls_peek(self, debug=False) -> Optional[str]:
        """Peek the tool use (action) and preview all arguments with truncated values"""
        tool_calls = self.get_tool_calls()
        if not tool_calls:
            return None

        previews = []
        for tool in tool_calls:
            if isinstance(tool, ToolUseBlock):
                tool_id = tool.id[:10]
                preview_args = []

                # Process all arguments in the input
                for key, value in tool.input.items():
                    str_value = str(value)
                    if len(str_value) > 20:
                        str_value = str_value[:17] + "..."
                    preview_args.append(f"{key}={str_value}")

                preview = f"{tool_id}:({', '.join(preview_args)})"
                if debug:
                    preview = f"ToolUseBlock-{preview}"
                previews.append(preview)

            elif isinstance(tool, ChatCompletionMessageToolCall):
                tool_id = tool.id[:10]

                try:
                    import json

                    args = json.loads(tool.function.arguments)
                    preview_args = []

                    # Process all arguments
                    for key, value in args.items():
                        str_value = str(value)
                        if len(str_value) > 20:
                            str_value = str_value[:17] + "..."
                        preview_args.append(f"{key}={str_value}")

                    preview = f"{tool_id}:({', '.join(preview_args)})"
                    if debug:
                        preview = f"ChatCompletion-{preview}"
                    previews.append(preview)

                except json.JSONDecodeError:
                    preview = f"{tool_id}:(invalid_json)"
                    if debug:
                        preview = f"ChatCompletion-{preview}"
                    previews.append(preview)

            else:
                raise InvalidResponseTypeError(
                    f"Expected ToolUseBlock or ChatCompletionMessageToolCall, got {type(tool)}"
                )

        return " | ".join(previews) if previews else None

    def get_usage(self) -> Optional[CompletionUsage]:
        if self.response is None:
            return None

        if isinstance(self.response, AnthropicMessage):
            return CompletionUsage(**self.response.usage.model_dump())
        elif isinstance(self.response, ChatCompletion):
            usage = self.response.usage

            completion_usage = CompletionUsage(**usage.model_dump())
            if usage.completion_tokens_details:
                completion_usage.reasoning_tokens = usage.completion_tokens_details.reasoning_tokens
            if usage.prompt_tokens_details:
                completion_usage.cached_tokens = usage.prompt_tokens_details.cached_tokens
            return completion_usage
        raise InvalidResponseTypeError(
            f"Expected AnthropicMessage or ChatCompletion, got {type(self.response).__name__}. "
            f"Response: \n{self.response}"
        )

    def __str__(self):
        response = self.response.model_dump() if self.response else None
        return f"msg_id: {self.msg_id}, Text: {self.get_text()}, Tools: {self.get_tool_calls()}, Response: {response}"

    def to_params(self):
        response = self.response
        error = self.error
        if "claude" in self.model:
            if isinstance(response, AnthropicMessage):
                content = self._response_to_anthropic_params(response)
                # Server can retrun empty content or array like `"content": []`
                # Check for empty content (None, empty array, or empty string)
                # If there is empty content, there will be 400 error like "all messages must have non-empty content
                # except for the optional final assistant message"
                if not content or (isinstance(content, list) and len(content) == 0) or content == "":
                    return "EMPTY"
                return AnthropicMessageParam(role="assistant", content=self._response_to_anthropic_params(response))
            elif isinstance(self.error, ErrorResponse):
                return AnthropicMessageParam(role="assistant", content=error.model_dump_json())
            else:
                raise ValueError(f"Unexpected subclass of CompletionResponse: {type(response)}, {self.model}")
        else:
            if isinstance(self.response, ChatCompletion):
                return self._response_to_chat_completion_params(self.response)
            elif isinstance(self.error, ErrorResponse):
                return ChatCompletionAssistantMessageParam(role="assistant", content=error.model_dump_json())
            else:
                raise ValueError(f"Unexpected subclass of CompletionResponse: {type(response)}, {self.model}")

    def _response_to_anthropic_params(
        self,
        response: AnthropicMessage,
    ) -> list[Union[TextBlockParam, ToolUseBlockParam]]:
        res: list[Union[TextBlockParam, ToolUseBlockParam]] = []
        for block in response.content:
            if isinstance(block, TextBlock):
                res.append({"type": "text", "text": block.text})
            else:
                res.append(cast(ToolUseBlockParam, block.model_dump()))
        return res

    def _response_to_chat_completion_params(self, response: ChatCompletion):
        return cast(ChatCompletionAssistantMessageParam, response.choices[0].message.model_dump())

    @staticmethod
    def parse_response_data(response_data: dict, model: str) -> "CompletionResponse":
        """
        Parse response data from various formats into a standardized CompletionResponse.

        Handles three formats:
        1. AnthropicMessage format: {'content': [{'text': '...', 'type': 'text'}], ...}
        2. ChatCompletion format: {'choices': [{'message': {'content': '...'}}], ...}
        3. CompletionResponse format: {'model': '...', 'content': '...'}
        """

        # AnthropicMessage format
        if "content" in response_data and isinstance(response_data.get("content"), list):
            return CompletionResponse(
                model=model,
                anthropic_message=response_data,
            )

        # ChatCompletion format
        elif "choices" in response_data and isinstance(response_data.get("choices"), list):
            return CompletionResponse(
                model=model,
                chat_completion=response_data,
            )

        # Has chat_completion field
        elif "chat_completion" in response_data and isinstance(response_data["chat_completion"], dict):
            chat_completion = response_data["chat_completion"]
            return CompletionResponse(
                model=response_data.get("model", model),
                chat_completion=chat_completion,
            )

        # Has anthropic_message field
        elif "anthropic_message" in response_data and isinstance(response_data["anthropic_message"], dict):
            anthropic_message = response_data["anthropic_message"]
            return CompletionResponse(
                model=model,
                anthropic_message=anthropic_message,
            )

        # Direct CompletionResponse format
        elif "model" in response_data and "content" in response_data:
            return CompletionResponse(**response_data)

        print(f"Unrecognized response format with keys: {list(response_data.keys())}")
        return CompletionResponse(
            model=model,
            response=str(response_data),
        )
