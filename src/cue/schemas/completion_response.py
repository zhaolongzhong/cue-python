from typing import Any, Union, Optional, cast

from pydantic import Field, BaseModel, ConfigDict
from anthropic.types import (
    Message as AnthropicMessage,
    TextBlock,
    ToolUseBlock,
)
from openai.types.chat import ChatCompletion, ChatCompletionMessageToolCall, ChatCompletionAssistantMessageParam
from anthropic.types.beta import (
    BetaTextBlock,
    BetaMessageParam,
    BetaTextBlockParam,
    BetaToolUseBlockParam,
)
from anthropic.types.beta.prompt_caching import PromptCachingBetaMessage

from ..schemas.error import ErrorResponse
from ..schemas.message import Author, Content, Metadata, ContentType, MessageCreate

ToolCallToolUseBlock = Union[ChatCompletionMessageToolCall, ToolUseBlock]


class CompletionUsage(BaseModel):
    input_tokens: int = Field(default=0, alias="prompt_tokens")
    output_tokens: int = Field(default=0, alias="completion_tokens")

    # https://platform.openai.com/docs/guides/prompt-caching/requirements
    total_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0

    # https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

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
        author: Optional[Any],
        response: Optional[Any] = None,
        error: Optional[Any] = None,
        metadata: Optional[Any] = None,
        msg_id: Optional[str] = None,
    ):
        self.msg_id = msg_id
        self.author = author
        self.model = model
        self.response = response
        self.error = error
        self.metadata = metadata

    def get_id(self) -> str:
        if isinstance(self.response, (AnthropicMessage, PromptCachingBetaMessage)):
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

        if isinstance(self.response, (AnthropicMessage, PromptCachingBetaMessage)):
            return "\n".join(content.text for content in self.response.content if isinstance(content, TextBlock))
        elif isinstance(self.response, ChatCompletion):
            return self.response.choices[0].message.content
        raise InvalidResponseTypeError(
            f"Expected AnthropicMessage or ChatCompletion, got {type(self.response).__name__}"
        )

    def get_tool_calls(self) -> Optional[list[Any]]:
        if isinstance(self.response, (AnthropicMessage, PromptCachingBetaMessage)):
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

        if isinstance(self.response, (AnthropicMessage, PromptCachingBetaMessage)):
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
            f"Expected AnthropicMessage or ChatCompletion, got {type(self.response).__name__}. Response: \n{self.response}"
        )

    def __str__(self):
        response = self.response.model_dump() if self.response else None
        return f"msg_id: {self.msg_id}, Text: {self.get_text()}, Tools: {self.get_tool_calls()}, Response: {response}"

    def to_params(self):
        response = self.response
        error = self.error
        if "claude" in self.model:
            if isinstance(response, (AnthropicMessage, PromptCachingBetaMessage)):
                content = self._response_to_anthropic_params(response)
                # Server can retrun empty content or array like `"content": []`
                # Check for empty content (None, empty array, or empty string)
                # If there is empty content, there will be 400 error like "all messages must have non-empty content except for the optional final assistant message"
                if not content or (isinstance(content, list) and len(content) == 0) or content == "":
                    return "EMPTY"
                return BetaMessageParam(role="assistant", content=self._response_to_anthropic_params(response))
            elif isinstance(self.error, ErrorResponse):
                return BetaMessageParam(role="assistant", content=error.model_dump_json())
            else:
                print(f"Unexpected subclass of CompletionResponse: {type(response)}, {self.model}")
        else:
            if isinstance(self.response, ChatCompletion):
                return self._response_to_chat_completion_params(self.response)
            elif isinstance(self.error, ErrorResponse):
                return ChatCompletionAssistantMessageParam(role="assistant", content=error.model_dump_json())
            else:
                print(f"Unexpected subclass of CompletionResponse: {type(response)}, {self.model}")

    def to_message_create(self) -> MessageCreate:
        response = self.response
        error = self.error

        if "claude" in self.model:
            if isinstance(response, (AnthropicMessage, PromptCachingBetaMessage)):
                if isinstance(response, PromptCachingBetaMessage):
                    author = Author(role=response.role)
                    metadata = Metadata(model=response.model, payload=response.model_dump())
                    processed_content = []
                    has_tool_use = False

                    for block in response.content:
                        if isinstance(block, ToolUseBlock):
                            has_tool_use = True
                            processed_content.append(block.model_dump())
                        elif isinstance(block, TextBlock):
                            processed_content.append(block.model_dump())
                        else:
                            raise Exception(f"Unhandled content block type: {block}")

                    content_type = ContentType.tool_use if has_tool_use else ContentType.text
                    content = Content(type=content_type, content=processed_content)
                    return MessageCreate(author=author, content=content, metadata=metadata)
                else:
                    raise Exception(f"Unhandled response: {response}")
            elif isinstance(self.error, ErrorResponse):
                author = Author(role="assistant")
                content = Content(type=ContentType.text, content=error.model_dump_json())
                metadata = Metadata(model=self.model)
                return MessageCreate(author=author, content=content, metadata=metadata)
            else:
                raise Exception(f"Unexpected subclass of CompletionResponse: {type(response)}, {self.model}")
        else:
            if isinstance(response, ChatCompletion):
                message = response.choices[0].message
                author = Author(role=message.role)
                metadata = Metadata(model=response.model, payload=response.model_dump())
                if message.tool_calls:
                    content = Content(
                        type=ContentType.tool_calls,
                        content=message.content if message.content else "",
                        tool_calls=[item.model_dump() for item in message.tool_calls],
                    )
                elif message.content:
                    content = Content(type=ContentType.text, content=message.content)
                else:
                    raise Exception(f"Unhandled message: {message}")

                return MessageCreate(author=author, content=content, metadata=metadata)
            elif isinstance(self.error, ErrorResponse):
                author = Author(role="assistant")
                content = Content(content=error.model_dump_json())
                metadata = Metadata(model=self.model)
                return MessageCreate(author=author, content=content, metadata=metadata)
            else:
                raise Exception(f"Unexpected subclass of CompletionResponse: {type(response)}, {self.model}")

    def _response_to_anthropic_params(
        self,
        response: Union[AnthropicMessage, PromptCachingBetaMessage],
    ) -> list[Union[BetaTextBlockParam, BetaToolUseBlockParam]]:
        res: list[Union[BetaTextBlockParam, BetaToolUseBlockParam]] = []
        for block in response.content:
            if isinstance(block, BetaTextBlock):
                res.append({"type": "text", "text": block.text})
            else:
                res.append(cast(BetaToolUseBlockParam, block.model_dump()))
        return res

    def _response_to_chat_completion_params(self, response: ChatCompletion):
        return cast(ChatCompletionAssistantMessageParam, response.choices[0].message.model_dump())
