from typing import Union

from anthropic.types import Message as AnthropicMessage, TextBlock, ToolUseBlock
from openai.types.chat import ChatCompletion

from ..types import MessageParam, ErrorResponse, CompletionResponse, ToolResponseWrapper
from .message import MessageCreate
from ..types.message import Author, Content, Metadata, ContentType

__all__ = ["MessageCreateFactory"]


class MessageCreateFactory:
    """Factory for creating MessageCreate."""

    @staticmethod
    def create_from(message: Union[MessageParam, ToolResponseWrapper, CompletionResponse]):
        if isinstance(message, MessageParam):
            message_create = MessageCreateFactory.from_message_param(message)
        elif isinstance(message, ToolResponseWrapper):
            message_create = MessageCreateFactory.from_tool_response(message)
        elif isinstance(message, CompletionResponse):
            message_create = MessageCreateFactory.from_completion_response(message)
        else:
            raise ValueError(f"Unsupported type: {type(message)}")
        return message_create

    @staticmethod
    def from_message_param(message_param: MessageParam) -> "MessageCreate":
        author = Author(role=message_param.role)
        content = Content(type=ContentType.text, content=message_param.content)
        metadata = Metadata(model=message_param.model)
        return MessageCreate(author=author, content=content, metadata=metadata)

    @staticmethod
    def from_tool_response(response: ToolResponseWrapper) -> "MessageCreate":
        if "claude" in response.model:
            author = Author(role="user")
            content = Content(type=ContentType.tool_result, content=response.tool_result_message["content"])
            metadata = Metadata(model=response.model)
            return MessageCreate(author=author, content=content, metadata=metadata)
        else:
            author = Author(role="tool")
            content = Content(type=ContentType.tool_message, content=response.tool_messages)
            metadata = Metadata(model=response.model)
            return MessageCreate(author=author, content=content, metadata=metadata)

    @staticmethod
    def from_completion_response(completion_response: CompletionResponse) -> "MessageCreate":
        response = completion_response.response
        error = completion_response.error

        if "claude" in completion_response.model:
            if isinstance(response, AnthropicMessage):
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
            elif isinstance(completion_response.error, ErrorResponse):
                author = Author(role="assistant")
                content = Content(type=ContentType.text, content=error.model_dump_json())
                metadata = Metadata(model=completion_response.model)
                return MessageCreate(author=author, content=content, metadata=metadata)
            else:
                raise Exception(
                    f"Unexpected subclass of CompletionResponse: {type(response)}, {completion_response.model}"
                )
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
            elif isinstance(completion_response.error, ErrorResponse):
                author = Author(role="assistant")
                content = Content(content=error.model_dump_json())
                metadata = Metadata(model=completion_response.model)
                return MessageCreate(author=author, content=content, metadata=metadata)
            else:
                raise Exception(
                    f"Unexpected subclass of CompletionResponse: {type(response)}, {completion_response.model}"
                )
