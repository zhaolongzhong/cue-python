import json
import asyncio
import logging
from typing import Optional

from mcp.types import CallToolResult
from anthropic.types import ToolUseBlock, TextBlockParam, ImageBlockParam, ToolResultBlockParam
from openai.types.chat import ChatCompletionMessageToolCall as ToolCall

from ..tools import ToolResult, ToolManager
from ..types import (
    Author,
    AgentConfig,
    CompletionRequest,
    CompletionResponse,
    ToolResponseWrapper,
    ToolCallToolUseBlock,
)
from .llm_model import ChatModel
from .cue_client import CueClient
from .llm_request import LLMRequest
from .gemini_client import GeminiClient
from .openai_client import OpenAIClient
from .anthropic_client import AnthropicClient

logger = logging.getLogger(__name__)


class LLMClient(LLMRequest):
    def __init__(self, config: AgentConfig):
        self.model = config.model
        self.llm_client: LLMRequest = self._initialize_client(config)

    def _initialize_client(self, config: AgentConfig):
        chat_model = ChatModel.from_model_id(config.model)
        provider = chat_model.provider
        client_class_mapping = {
            "openai": OpenAIClient,
            "anthropic": AnthropicClient,
            "google": GeminiClient,
            "cue": CueClient,
        }

        if config.use_cue:
            client_class = client_class_mapping.get("cue")
        else:
            client_class = client_class_mapping.get(provider)
            if not client_class:
                raise ValueError(f"Client class for key prefix {provider} not found")

        return client_class(config=config)

    async def send_completion_request(self, request: CompletionRequest) -> CompletionResponse:
        return await self.llm_client.send_completion_request(request=request)

    async def process_tools_with_timeout(
        self,
        tool_manager: ToolManager,
        tool_calls: list[ToolCallToolUseBlock],
        timeout: int = 60,
        author: Optional[Author] = None,
    ) -> ToolResponseWrapper:
        tool_results = []
        tasks = []

        for tool_call in tool_calls:
            if isinstance(tool_call, ToolCall):
                tool_name = tool_call.function.name
                tool_id = tool_call.id
                kwargs = json.loads(tool_call.function.arguments)
            elif isinstance(tool_call, ToolUseBlock):
                tool_name = tool_call.name
                tool_id = tool_call.id
                kwargs = tool_call.input
            else:
                raise ValueError(f"Unsupported tool call type: {type(tool_call)}")

            if not tool_manager.has_tool(tool_name):
                error_message = (
                    f"Tool '{tool_name}' not found. The name can be only one of those names: "
                    f"{tool_manager.tools.keys()}."
                )
                logger.error(f"{error_message}, tool_call: {tool_call}")
                tool_name = tool_name.replace(".", "")
                tool_results.append(
                    self.create_error_response(
                        tool_id,
                        error_message,
                        tool_name,
                    )
                )
                continue

            if tool_manager.mcp is not None and (result := tool_manager.mcp.find_tool(tool_name)):
                server_name, _tool_info = result
                task = asyncio.create_task(
                    tool_manager.mcp.call_tool(server_name=server_name, tool_name=tool_name, arguments=kwargs)
                )
                tasks.append((task, tool_id, tool_name))
            else:
                tool_func = tool_manager.tools[tool_name]
                task = asyncio.create_task(self.run_tool(tool_func, **kwargs))
                tasks.append((task, tool_id, tool_name))

        base64_images = []
        agent_transfer = None
        for task, tool_id, tool_name in tasks:
            try:
                tool_result: ToolResult = await asyncio.wait_for(task, timeout=timeout)
                if isinstance(tool_result, ToolResult):
                    agent_transfer = tool_result.agent_transfer
                    if agent_transfer:
                        # if we have transfer tool use, ignore other tools
                        break

                    base64_image = tool_result.base64_image
                    if base64_image:
                        base64_images.append(base64_image)

                    tool_results.append(self.create_success_response(tool_id, tool_result, tool_name))
                elif isinstance(tool_result, CallToolResult):
                    # convert CallToolResult to ToolResult
                    content_list = tool_result.content  # a list
                    output = ""
                    data = ""
                    for content in content_list:
                        if content.type == "text":
                            output += "\n" if output else "" + content.text
                        elif content.type == "image":
                            data = content.data
                    if tool_result.isError:
                        tool_result = ToolResult(output=None, error=output)
                    else:
                        tool_result = ToolResult(output=output, base64_image=data)

                    tool_results.append(self.create_success_response(tool_id, tool_result, tool_name))

            except asyncio.TimeoutError:
                error_message = f"Timeout while calling tool <{tool_name}> after {timeout}s."
                logger.error(error_message)
                tool_results.append(self.create_error_response(tool_id, error_message, tool_name))
            except Exception as e:
                error_message = f"Error while calling tool <{tool_name}>: {e}"
                logger.error(error_message)
                tool_results.append(self.create_error_response(tool_id, error_message, tool_name))

        response = None
        if "claude" in self.model:
            tool_result_message = {"role": "user", "content": tool_results}
            response = ToolResponseWrapper(
                tool_result_message=tool_result_message,
                author=author,
                agent_transfer=agent_transfer,
                model=self.model,
            )
        else:
            response = ToolResponseWrapper(
                tool_messages=tool_results,
                author=author,
                base64_images=base64_images,
                agent_transfer=agent_transfer,
                model=self.model,
            )

        return response

    async def run_tool(self, tool_func, **kwargs):
        return await tool_func(**kwargs)

    def create_success_response(self, tool_id: str, result: ToolResult, tool_name: Optional[str] = None):
        if "claude" in self.model:
            tool_result_content: list[TextBlockParam | ImageBlockParam] | str = []
            is_error = False
            if result.error:
                is_error = True
                tool_result_content = _maybe_prepend_system_tool_result(result, result.error)
            else:
                if result.output:
                    tool_result_content.append(
                        {
                            "type": "text",
                            "text": _maybe_prepend_system_tool_result(result, result.output),
                        }
                    )
                if result.base64_image:
                    tool_result_content.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": result.base64_image,
                            },
                        }
                    )

            # BetaToolResultBlockParam
            tool_result_block_param = {
                "tool_use_id": tool_id,
                "content": tool_result_content,
                "type": "tool_result",
                "is_error": is_error,
            }
            return tool_result_block_param
        else:
            tool_result_content = ""
            if result.error:
                is_error = True
                tool_result_content = _maybe_prepend_system_tool_result(result, result.error)
            else:
                if result.output:
                    tool_result_content = _maybe_prepend_system_tool_result(result, result.output)
                # if result.base64_image:
                #     # handle in loop
                #     pass

            # ChatCompletionToolMessageParam
            tool_message_param = {
                "role": "tool",
                "content": str(tool_result_content),
                "tool_call_id": tool_id,
            }
            return tool_message_param

    def create_error_response(self, tool_id: str, error_message: str, tool_name: str):
        if "claude" in self.model:
            result_param = ToolResultBlockParam(
                tool_use_id=tool_id, content=error_message, type="tool_result", is_error=True
            )
            return result_param
        else:
            tool_message_param = {
                "role": "tool",
                "content": error_message,
                "tool_call_id": tool_id,
            }
            return tool_message_param


def _maybe_prepend_system_tool_result(result: ToolResult, result_text: str):
    if result.system:
        result_text = f"<system>{result.system}</system>\n{result_text}"
    return result_text
