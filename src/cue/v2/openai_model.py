from typing import List, Optional, AsyncGenerator

try:
    import openai
except ImportError:
    openai = None

from .types import Message, InputItem, StepResult, SimpleAgent, NextStepRunAgain, NextStepFinalOutput
from .model_base import ModelBase
from .tool_executor import ToolExecutor


class OpenAIModel(ModelBase):
    """OpenAI-specific model with its own message format and tool logic"""

    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key)
        if not openai:
            raise ImportError("openai library not installed")
        self.tool_executor = ToolExecutor()

    async def get_response(self, agent: SimpleAgent, input_items: List[InputItem]) -> StepResult:
        """Run OpenAI agent to completion"""
        api_key = self._get_api_key(agent, "OPENAI_API_KEY")
        client = openai.AsyncOpenAI(api_key=api_key)

        # Build messages from agent history + new inputs
        messages = self._build_messages(agent, input_items)

        # Always run with tools available - let LLM decide whether to use them
        return await self._run_with_tools(client, agent, messages, input_items)

    async def stream_response(
        self, agent: SimpleAgent, input_items: List[InputItem], hooks=None
    ) -> AsyncGenerator[str, None]:
        """Stream OpenAI responses"""
        api_key = self._get_api_key(agent, "OPENAI_API_KEY")
        client = openai.AsyncOpenAI(api_key=api_key)

        messages = self._build_messages(agent, input_items)

        stream = await client.chat.completions.create(
            model=agent.model, messages=messages, max_tokens=4096, stream=True
        )

        full_response = ""
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_response += content
                yield content

        # Update agent after streaming
        agent.messages.extend(
            [
                Message(role="user", content=self._format_inputs(input_items)),
                Message(role="assistant", content=full_response),
            ]
        )

    def _build_messages(self, agent: SimpleAgent, input_items: List[InputItem]) -> List[dict]:
        """Build OpenAI message format"""
        messages = []

        # System message
        if agent.system_prompt:
            messages.append({"role": "system", "content": agent.system_prompt})

        # Agent history
        for msg in agent.messages:
            messages.append({"role": msg.role, "content": msg.content})

        # New input
        if input_items:
            messages.append({"role": "user", "content": self._format_inputs(input_items)})

        return messages

    def _format_inputs(self, input_items: List[InputItem]) -> str:
        """Format input items to string"""
        contents = []
        for item in input_items:
            if item.type == "text":
                contents.append(str(item.content))
            # Can extend for other types later
        return " ".join(contents)

    async def _run_with_tools(
        self, client, agent: SimpleAgent, messages: List[dict], input_items: List[InputItem] = None
    ) -> StepResult:
        """OpenAI-specific tool calling logic"""
        # Get tool schemas from tool executor
        openai_tools = self.tool_executor.get_tool_schemas()

        # Single model call
        response = await client.chat.completions.create(
            model=agent.model, messages=messages, tools=openai_tools, max_tokens=4096
        )

        message = response.choices[0].message
        usage = {
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }

        # If no tool calls, we're done
        if not message.tool_calls:
            return StepResult(content=message.content, usage=usage, next_step=NextStepFinalOutput())

        # Tools were used - execute them and prepare for next step
        # Add assistant message with tool calls
        messages.append(
            {
                "role": "assistant",
                "content": message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in message.tool_calls
                ],
            }
        )

        # Execute tools and add results
        executed_results = []
        for tool_call in message.tool_calls:
            # Parse tool arguments
            import json

            try:
                tool_args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                tool_args = {}

            # Execute tool
            tool_result = await self.tool_executor.execute(tool_call.function.name, tool_args)

            executed_results.append(f"{tool_call.function.name}: {tool_result}")
            messages.append({"role": "tool", "content": str(tool_result), "tool_call_id": tool_call.id})

        # Update agent messages with tool interaction
        agent.messages.extend(
            [
                Message(
                    role="assistant",
                    content=f"[Used tools: {', '.join(tc.function.name for tc in message.tool_calls)}]",
                ),
                Message(role="user", content=f"[Tool results: {'; '.join(executed_results)}]"),
            ]
        )

        return StepResult(
            content=f"Tools executed: {', '.join(tc.function.name for tc in message.tool_calls)}",
            usage=usage,
            metadata={"tool_results": executed_results, "tool_count": len(message.tool_calls)},
            next_step=NextStepRunAgain(),
        )
