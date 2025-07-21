from typing import List, Optional, AsyncGenerator

try:
    from google import genai
except ImportError:
    genai = None

from .types import Message, InputItem, StepResult, SimpleAgent, NextStepRunAgain, NextStepFinalOutput
from .model_base import ModelBase
from .tool_executor import ToolExecutor


class GeminiModel(ModelBase):
    """Gemini-specific model with Google's message format"""

    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key)
        if not genai:
            raise ImportError("google-generativeai library not installed")
        self.tool_executor = ToolExecutor()

    async def get_response(self, agent: SimpleAgent, input_items: List[InputItem]) -> StepResult:
        """Run Gemini agent to completion"""
        api_key = self._get_api_key(agent, "GEMINI_API_KEY")
        genai.configure(api_key=api_key)
        client = genai.Client()

        # Build Gemini contents format
        contents = self._build_contents(agent, input_items)

        # Always run with tools available - let LLM decide whether to use them
        return await self._run_with_tools(client, agent, contents, input_items)

    async def stream_response(
        self, agent: SimpleAgent, input_items: List[InputItem], hooks=None
    ) -> AsyncGenerator[str, None]:
        """Stream Gemini responses"""
        api_key = self._get_api_key(agent, "GEMINI_API_KEY")
        genai.configure(api_key=api_key)
        client = genai.Client()

        contents = self._build_contents(agent, input_items)

        stream = await client.aio.models.generate_content_stream(model=agent.model, contents=contents)

        full_response = ""
        async for chunk in stream:
            if chunk.text:
                full_response += chunk.text
                yield chunk.text

        # Update agent after streaming
        agent.messages.extend(
            [
                Message(role="user", content=self._format_inputs(input_items)),
                Message(role="assistant", content=full_response),
            ]
        )

    def _build_contents(self, agent: SimpleAgent, input_items: List[InputItem]) -> List[dict]:
        """Build Gemini contents format"""
        contents = []

        # Add system prompt as first user message if present
        if agent.system_prompt:
            contents.append({"role": "user", "parts": [{"text": f"System: {agent.system_prompt}"}]})

        # Agent history
        for msg in agent.messages:
            if msg.role != "system":
                contents.append({"role": msg.role, "parts": [{"text": msg.content}]})

        # New input
        if input_items:
            contents.append({"role": "user", "parts": [{"text": self._format_inputs(input_items)}]})

        return contents

    def _format_inputs(self, input_items: List[InputItem]) -> str:
        """Format input items to string"""
        contents = []
        for item in input_items:
            if item.type == "text":
                contents.append(str(item.content))
        return " ".join(contents)

    async def _run_with_tools(
        self, client, agent: SimpleAgent, contents: List[dict], input_items: List[InputItem] = None
    ) -> StepResult:
        """Gemini-specific tool use logic"""
        # Convert tool schemas to Gemini format
        tool_schemas = self.tool_executor.get_tool_schemas()
        gemini_tools = []

        for schema in tool_schemas:
            func = schema.get("function", {})
            gemini_tools.append(
                {
                    "function_declarations": [
                        {
                            "name": func.get("name", ""),
                            "description": func.get("description", ""),
                            "parameters": func.get("parameters", {}),
                        }
                    ]
                }
            )

        # Single model call
        try:
            response = await client.aio.models.generate_content(
                model=agent.model, contents=contents, tools=gemini_tools if gemini_tools else None
            )

            # Check if response has function calls
            function_calls = []
            if hasattr(response, "candidates") and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, "content") and hasattr(candidate.content, "parts"):
                    for part in candidate.content.parts:
                        if hasattr(part, "function_call"):
                            function_calls.append(part.function_call)

            usage = {
                "input_tokens": 0,  # Gemini doesn't always provide usage
                "output_tokens": 0,
                "total_tokens": 0,
            }

            # If no function calls, we're done
            if not function_calls:
                result_text = response.text if hasattr(response, "text") else str(response)
                return StepResult(content=result_text, usage=usage, next_step=NextStepFinalOutput())

            # Tools were used - execute them and prepare for next step
            # Add assistant message with function calls
            contents.append({"role": "model", "parts": [{"function_call": fc} for fc in function_calls]})

            # Execute tools and add results
            function_responses = []
            executed_results = []
            for function_call in function_calls:
                # Execute tool
                tool_result = await self.tool_executor.execute(
                    function_call.name, dict(function_call.args) if hasattr(function_call, "args") else {}
                )

                executed_results.append(f"{function_call.name}: {tool_result}")
                function_responses.append(
                    {"function_response": {"name": function_call.name, "response": {"result": str(tool_result)}}}
                )

            contents.append({"role": "user", "parts": function_responses})

            # Update agent messages with tool interaction
            agent.messages.extend(
                [
                    Message(role="assistant", content=f"[Used tools: {', '.join(fc.name for fc in function_calls)}]"),
                    Message(role="user", content=f"[Tool results: {'; '.join(executed_results)}]"),
                ]
            )

            return StepResult(
                content=f"Tools executed: {', '.join(fc.name for fc in function_calls)}",
                usage=usage,
                metadata={"tool_results": executed_results, "tool_count": len(function_calls)},
                next_step=NextStepRunAgain(),
            )

        except Exception as e:
            # Handle Gemini API errors
            return StepResult(
                content=f"Gemini API error: {str(e)}",
                usage={},
                metadata={"error": True},
                next_step=NextStepFinalOutput(),
            )
