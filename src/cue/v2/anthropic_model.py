from typing import List, Optional, AsyncGenerator

try:
    import anthropic
except ImportError:
    anthropic = None

from .types import Message, InputItem, RunResult, StepResult, SimpleAgent, NextStepRunAgain, NextStepFinalOutput
from .model_base import ModelBase
from .tool_executor import ToolExecutor
from .streaming_hooks import StreamEvent, StreamingHooks, DefaultStreamingHooks


class AnthropicModel(ModelBase):
    """Anthropic-specific model with Claude message format and tool logic"""

    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key)
        if not anthropic:
            raise ImportError("anthropic library not installed")
        self.tool_executor = ToolExecutor()

        # Cache usage logging (create new instance for proper test isolation)
        from .cache_logger import CacheLogger

        self.cache_logger = CacheLogger()

    def _extract_tools_used(self, all_tool_results):
        """Extract unique tool names from tool results"""
        return list({result["tool_name"] for result in all_tool_results}) if all_tool_results else []

    def _log_cache_usage(self, usage: dict, context: dict = None):
        """Log cache usage statistics for development analysis"""
        self.cache_logger.log_turn_usage(usage, context)

    async def get_response(self, agent: SimpleAgent, input_items: List[InputItem]) -> StepResult:
        """Run Anthropic agent to completion"""
        api_key = self._get_api_key(agent, "ANTHROPIC_API_KEY")
        client = anthropic.AsyncAnthropic(api_key=api_key)

        # Build Anthropic message format
        messages = self._build_messages(agent, input_items)

        # Always run with tools available - let LLM decide whether to use them
        return await self._run_with_tools(client, agent, messages)

    async def stream_response(
        self, agent: SimpleAgent, input_items: List[InputItem], hooks: Optional[StreamingHooks] = None
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream Anthropic responses with tool support and hooks

        Yields StreamEvent objects with accumulated content tracking.
        The final event will have type="agent_done" with the complete result.
        """
        if hooks is None:
            hooks = DefaultStreamingHooks()

        await hooks.on_stream_start(agent)

        # Track accumulated content across turns
        accumulated_content = ""
        all_tool_results = []

        # Run with tools but in streaming mode
        api_key = self._get_api_key(agent, "ANTHROPIC_API_KEY")
        client = anthropic.AsyncAnthropic(api_key=api_key)
        messages = self._build_messages(agent, input_items)

        # Get tool schemas
        tool_schemas = self.tool_executor.get_tool_schemas()
        anthropic_tools = []
        for schema in tool_schemas:
            func = schema.get("function", {})
            anthropic_tools.append(
                {
                    "name": func.get("name", ""),
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {}),
                }
            )

        for turn in range(agent.max_turns):
            # Add cache control to system prompt if it exists
            system_param = None
            if agent.system_prompt:
                system_param = [{"type": "text", "text": agent.system_prompt, "cache_control": {"type": "ephemeral"}}]

            async with client.messages.stream(
                model=agent.model,
                max_tokens=4096,
                system=system_param,
                messages=messages,
                tools=anthropic_tools if anthropic_tools else None,
            ) as stream:
                content_blocks = []
                full_text = ""
                current_tool_uses = {}  # Track by ID to avoid duplicates

                # Track usage and stream status
                usage_data = {}

                async for event in stream:
                    # Handle different event types per official spec
                    if hasattr(event, "type"):
                        event_type = event.type

                        # message_start - ignore (already have message info)
                        if event_type == "message_start":
                            continue

                        # ping events - ignore gracefully per spec
                        elif event_type == "ping":
                            continue

                        # error events - handle streaming errors
                        elif event_type == "error":
                            error_info = getattr(event, "error", {})
                            yield StreamEvent(
                                type="error",
                                content=f"Streaming error: {error_info.get('message', 'Unknown error')}",
                                metadata={"error": error_info},
                            )
                            return

                        # message_delta - track usage stats (IMPORTANT!)
                        elif event_type == "message_delta":
                            if hasattr(event, "usage"):
                                usage_dict = event.usage.__dict__ if hasattr(event.usage, "__dict__") else {}
                                # Usage counts in message_delta are CUMULATIVE, so we replace not add
                                for key in [
                                    "input_tokens",
                                    "output_tokens",
                                    "cache_creation_input_tokens",
                                    "cache_read_input_tokens",
                                ]:
                                    if key in usage_dict:
                                        usage_data[key] = usage_dict[key]  # Replace, don't accumulate
                            continue

                        # message_stop - stream ended
                        elif event_type == "message_stop":
                            # Continue to tool processing below
                            pass

                        # content_block_delta - handle text and tool input streaming
                        elif event_type == "content_block_delta":
                            if hasattr(event, "delta"):
                                delta = event.delta

                                # Text content streaming
                                if hasattr(delta, "type") and delta.type == "text_delta":
                                    if hasattr(delta, "text"):
                                        text_chunk = delta.text

                                        # Apply text hook
                                        modified_chunk = await hooks.on_text_chunk(text_chunk, agent)
                                        if modified_chunk is not None:
                                            full_text += modified_chunk
                                            accumulated_content += modified_chunk
                                            yield StreamEvent(
                                                type="text",
                                                content=modified_chunk,
                                                metadata={"accumulated": accumulated_content},
                                            )

                                # Tool input streaming (input_json_delta) - we handle this at content_block_stop
                                elif hasattr(delta, "type") and delta.type == "input_json_delta":
                                    # Let it accumulate, we'll get final tool at content_block_stop
                                    continue

                                # Extended thinking support
                                elif hasattr(delta, "type") and delta.type == "thinking_delta":
                                    if hasattr(delta, "thinking"):
                                        thinking_chunk = delta.thinking
                                        yield StreamEvent(
                                            type="thinking",
                                            content=thinking_chunk,
                                            metadata={"accumulated": accumulated_content},
                                        )

                                # Thinking signature (verification)
                                elif hasattr(delta, "type") and delta.type == "signature_delta":
                                    if hasattr(delta, "signature"):
                                        yield StreamEvent(
                                            type="thinking_signature",
                                            content="",
                                            metadata={"signature": delta.signature},
                                        )

                        # content_block_start - acknowledge start of content block
                        elif event_type == "content_block_start":
                            # Just acknowledge, we collect complete blocks at content_block_stop
                            continue

                        # content_block_stop - collect complete content blocks
                        elif event_type == "content_block_stop":
                            if hasattr(event, "content_block"):
                                block = event.content_block
                                if hasattr(block, "type") and block.type == "tool_use":
                                    current_tool_uses[block.id] = block
                                else:
                                    content_blocks.append(block)

                # Get unique tool uses
                tool_uses = list(current_tool_uses.values())

                # Get final usage data from completed stream
                try:
                    final_message = await stream.get_final_message()
                    if hasattr(final_message, "usage") and final_message.usage:
                        # Use complete usage data from final message
                        final_usage = final_message.usage
                        usage_data = {
                            "input_tokens": getattr(final_usage, "input_tokens", 0),
                            "output_tokens": getattr(final_usage, "output_tokens", 0),
                            "cache_creation_input_tokens": getattr(final_usage, "cache_creation_input_tokens", 0),
                            "cache_read_input_tokens": getattr(final_usage, "cache_read_input_tokens", 0),
                        }
                except Exception:
                    # Fallback to partial usage_data if final message unavailable
                    pass

                if not tool_uses:
                    # No tools used, we're done
                    if turn == 0 and input_items:
                        agent.messages.extend(
                            [
                                Message(role="user", content=self._format_inputs(input_items)),
                                Message(role="assistant", content=full_text),
                            ]
                        )

                    # full_text already added to accumulated_content during streaming

                    final_result = RunResult(
                        content=accumulated_content, usage=usage_data, metadata={"tool_results": all_tool_results}
                    )

                    # Log cache usage for development
                    context_type = "streaming_with_tools" if all_tool_results else "streaming_no_tools"
                    tools_used = self._extract_tools_used(all_tool_results)

                    self._log_cache_usage(
                        usage_data,
                        {
                            "type": context_type,
                            "turn": turn + 1,
                            "model": agent.model,
                            "tools_used": tools_used if tools_used else None,
                            "total_turns": turn + 1,
                        },
                    )

                    await hooks.on_stream_end(agent, final_result)

                    yield StreamEvent(
                        type="agent_done",
                        content=accumulated_content,
                        metadata={
                            "final": True,
                            "tool_results": all_tool_results,
                            "turns": turn + 1,
                            "usage": usage_data,
                        },
                    )
                    return

                # Handle tool execution - ensure we have proper content format
                assistant_content = []

                # Add text content if we have any
                if full_text.strip():
                    assistant_content.append({"type": "text", "text": full_text})

                # Add tool use blocks
                assistant_content.extend(
                    [
                        {"type": "tool_use", "id": tool_use.id, "name": tool_use.name, "input": tool_use.input}
                        for tool_use in tool_uses
                    ]
                )

                messages.append({"role": "assistant", "content": assistant_content})

                tool_results = []
                for tool_use in tool_uses:
                    await hooks.on_tool_start(tool_use.name, tool_use.input, agent)

                    yield StreamEvent(
                        type="tool_start",
                        content=f"Using {tool_use.name}...",
                        metadata={"tool_name": tool_use.name, "args": tool_use.input},
                    )

                    # Execute tool
                    tool_result = await self.tool_executor.execute(tool_use.name, tool_use.input)

                    # Apply tool result hook
                    modified_result = await hooks.on_tool_end(tool_use.name, str(tool_result), agent)
                    final_tool_result = modified_result if modified_result is not None else str(tool_result)

                    tool_result_obj = {"type": "tool_result", "tool_use_id": tool_use.id, "content": final_tool_result}
                    tool_results.append(tool_result_obj)

                    # Track all tool results across turns
                    all_tool_results.append(
                        {
                            "turn": turn,
                            "tool_name": tool_use.name,
                            "arguments": tool_use.input,
                            "result": final_tool_result,
                        }
                    )

                    yield StreamEvent(
                        type="tool_end",
                        content=final_tool_result,
                        metadata={
                            "tool_name": tool_use.name,
                            "accumulated": accumulated_content,
                            "all_tools": all_tool_results,
                        },
                    )

                messages.append({"role": "user", "content": tool_results})

                # full_text already added to accumulated_content during streaming
                # NOTE: Don't log here - usage_data may not be populated until stream ends

        # Max turns reached
        final_result = RunResult(
            content=accumulated_content or "Max turns reached",
            usage=usage_data,
            metadata={"max_turns_reached": True, "tool_results": all_tool_results},
        )

        # Log cache usage for development
        context_type = "streaming_with_tools_max_turns" if all_tool_results else "streaming_max_turns"
        tools_used = self._extract_tools_used(all_tool_results)

        self._log_cache_usage(
            usage_data,
            {
                "type": context_type,
                "turns": agent.max_turns,
                "model": agent.model,
                "tools_used": tools_used if tools_used else None,
            },
        )

        await hooks.on_stream_end(agent, final_result)
        yield StreamEvent(
            type="agent_done",
            content=accumulated_content or "Max turns reached",
            metadata={
                "max_turns": True,
                "final": True,
                "tool_results": all_tool_results,
                "turns": agent.max_turns,
                "usage": usage_data,
            },
        )

    def _build_messages(self, agent: SimpleAgent, input_items: List[InputItem]) -> List[dict]:
        """Build Anthropic message format (no system in messages)"""
        messages = []

        # Agent history (no system messages in Anthropic format)
        for msg in agent.messages:
            if msg.role != "system":
                messages.append({"role": msg.role, "content": msg.content})

        # New input
        if input_items:
            messages.append({"role": "user", "content": self._format_inputs(input_items)})

        # Apply prompt caching to optimize API usage
        self._inject_prompt_caching(messages)

        return messages

    def _inject_prompt_caching(self, messages: List[dict]) -> None:
        """Add cache control to the 3 most recent user messages to optimize API usage.

        This follows the browser-use pattern of caching recent conversation context
        to reduce token usage and improve response times.
        """
        breakpoints_remaining = 3

        for message in reversed(messages):
            if message["role"] == "user":
                # Handle both string and list content formats
                content = message.get("content")
                if isinstance(content, str):
                    # Convert string to list format for cache control
                    message["content"] = [{"type": "text", "text": content}]
                    content = message["content"]

                if isinstance(content, list) and content:
                    if breakpoints_remaining > 0:
                        # Add cache control to last content block
                        content[-1]["cache_control"] = {"type": "ephemeral"}
                        breakpoints_remaining -= 1
                    else:
                        # Remove cache control from older messages
                        content[-1].pop("cache_control", None)

    def _format_inputs(self, input_items: List[InputItem]) -> str:
        """Format input items to string"""
        contents = []
        for item in input_items:
            if item.type == "text":
                contents.append(str(item.content))
        return " ".join(contents)

    async def _run_with_tools(self, client, agent: SimpleAgent, messages: List[dict]) -> StepResult:
        """Anthropic-specific single step with tool use logic"""
        # Convert tool schemas to Anthropic format
        tool_schemas = self.tool_executor.get_tool_schemas()
        anthropic_tools = []
        for schema in tool_schemas:
            func = schema.get("function", {})
            anthropic_tools.append(
                {
                    "name": func.get("name", ""),
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {}),
                }
            )

        # Add cache control to system prompt if it exists
        system_param = None
        if agent.system_prompt:
            system_param = [{"type": "text", "text": agent.system_prompt, "cache_control": {"type": "ephemeral"}}]

        # Single model call
        response = await client.messages.create(
            model=agent.model, max_tokens=4096, system=system_param, messages=messages, tools=anthropic_tools
        )

        # Check if response has tool use
        tool_uses = [block for block in response.content if hasattr(block, "type") and block.type == "tool_use"]

        usage = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
            "cache_creation_input_tokens": getattr(response.usage, "cache_creation_input_tokens", 0),
            "cache_read_input_tokens": getattr(response.usage, "cache_read_input_tokens", 0),
        }

        if not tool_uses:
            # No tools used, this is final output
            text_content = next((block.text for block in response.content if hasattr(block, "text")), "")

            # Log cache usage for development
            self._log_cache_usage(
                usage,
                {
                    "type": "single_request_no_tools",
                    "model": agent.model,
                    "system_prompt_length": len(agent.system_prompt) if agent.system_prompt else 0,
                },
            )

            return StepResult(content=text_content, usage=usage, next_step=NextStepFinalOutput())

        # Tools were used - execute them and prepare for next step
        # Add assistant message with tool use
        messages.append({"role": "assistant", "content": response.content})

        # Execute tools and add results
        tool_results = []
        executed_results = []
        for tool_use in tool_uses:
            # Execute tool
            tool_result = await self.tool_executor.execute(tool_use.name, tool_use.input)
            tool_results.append({"type": "tool_result", "tool_use_id": tool_use.id, "content": str(tool_result)})
            executed_results.append(f"{tool_use.name}: {tool_result}")

        messages.append({"role": "user", "content": tool_results})

        # Update agent messages with tool interaction
        agent.messages.extend(
            [
                Message(role="assistant", content=f"[Used tools: {', '.join(t.name for t in tool_uses)}]"),
                Message(role="user", content=f"[Tool results: {'; '.join(executed_results)}]"),
            ]
        )

        # Log cache usage for development
        self._log_cache_usage(
            usage,
            {
                "type": "single_request_with_tools",
                "model": agent.model,
                "tools_used": [t.name for t in tool_uses],
                "system_prompt_length": len(agent.system_prompt) if agent.system_prompt else 0,
            },
        )

        return StepResult(
            content=f"Tools executed: {', '.join(t.name for t in tool_uses)}",
            usage=usage,
            metadata={"tool_results": executed_results, "tool_count": len(tool_uses)},
            next_step=NextStepRunAgain(),
        )
