from typing import List, Union, Optional, AsyncGenerator

from .types import Session, InputItem, RunResult, SimpleAgent, NextStepRunAgain, NextStepFinalOutput
from .model_base import ModelBase
from .streaming_hooks import StreamEvent, StreamingHooks


class AgentRunner:
    """Agent runner for multi-turn conversations using ModelBase

    This class orchestrates conversations between user and AI models:
    - Manages conversation state and memory
    - Coordinates multiple turns with the underlying model
    - Handles conversation-level logic and flow control

    Currently delegates to ModelBase for single-turn behavior.
    Future enhancements can add sophisticated multi-turn logic here.
    """

    def __init__(self, model: ModelBase):
        self.model = model

        # Cache usage logging (create new instance for proper test isolation)
        from .cache_logger import CacheLogger

        self.cache_logger = CacheLogger()

    def _log_conversation_usage(self, usage: dict, agent: SimpleAgent, steps: list, metadata: dict = None):
        """Log conversation-level cache usage statistics"""
        self.cache_logger.log_conversation_usage(usage, agent, steps, metadata)

    async def run(
        self,
        agent: SimpleAgent,
        session_or_input: Union[Session, List[InputItem]],
        input_content: Optional[str] = None,
        max_turns: Optional[int] = None,
    ) -> RunResult:
        """Run multi-turn conversation until completion

        Args:
            agent: The agent configuration
            session_or_input: Either a Session object or List[InputItem] for backward compatibility
            input_content: New input content to add to session (only used if session_or_input is Session)
            max_turns: Maximum number of turns (defaults to agent.max_turns)

        Returns:
            RunResult with final content and all intermediate steps
        """
        max_turns = max_turns or agent.max_turns
        steps = []
        accumulated_usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        }

        # Handle both session-based and legacy input_items-based calls
        if isinstance(session_or_input, list):
            # Legacy mode: input_items passed directly
            input_items = session_or_input
            session = None
        else:
            # Session mode: get items from session and optionally add new input
            session = session_or_input
            if input_content:
                await session.add_items([InputItem(type="text", content=input_content)])
            input_items = await session.get_items()

        # Multi-turn loop
        for turn in range(max_turns):
            # Get single step result from model
            step_result = await self.model.get_response(agent, input_items)
            steps.append(step_result)

            # Accumulate usage - include any cache-related tokens from model response
            for key in accumulated_usage:
                accumulated_usage[key] += step_result.usage.get(key, 0)

            # Also accumulate any additional usage keys from step_result (e.g., other cache tokens)
            for key, value in step_result.usage.items():
                if key not in accumulated_usage:
                    accumulated_usage[key] = value
                # Note: Keys already in accumulated_usage were handled above

            # Check what to do next
            if isinstance(step_result.next_step, NextStepFinalOutput):
                # We're done - update session with final result and return
                if session and step_result.content:
                    await session.add_items([InputItem(type="text", content=step_result.content)])

                result = RunResult(
                    content=step_result.content, steps=steps, usage=accumulated_usage, metadata={"turns": turn + 1}
                )

                # Log conversation-level cache usage
                self._log_conversation_usage(accumulated_usage, agent, steps, {"turns": turn + 1})

                return result
            elif isinstance(step_result.next_step, NextStepRunAgain):
                # Tools were used - update input_items and continue
                if session:
                    # For session mode, we need to add tool interactions to session
                    # The model updated agent.messages, so we sync the new messages to session
                    current_session_items = await session.get_items()

                    # Add tool interaction messages to session (they're in agent.messages but not session)
                    # Get the latest messages that aren't in session yet
                    if len(agent.messages) > len(current_session_items):
                        new_messages = agent.messages[len(current_session_items) :]
                        new_items = [InputItem(type="text", content=msg.content) for msg in new_messages]
                        await session.add_items(new_items)

                    input_items = await session.get_items()
                else:
                    # For legacy mode, we need to manually update input_items from agent.messages
                    # Use the last few messages as the new input
                    input_items = [InputItem(type="text", content=msg.content) for msg in agent.messages[-2:]]
                continue

        # Max turns reached
        if session:
            await session.add_items([InputItem(type="text", content="Max turns reached")])

        result = RunResult(
            content=steps[-1].content if steps else "Max turns reached",
            steps=steps,
            usage=accumulated_usage,
            metadata={"max_turns_reached": True, "turns": max_turns},
        )

        # Log conversation-level cache usage
        self._log_conversation_usage(accumulated_usage, agent, steps, {"max_turns_reached": True, "turns": max_turns})

        return result

    async def run_streamed(
        self,
        agent: SimpleAgent,
        session_or_input: Union[Session, List[InputItem]],
        input_content: Optional[str] = None,
        max_turns: Optional[int] = None,
        hooks: Optional[StreamingHooks] = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream multi-turn conversation until completion

        Args:
            agent: The agent configuration
            session_or_input: Either a Session object or List[InputItem] for backward compatibility
            input_content: New input content to add to session (only used if session_or_input is Session)
            max_turns: Maximum number of turns (defaults to agent.max_turns)
            hooks: Optional streaming hooks

        Yields:
            StreamEvent objects with step boundaries marked
        """
        max_turns = max_turns or agent.max_turns

        # Handle both session-based and legacy input_items-based calls
        if isinstance(session_or_input, list):
            # Legacy mode: input_items passed directly - just stream single response
            async for event in self.model.stream_response(agent, session_or_input, hooks):
                yield event
            return

        # Session mode: get items from session and optionally add new input
        session = session_or_input
        if input_content:
            await session.add_items([InputItem(type="text", content=input_content)])

        input_items = await session.get_items()

        # Multi-turn streaming loop
        for turn in range(max_turns):
            yield StreamEvent(type="step_start", content="", metadata={"turn": turn + 1})

            # Stream single step from model
            accumulated_content = ""
            async for event in self.model.stream_response(agent, input_items, hooks):
                if event.type == "text" and event.content:
                    accumulated_content += event.content
                yield event

            # Determine next step based on final event or accumulated content
            # For now, we assume if accumulated_content exists, it's final output
            # Models will need to emit proper step metadata in the future
            if accumulated_content and not any(
                keyword in accumulated_content.lower() for keyword in ["tool", "executed", "bash", "edit"]
            ):
                # Looks like final text output
                if session:
                    await session.add_items([InputItem(type="text", content=accumulated_content)])

                yield StreamEvent(
                    type="conversation_done", content=accumulated_content, metadata={"turns": turn + 1, "final": True}
                )
                return
            else:
                # Likely tool use - continue to next turn
                if session:
                    input_items = await session.get_items()
                else:
                    # For legacy mode, update from agent.messages
                    input_items = [InputItem(type="text", content=msg.content) for msg in agent.messages[-2:]]

                yield StreamEvent(type="step_end", content="", metadata={"turn": turn + 1, "continue": True})
                continue

        # Max turns reached
        if session:
            await session.add_items([InputItem(type="text", content="Max turns reached")])

        yield StreamEvent(
            type="conversation_done",
            content="Max turns reached",
            metadata={"max_turns_reached": True, "turns": max_turns},
        )
