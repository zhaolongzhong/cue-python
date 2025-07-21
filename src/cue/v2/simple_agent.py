import asyncio
from typing import Any, Dict, List, Callable, Optional
from dataclasses import field, dataclass

from .simple_llm_client import SimpleMessage, SimpleLLMClient


@dataclass
class SimpleStatus:
    status: str  # "thinking", "responding", "done", "error"
    message: str = ""
    data: Dict[str, Any] = field(default_factory=dict)


StatusCallback = Callable[[SimpleStatus], None]


class SimpleAgent:
    def __init__(
        self,
        model: str,
        system_prompt: str = "",
        api_key: Optional[str] = None,
        status_callback: Optional[StatusCallback] = None,
    ):
        self.llm_client = SimpleLLMClient(model, api_key)
        self.messages: List[SimpleMessage] = []
        self.system_prompt = system_prompt
        self.status_callback = status_callback

        if system_prompt:
            self.messages.append(SimpleMessage(role="system", content=system_prompt))

    def _notify_status(self, status: str, message: str = "", **data):
        if self.status_callback:
            self.status_callback(SimpleStatus(status, message, data))

    async def chat(self, user_message: str) -> str:
        self._notify_status("thinking", "Processing your message")

        try:
            # Add user message
            self.messages.append(SimpleMessage(role="user", content=user_message))

            # Get response
            self._notify_status("responding", "Generating response")
            response = await self.llm_client.complete(self.messages)

            # Add assistant response
            self.messages.append(SimpleMessage(role="assistant", content=response.content))

            self._notify_status("done", "Response completed", usage=response.usage, model=response.model)

            return response.content

        except Exception as e:
            self._notify_status("error", f"Error: {str(e)}")
            raise

    async def stream_chat(self, user_message: str):
        self._notify_status("thinking", "Processing your message")

        try:
            # Add user message
            self.messages.append(SimpleMessage(role="user", content=user_message))

            # Stream response
            self._notify_status("responding", "Streaming response")
            full_response = ""

            async for chunk in self.llm_client.stream_complete(self.messages):
                full_response += chunk
                yield chunk

            # Add complete assistant response
            self.messages.append(SimpleMessage(role="assistant", content=full_response))

            self._notify_status("done", "Streaming completed")

        except Exception as e:
            self._notify_status("error", f"Error: {str(e)}")
            raise

    def reset(self):
        """Clear conversation history but keep system prompt"""
        self.messages = []
        if self.system_prompt:
            self.messages.append(SimpleMessage(role="system", content=self.system_prompt))
        self._notify_status("done", "Conversation reset")

    def get_history(self) -> List[SimpleMessage]:
        """Get conversation history"""
        return self.messages.copy()

    def set_system_prompt(self, prompt: str):
        """Update system prompt and reset conversation"""
        self.system_prompt = prompt
        self.reset()


class SimpleAgentLoop:
    def __init__(self, agent: SimpleAgent):
        self.agent = agent
        self.running = False
        self.stop_event = asyncio.Event()

    async def run_until_complete(
        self, initial_message: str, max_turns: int = 10, continue_condition: Optional[Callable[[str], bool]] = None
    ) -> List[str]:
        """Simple loop that continues until a condition is met or max turns reached"""
        self.running = True
        responses = []
        current_message = initial_message

        try:
            for _turn in range(max_turns):
                if self.stop_event.is_set():
                    break

                response = await self.agent.chat(current_message)
                responses.append(response)

                # Check if we should continue
                if continue_condition and not continue_condition(response):
                    break

                # For simplicity, we'll just stop after first response unless
                # continue_condition explicitly says to continue
                if not continue_condition:
                    break

                # Use the response as next input if continuing
                current_message = response

        except Exception as e:
            self.agent._notify_status("error", f"Loop error: {str(e)}")
            raise
        finally:
            self.running = False

        return responses

    def stop(self):
        """Stop the agent loop"""
        self.stop_event.set()
        self.running = False
