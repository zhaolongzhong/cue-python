from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from .types import RunResult, SimpleAgent


@dataclass
class StreamEvent:
    """Simple streaming event with accumulated content tracking"""

    type: str  # "text", "tool_start", "tool_end", "agent_done"
    content: str = ""
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    @property
    def accumulated(self) -> str:
        """Get accumulated content from metadata"""
        return self.metadata.get("accumulated", self.content)

    @property
    def is_final(self) -> bool:
        """Check if this is the final event"""
        return self.metadata.get("final", False)

    @property
    def tool_results(self) -> List[Dict[str, Any]]:
        """Get all tool results from metadata"""
        return self.metadata.get("tool_results", [])


class StreamingHooks(ABC):
    """Simple hooks for streaming events"""

    @abstractmethod
    async def on_stream_start(self, agent: SimpleAgent) -> None:
        """Called when streaming starts"""
        pass

    async def on_text_chunk(self, chunk: str, agent: SimpleAgent) -> Optional[str]:
        """Called for each text chunk. Return modified chunk or None to skip"""
        return chunk

    @abstractmethod
    async def on_tool_start(self, tool_name: str, arguments: Dict[str, Any], agent: SimpleAgent) -> None:
        """Called when tool execution starts"""
        pass

    async def on_tool_end(self, tool_name: str, result: str, agent: SimpleAgent) -> Optional[str]:
        """Called when tool execution ends. Return modified result or None"""
        return result

    @abstractmethod
    async def on_stream_end(self, agent: SimpleAgent, final_result: RunResult) -> None:
        """Called when streaming completes"""
        pass


class DefaultStreamingHooks(StreamingHooks):
    """Default no-op implementation"""

    pass


class LoggingStreamingHooks(StreamingHooks):
    """Example logging hooks"""

    async def on_stream_start(self, agent: SimpleAgent) -> None:
        print(f"🚀 Starting stream for {agent.model}")

    async def on_tool_start(self, tool_name: str, arguments: Dict[str, Any], agent: SimpleAgent) -> None:
        print(f"🔧 Tool {tool_name} starting...")

    async def on_tool_end(self, tool_name: str, result: str, agent: SimpleAgent) -> Optional[str]:
        print(f"✅ Tool {tool_name} completed")
        return result

    async def on_stream_end(self, agent: SimpleAgent, final_result: RunResult) -> None:
        print(f"🏁 Stream completed for {agent.model}")
