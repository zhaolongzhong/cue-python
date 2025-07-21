#!/usr/bin/env python3
"""
Streaming with Tools and Hooks

Demonstrates the new simplified streaming approach:
- Streams text chunks as they arrive
- Handles tool calls during streaming
- Uses hooks for extensibility

Run example:

```
export ANTHROPIC_API_KEY=sk-ant-ap...
uv run streaming_with_tools.py
```

"""

import os
import sys
import asyncio

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cue.v2 import InputItem, SimpleAgent, StreamingHooks, LoggingStreamingHooks, get_runner_for_model

env_file = ".env.development"
# test_model = "claude-3-5-haiku-20241022"
test_model = "claude-sonnet-4-20250514"


class CustomStreamingHooks(StreamingHooks):
    """Custom hooks for demonstration"""

    def __init__(self):
        self.events = []

    async def on_stream_start(self, agent):
        self.events.append("stream_start")
        print(f"🚀 Stream starting for {agent.model}")

    async def on_text_chunk(self, chunk, agent):
        self.events.append("text_chunk")
        # Could modify text here, add colors, etc.
        return chunk

    async def on_tool_start(self, tool_name, arguments, agent):
        self.events.append("tool_start")
        print(f"\n🔧 Tool {tool_name} starting with args: {arguments}")

    async def on_tool_end(self, tool_name, result, agent):
        self.events.append("tool_end")
        print(f"✅ Tool {tool_name} result: {result[:50]}...")
        return result

    async def on_stream_end(self, agent, final_result):
        self.events.append("stream_end")
        print(f"\n🏁 Stream completed. Events: {len(self.events)}")


async def test_streaming_text_only():
    """Test streaming without tools"""
    print("=== Testing Streaming Text Only ===")

    agent = SimpleAgent(model=test_model, system_prompt="Be helpful and concise.")

    runner = get_runner_for_model(agent.model)
    hooks = CustomStreamingHooks()

    try:
        print("Assistant: ", end="", flush=True)

        async for event in runner.stream_response(
            agent, [InputItem(type="text", content="Say hello in exactly 5 words")], hooks=hooks
        ):
            if event.type == "text":
                print(event.content, end="", flush=True)
            elif event.type == "agent_done":
                print("\n")

        print(f"Events captured: {hooks.events}")
        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


async def test_streaming_with_tools():
    """Test streaming with tool execution"""
    print("\n=== Testing Streaming with Tools ===")

    agent = SimpleAgent(model=test_model, system_prompt="Use tools when you need system information. Be helpful.")

    runner = get_runner_for_model(agent.model)
    hooks = LoggingStreamingHooks()  # Use built-in logging hooks

    try:
        print("Assistant: ", end="", flush=True)

        async for event in runner.stream_response(
            agent,
            [
                InputItem(
                    type="text",
                    content="What's the current time? Use bash date to get accurate info. Run bash command: date",
                )
            ],
            hooks=hooks,
        ):
            print(f"test_streaming_with_tools event: {event}")
            if event.type == "text":
                print(event.content, end="", flush=True)
            elif event.type == "tool_start":
                print(f"\n[🔧 {event.metadata.get('tool_name', 'tool')}]", end="", flush=True)
            elif event.type == "tool_end":
                print(f"\n[✅ {event.metadata.get('tool_name', 'tool')} completed]", end="", flush=True)
            elif event.type == "agent_done":
                print("\n")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


async def main():
    # Load environment variables
    env_path = os.path.join(os.path.dirname(__file__), "..", ".env.development")
    print(f"inx env_path: {env_path}")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key] = value.strip("\"'")

    # Run tests
    tests = [test_streaming_text_only(), test_streaming_with_tools()]

    success_count = 0
    for test in tests:
        if await test:
            success_count += 1
    print(f"\n📊 Tests: {success_count}/{len(tests)} passed")


if __name__ == "__main__":
    asyncio.run(main())
