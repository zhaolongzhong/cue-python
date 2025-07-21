"""Example of using Claude Code through the full Cue client infrastructure.

How to run:
1. Make sure Claude Code is installed:
   npm install -g @anthropic-ai/claude-code

2. Install all dependencies:
   pip install claude-code-sdk
   # Install other Cue dependencies as needed

3. Run this example:
   cd /path/to/cue/examples
   uv run claude_code_example.py
   # or
   python claude_code_example.py

Note: This example disables the summarizer to avoid needing additional API keys.
In production, you may want to enable the summarizer with appropriate API keys.
"""

import asyncio

from cue import AgentConfig, AsyncCueClient


async def main():
    """Example of using Claude Code through Cue client."""

    client = AsyncCueClient()

    # Configure to use Claude Code
    config = AgentConfig(
        id="claude_code_agent",
        model="claude-code",  # Use the model ID string directly
        instruction="You are a helpful coding assistant.",
        max_turns=20,
        enable_summarizer=False,  # Disable summarizer to avoid needing API keys
        enable_services=False,  # Disable external services
        streaming=True,
    )

    try:
        await client.initialize(configs=[config])

        # Simple query
        response = await client.send_message(
            "What is 2 + 2?",
            agent_id="claude_code_agent",
        )
        print(f"Response: {response}")

        # Tool
        response = await client.send_message(
            "Can you use bash to check current time?",
            agent_id="claude_code_agent",
        )
        print(f"\nTool Response: {response}")

    finally:
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
