"""Simple example of using Claude Code SDK directly without Cue infrastructure.

How to run:
1. Make sure Claude Code is installed:
   npm install -g @anthropic-ai/claude-code

2. Install the Python SDK:
   pip install claude-code-sdk

3. Run this example:
   cd /path/to/cue/examples
   uv run claude_code_simple.py
   # or
   python claude_code_simple.py
"""

import asyncio

from claude_code_sdk import TextBlock, AssistantMessage, ClaudeCodeOptions, query


async def main():
    # Simple query
    print("=== Simple Query ===")
    async for message in query(prompt="What is 2 + 2?"):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    print(block.text)

    # Code generation query
    print("\n=== Code Generation ===")
    options = ClaudeCodeOptions(
        allowed_tools=["Read", "Write", "Bash"],
        permission_mode="acceptEdits",
        system_prompt="You are a helpful coding assistant.",
        max_turns=5,
    )

    async for message in query(
        # prompt="Write a Python function to calculate fibonacci numbers and provide me the path",
        prompt="Can you use bash to check current time?",
        options=options,
    ):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                print(block)


if __name__ == "__main__":
    asyncio.run(main())
