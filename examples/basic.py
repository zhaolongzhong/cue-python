import asyncio

from cue import ChatModel, AgentConfig, AsyncCueClient


async def main():
    config = AgentConfig(
        # api_key=api_key,
        model=ChatModel.GPT_4O_MINI.id,
    )
    client = AsyncCueClient()

    try:
        await client.initialize(configs=[config])
        response = await client.send_message("Hello, there!")
        print(f"{response}")
    finally:
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
