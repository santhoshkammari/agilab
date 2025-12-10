import asyncio
from claude_agent_sdk import query, ClaudeAgentOptions

async def main():
    options = ClaudeAgentOptions(
    )

    async for message in query(
        prompt="/usage",
        options=options
    ):
        print(message)


asyncio.run(main())
