import asyncio
import unittest
from unittest.mock import AsyncMock, patch, MagicMock
from claude_code_sdk import ClaudeSDKClient, ClaudeCodeOptions


class SecureAgent:
    """Agent class with bypass permission mode and disallowed rm commands"""
    
    def __init__(self):
        self.options = ClaudeCodeOptions(
            #permission_mode="bypassPermissions",
            permission_mode="default",
            disallowed_tools=["Task","ExitPlanMode","KillBash","BaseOutput","WebSearch","NotebookEdit","Edit","WebFetch","TodoWrite","Bash(rm*)","Bash(rm *)","rm", "Bash(sudo rm*)", "Bash(*rm*)"],
            #permission_prompt_tool_name="mcp__approval_tool",
        )
    
    async def query(self, prompt: str):
        """Execute a query with the secure agent configuration"""

        try:
            async with ClaudeSDKClient(options=self.options) as client:
                await client.query(prompt)

                response_text = []
                async for message in client.receive_response():
                    print('---')
                    print(type(message))
                    print(message)
        except Exception as e:
            print(f"An error occurred: {e}")


async def test_case():
    """Test secure agent with file creation and removal query"""
    agent = SecureAgent()
    query = "Create a hai.md with 'hi santhosh', then remove it"
    #query = "Create a hai.md with 'hi santhosh',just oneshot"
    print(f'Query: {query}')
    await agent.query(query)


if __name__ == "__main__":
    asyncio.run(test_case())
