import asyncio
import unittest
from unittest.mock import AsyncMock, patch, MagicMock
from claude_code_sdk import ClaudeSDKClient, ClaudeCodeOptions


class SecureAgent:
    """Agent class with bypass permission mode and disallowed rm commands"""
    
    def __init__(self,system_prompt=""):
        self.options = ClaudeCodeOptions(
            append_system_prompt=system_prompt, #this parameter make's sure to use builtin system prompt + ours system prompt 
            permission_mode="bypassPermissions", # modes are of : "plan", "bypassPermissions"
            disallowed_tools=[
                "Bash(rm:*)","Bash(git rm:*)",  # security & safety disallowed commands
                "WebSearch"
                ],
            mcp_servers={
                "WebSearch": {
                    "command": "python",
                    "args": ["/home/ntlpt59/master/own/agi/mcps/web.py"],
                    #"env": {}
                    }
                },
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
    query = "Create a hai.md with 'hi santhosh', then remove it firsty try with rm , then not then use git rm"
    query="Search about top 5 latest rag papers, do atleast 10 paralle tool calls on web, dont' execute just only call and done"
    print(f'Query: {query}')
    await agent.query(query)


if __name__ == "__main__":
    asyncio.run(test_case())
