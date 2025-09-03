import uuid
import asyncio
import unittest
from dataclasses import dataclass, asdict
from unittest.mock import AsyncMock, patch, MagicMock
from claude_code_sdk import ClaudeSDKClient, ClaudeCodeOptions
from claude_code_sdk import CLINotFoundError, ProcessError


def to_dict_with_type(obj):
    return {
        "type": obj.__class__.__name__,
        **asdict(obj)
    }


async def claude_code(prompt: str, session_id:str|None=None, mode: str = "Scout", cwd: str|None=None, append_system_prompt: str = ""):
    """Execute a query with the secure agent configuration"""
    options = ClaudeCodeOptions(
        cwd=cwd, #by default it runs in scripts current directory
        append_system_prompt=append_system_prompt, #this parameter make's sure to use builtin system prompt + ours system prompt
        resume=session_id, #str(uuid.uuid4()), #Uses session-id specific message as context/conversation maintainence , None will autocreates session-uid
        permission_mode="plan" if mode == "Plan" else "bypassPermissions", # modes are of : "plan", "bypassPermissions"
        disallowed_tools=[
            "Bash(rm:*)","Bash(git rm:*)",  # security & safety disallowed commands
            "WebSearch", # disabling default WebSearch for TokenEfficiency and leveraging custom web Search mcp server
            "WebFetch", # disabling default WebFetch for TokenEfficiency and leveraging custom web Fetch mcp server
            "ExitPlanMode", # disabling default ExitPlanModel for TokenEfficiency
            "Task",
            "NotebookEdit"
            ],
        mcp_servers={
            "WebSearch": {
                "command": "python",
                "args": ["tools/web.py"],
                #"env": {}
                },
            "WebFetch": {
                "command": "python",
                "args": ["tools/fetch.py"],
                #"env": {}
                }
            },
    )

    client = ClaudeSDKClient(options)
    await client.connect()
    await client.query(prompt)
    
    async for message in client.receive_response():
        yield to_dict_with_type(message)
    
    await client.disconnect()


async def test_case():
    """Test secure agent with file creation and removal query"""
    #query = "Create a hai.md with 'hi santhosh', then remove it firsty try with rm , then not then use git rm"
    query="Search about top 5 latest rag papers, do atleast 2 paralle tool calls on web, dont' execute just only call and done"

    print(f'Query: {query}')
    async for x in claude_code(query,session_id=None):
        print(x)
        print('----')

    ''' Example using sessionid
    async for x in claude_code(query,session_id="280cc32d-ea03-4d9d-8b1e-9a774f326f49"):
        print(x)
        print('----')
    '''

if __name__ == "__main__":
    asyncio.run(test_case())
