import sys
import os
import asyncio
os.environ.pop("CLAUDECODE", None)
from claude_agent_sdk import (
    query, ClaudeAgentOptions,
    AssistantMessage, UserMessage, SystemMessage, ResultMessage,
    TextBlock, ToolUseBlock, ToolResultBlock, ThinkingBlock,
)

async def main():
    prompt = " ".join(sys.argv[1:])
    options = ClaudeAgentOptions(permission_mode="bypassPermissions")
    async for msg in query(prompt=prompt, options=options):
        if isinstance(msg, AssistantMessage):
            for block in msg.content:
                if isinstance(block, TextBlock):
                    print(block.text, flush=True)
                elif isinstance(block, ThinkingBlock):
                    print(f"[thinking] {block.thinking}", flush=True)
                elif isinstance(block, ToolUseBlock):
                    print(f"[tool_use] {block.name}({block.input})", flush=True)
                elif isinstance(block, ToolResultBlock):
                    print(f"[tool_result] {block.content}", flush=True)
        elif isinstance(msg, UserMessage):
            # tool results come back as UserMessages with ToolResultBlock content
            if isinstance(msg.content, list):
                for block in msg.content:
                    if isinstance(block, ToolResultBlock):
                        print(f"[tool_result] {block.content}", flush=True)
        elif isinstance(msg, SystemMessage):
            if msg.subtype == "init":
                session_id = msg.data.get("session_id", "")
                print(f"[session] {session_id}", flush=True, file=sys.stderr)
        elif isinstance(msg, ResultMessage):
            if msg.result:
                print(msg.result, flush=True)
            print(
                f"[done] turns={msg.num_turns} cost=${msg.total_cost_usd:.4f} "
                f"err={msg.is_error}",
                flush=True, file=sys.stderr,
            )

asyncio.run(main())
