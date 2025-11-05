"""
DRA Main - LLM with tool calling capability
Let the LLM decide when to call the web search tool
"""
import sys
import os
import asyncio
import json

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm.vllm import vLLM


async def async_main():
    print("🚀 Starting DRA with Web Search Tool Calling...")

    # Import the web search tool directly from dra/mcp/web.py
    mcp_path = os.path.join(os.path.dirname(__file__), "mcp")
    sys.path.insert(0, mcp_path)

    from web import async_web_search

    print("✅ Web search tool loaded successfully!")

    # Define web search tool function for LLM
    async def web_search_tool(query: str, max_results: int = 3):
        """Search the web for information about a given query"""
        results = await async_web_search(query=query, max_results=max_results)
        return results

    # Initialize vLLM
    llm = vLLM(
        model="/home/ng6309/datascience/santhosh/models/Qwen3-14B",
        base_url="http://localhost:8000/v1"
    )

    # User query - let LLM decide if it needs to search
    query = "What are the latest developments in AI agents?"
    print(f"\n🔍 User Query: {query}")

    messages = [{"role": "user", "content": query}]

    print(f"\n🤖 Calling LLM with web search tool available...")

    # Pass the tool to LLM - it will decide whether to call it
    response = llm(messages, tools=[web_search_tool])

    print(f"\n📋 Initial LLM Response:")
    if response['think']:
        print(f"[Thinking]: {response['think']}")
    if response['content']:
        print(f"Content: {response['content']}")

    # Check if LLM wants to use the tool
    if response['tool_calls']:
        print(f"\n🔧 LLM requested {len(response['tool_calls'])} tool call(s)!")

        # Add assistant message with tool calls
        messages.append({
            "role": "assistant",
            "content": response['content'],
            "tool_calls": response['tool_calls']
        })

        # Execute each tool call
        for tool_call in response['tool_calls']:
            func_name = tool_call['function']['name']
            func_args = json.loads(tool_call['function']['arguments'])

            print(f"\n  → Executing: {func_name}")
            print(f"    Arguments: {func_args}")

            # Execute the tool
            if func_name == 'web_search_tool':
                search_results = await web_search_tool(**func_args)

                print(f"\n  📊 Search Results ({len(search_results)} found):")
                for idx, result in enumerate(search_results, 1):
                    print(f"    {idx}. {result['title']}")
                    print(f"       {result['url']}")
                    print(f"       {result['description'][:80]}...")

                # Add tool result to messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call['id'],
                    "content": json.dumps(search_results)
                })

        # Get final response from LLM with tool results
        print(f"\n🤖 Getting final answer from LLM...")
        final_response = llm(messages)

        print(f"\n💬 Final LLM Response:")
        if final_response['think']:
            print(f"\n[Thinking]: {final_response['think']}")
        print(f"\n{final_response['content']}")
    else:
        print(f"\n💬 LLM answered directly without using tools")

    print("\n✨ Done!")


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
