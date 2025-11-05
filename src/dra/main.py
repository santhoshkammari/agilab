"""
DRA Main - Using MCP web search tool with LLM
Direct integration with FastMCP web search tool
"""
import sys
import os
import asyncio
import json

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm.vllm import vLLM


async def async_main():
    print("🚀 Starting DRA with Web Search MCP Tool...")

    # Import the web search tool directly from dra/mcp/web.py
    # Add dra/mcp to path
    mcp_path = os.path.join(os.path.dirname(__file__), "mcp")
    sys.path.insert(0, mcp_path)

    from web import async_web_search

    print("✅ Web search tool loaded successfully!")

    # Define web search tool function for LLM
    async def web_search_tool(query: str, max_results: int = 3):
        """Search the web for information"""
        results = await async_web_search(query=query, max_results=max_results)
        return results

    # Initialize vLLM with the web search tool
    llm = vLLM(
        model="/home/ng6309/datascience/santhosh/models/Qwen3-14B",
        base_url="http://localhost:8000/v1"
    )

    # Test query
    query = "What are the latest developments in AI agents?"
    print(f"\n🔍 User Query: {query}")

    # Call web search tool
    print(f"\n🔧 Calling web search tool...")
    search_results = await web_search_tool(query, max_results=3)

    print(f"\n📊 Search Results:")
    for idx, result in enumerate(search_results, 1):
        print(f"  {idx}. {result['title']}")
        print(f"     URL: {result['url']}")
        print(f"     {result['description'][:100]}...")
        print()

    # Format search results for LLM
    results_text = "\n\n".join([
        f"Result {idx}:\nTitle: {r['title']}\nURL: {r['url']}\nDescription: {r['description']}"
        for idx, r in enumerate(search_results, 1)
    ])

    # Create prompt with search results
    prompt = f"""Based on these web search results, answer the user's question comprehensively.

User Question: {query}

Search Results:
{results_text}

Provide a detailed answer using the information from the search results."""

    messages = [{"role": "user", "content": prompt}]

    print(f"\n🤖 Asking LLM to analyze search results...")
    response = llm(messages)

    print(f"\n💬 LLM Response:")
    if response['think']:
        print(f"\n[Thinking]: {response['think']}")
    print(f"\n{response['content']}")

    print("\n✨ Done!")


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
