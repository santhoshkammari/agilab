"""
DRA Main - Using MCP web search tool with LLM
Direct integration with FastMCP web search tool
"""
import sys
import os
import asyncio
import json
import requests

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def async_main():
    print("🚀 Starting DRA with Web Search MCP Tool...")

    # Import the web search tool directly from dra/mcp/web.py
    # Add dra/mcp to path
    mcp_path = os.path.join(os.path.dirname(__file__), "mcp")
    sys.path.insert(0, mcp_path)

    from web import async_web_search

    print("✅ Web search tool loaded successfully!")

    # Test query
    query = "What are the latest developments in AI agents?"
    print(f"\n🔍 Searching: {query}")

    # Call the web search tool
    search_results = await async_web_search(query=query, max_results=3)
    print(f"\n📊 Search Results:")
    print(json.dumps(search_results, indent=2))

    # Format search results for LLM
    results_text = "\n\n".join([
        f"Title: {r['title']}\nURL: {r['url']}\nDescription: {r['description']}"
        for r in search_results
    ])

    # Now use LLM with the search results
    prompt = f"""Based on these search results, summarize the latest developments in AI agents:

{results_text}

Provide a concise summary highlighting the key developments."""

    print(f"\n🤖 Asking LLM to summarize...")

    # Call vLLM API directly
    vllm_response = requests.post(
        "http://localhost:8000/v1/chat/completions",
        json={
            "model": "/home/ng6309/datascience/santhosh/models/Qwen3-14B",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 500
        }
    )

    response_data = vllm_response.json()
    llm_answer = response_data["choices"][0]["message"]["content"]

    print(f"\n💬 LLM Response:")
    print(llm_answer)

    print("\n✨ Done!")


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
