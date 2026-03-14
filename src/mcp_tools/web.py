import logging
from typing import Dict, List
from duckduckgo_search import DDGS
from fastmcp import FastMCP

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def search_web_duckduckgo(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    try:
        with DDGS() as ddgs:
            results = []
            for r in ddgs.text(query, max_results=max_results):
                results.append(
                    {
                        "url": r.get("href", ""),
                        "title": r.get("title", ""),
                        "description": r.get("body", ""),
                    }
                )
            return results
    except Exception as e:
        logger.error(f"DuckDuckGo search error: {str(e)}")
        return []


async def async_web_search_duckduckgo(
    query: str, max_results: int = 5
) -> List[Dict[str, str]]:
    return search_web_duckduckgo(query, max_results)


mcp = FastMCP("DuckDuckGo Search Server")


@mcp.tool
async def web_search_duckduckgo(query: str, max_results: int = 5) -> dict:
    try:
        results = await async_web_search_duckduckgo(query, max_results)
        return {"query": query, "results": results}
    except Exception as e:
        return {"error": str(e)}


tool_functions = {
    "async_web_search_duckduckgo": async_web_search_duckduckgo,
}

if __name__ == "__main__":
    mcp.run()
