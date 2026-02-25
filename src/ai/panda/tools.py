"""
Panda Agent tools — SEARCH, STORE, QUERY. That's it.
The agent code handles scratchpad. The LLM just executes.
"""
import sys
import os
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'mcp_tools'))

from research.memory import (
    memory_store as _raw_store,
    memory_search as _raw_search,
)

_current_task: str = ""


def set_current_task(task: str):
    global _current_task
    _current_task = task


def search(query: str, max_results: int = 5, fetch: bool = True) -> str:
    """Search the web. Returns titles, URLs, descriptions. Auto-scrapes all URLs and stores content in 'memory' collection.

    Args:
        query: The search query string
        max_results: Maximum number of search results to return (default 5)
        fetch: If True (default), auto-scrape all result URLs and store content in memory

    Returns:
        Search results with titles, URLs, descriptions. Scraped content is auto-stored.
    """
    from ddgs import DDGS

    try:
        results = list(DDGS().text(query, max_results=max_results))
    except Exception as e:
        return f"Search error: {e}"

    if not results:
        return "No search results found."

    lines = []
    for i, r in enumerate(results, 1):
        url = r.get('href', '')
        title = r.get('title', '')
        desc = r.get('body', '')
        lines.append(f"[{i}] {title}\n    URL: {url}\n    {desc}")

    if fetch:
        from fetch import scrapling_get
        scraped = 0
        for r in results:
            url = r.get('href', '')
            if not url:
                continue
            try:
                result = scrapling_get(url, extraction_type="markdown")
                if result.get("content") and result.get("status") == 200:
                    markdown = "".join(result["content"])
                    if markdown.strip():
                        now = datetime.now(timezone.utc).isoformat()
                        _raw_store(
                            collection="memory",
                            text=markdown,
                            metadata={
                                "source": url,
                                "type": "scraped_content",
                                "timestamp": now,
                                "task": _current_task,
                            },
                            id=url,
                        )
                        scraped += 1
            except Exception:
                pass
        lines.append(f"\n[FETCHED] {scraped}/{len(results)} URLs scraped and stored in memory")

    return "\n\n".join(lines)


def store(collection: str, text: str, source: str = "", type: str = "finding") -> str:
    """Store a document in a ChromaDB collection with standardized metadata.

    Args:
        collection: ChromaDB collection name ('mind' for insights, 'memory' for raw data)
        text: The text content to store
        source: Where this came from (URL, user, file, etc.)
        type: What this is (search_result, scraped_content, user_input, finding, link, insight)

    Returns:
        Confirmation with the document ID.
    """
    now = datetime.now(timezone.utc).isoformat()
    return _raw_store(
        collection=collection,
        text=text,
        metadata={
            "source": source,
            "type": type,
            "timestamp": now,
            "task": _current_task,
        },
    )


def query(collection: str, query_text: str, n: int = 5) -> str:
    """Semantic search across a ChromaDB collection. Use this to recall what you know.

    Args:
        collection: ChromaDB collection name ('mind' for insights, 'memory' for raw data)
        query_text: The search query (semantic similarity search)
        n: Maximum number of results to return (default 5)

    Returns:
        Matching documents with text, metadata, and distance scores.
    """
    return _raw_search(collection=collection, query=query_text, n=n)


_answer_path: str = "answer.md"


def set_answer_path(path: str):
    global _answer_path
    _answer_path = path


def answer(content: str) -> str:
    """Append content to answer.md. Use this to build up your final deliverable — summaries, insights, reports. Each call appends to the file.

    Args:
        content: Text to append (markdown format).

    Returns:
        Confirmation message.
    """
    with open(_answer_path, 'a') as f:
        f.write(content + "\n\n")
    return f"Appended to {_answer_path} ({len(content)} chars)"


ALL_TOOLS = [search, store, query, answer]
