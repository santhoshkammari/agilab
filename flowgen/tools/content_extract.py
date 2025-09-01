import json
from dataclasses import asdict
from typing import List

import requests
from pydantic import BaseModel
from liteauto.parselite import parse
from scrapling.fetchers import Fetcher
from scrapling.core.shell import Convertor

SYSTEM_PROMPT = """You are a content extraction assistant. You have access to the following tools for extracting and processing web content:

- web_fetch: Fetches content from a specified URL and converts HTML to markdown format. Takes a URL as input, fetches the URL content, converts HTML to markdown, and saves it to a .claudecode directory. Use this tool when you need to retrieve and analyze web content.

Use this tool to extract, parse, and convert web content as requested by the user."""

def extract_markdown_from_url(url: str):
    """Extract markdown content directly from URL using scrapling"""
    content = get_url_content_as_markdown(url)
    return {"url": url, "content": content}


def get_url_content_as_markdown(url: str) -> str | None:
    """
    Fetches the content of a given URL and returns it as markdown.

    Args:
        url (str): The URL to fetch.

    Returns:
        str: The content of the URL as markdown text.
    """
    try:
        response = Fetcher.get(url, timeout=30)
        if response.status != 200:
            return None
        
        # Extract content as markdown using scrapling's built-in converter
        markdown_content = list(Convertor._extract_content(
            response, 
            extraction_type='markdown',
            main_content_only=True
        ))
        
        return ''.join(markdown_content) if markdown_content else None
    except Exception as e:
        return None


from typing import List, Dict
import requests

def web_search(query: str) -> List[Dict]:
    """
    Search the web using the local search API.

    Args:
        query (str): Search query.

    Returns:
        List[Dict]: A list of search results, each containing
                    'url', 'title', and 'description'.
    """
    try:
        res = requests.post(
            url="http://0.0.0.0:8999/search",
            params={"query": query},
            timeout=10
        )
        res.raise_for_status()
        results = res.json()

        return results.get('results',[])
    except (requests.RequestException, ValueError) as e:
        print(f"Error during search: {e}")
        return []

def web_fetch(url: str):
    """Fetch URL and convert to markdown using scrapling's built-in conversion
    Args:
        url: the URL string to fetch
    """
    import re
    import urllib.parse
    from datetime import datetime
    from pathlib import Path
    
    markdown_data = extract_markdown_from_url(url)
    
    if not markdown_data['content']:
        return "Failed to fetch content from URL"
    
    # Create .claudecode directory if it doesn't exist
    Path(".claudecode").mkdir(exist_ok=True)
    
    # Generate a proper filename from URL
    parsed_url = urllib.parse.urlparse(url)
    
    # Start with domain name
    domain = parsed_url.netloc.replace('www.', '')
    
    # Get the path part and clean it
    path = parsed_url.path.strip('/')
    if path:
        # Replace path separators and clean special chars
        path_clean = re.sub(r'[^\w\-_.]', '_', path.replace('/', '_'))
        # Remove multiple underscores
        path_clean = re.sub(r'_+', '_', path_clean).strip('_')
        base_name = f"{domain}_{path_clean}"
    else:
        base_name = domain
    
    # Remove common file extensions to avoid double extensions
    base_name = re.sub(r'\.(html?|php|asp|jsp)$', '', base_name, flags=re.IGNORECASE)
    
    # Clean domain dots and ensure valid filename
    base_name = base_name.replace('.', '_')
    base_name = re.sub(r'[^\w\-_]', '_', base_name)
    base_name = re.sub(r'_+', '_', base_name).strip('_')
    
    # Truncate if too long and add timestamp for uniqueness
    if len(base_name) > 50:
        base_name = base_name[:50]
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{base_name}_{timestamp}.md"
    filepath = Path(".claudecode") / filename
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(markdown_data['content'])
    
    return f"Markdown saved at {filepath}"

tool_functions = {
    "extract_markdown_from_url": extract_markdown_from_url,
    "web_fetch": web_fetch,
}


def run_example():
    from flowgen.llm.gemini import Gemini
    llm = Gemini(tools=list(tool_functions.values()))
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Extract content from https://example.com and convert to markdown"}
    ]

    # Agentic loop - keep calling tools until no more tool calls
    while True:
        response = llm(messages)

        # Check if there are tool calls
        if 'tools' not in response or not response['tools']:
            # No more tool calls, show final content
            print("=== FINAL RESPONSE ===")
            print(response.get('content', 'No content'))
            break

        # Add the assistant message with tool calls first
        messages.append({
            "role": "assistant",
            "tool_calls": [
                {
                    "id": tool_call.get('id', f"call_{i}"),
                    "function": {
                        "name": tool_call['name'],
                        "arguments": json.dumps(tool_call['arguments'])
                    },
                    "type": "function"
                }
                for i, tool_call in enumerate(response['tools'])
            ]
        })

        # Process each tool call and add tool results
        for i, tool_call in enumerate(response['tools']):
            tool_name = tool_call['name']
            tool_args = tool_call['arguments']
            tool_id = tool_call.get('id', f"call_{i}")

            print(f"Calling tool: {tool_name} with args: {tool_args}")

            # Execute the tool
            tool_result = tool_functions[tool_name](**tool_args)
            print('============')
            print(tool_result)
            print('============')

            # Pretty print tool result
            print(f"=== TOOL RESULT: {tool_name} ===")
            if isinstance(tool_result, dict):
                for key, value in tool_result.items():
                    print(f"{key}:")
                    value_preview = str(value)[:200] + "..." if len(str(value)) > 200 else str(value)
                    print(f"  {value_preview}")
            else:
                result_preview = str(tool_result)[:200] + "..." if len(str(tool_result)) > 200 else str(tool_result)
                print(result_preview)
            print("=" * 40)

            # Add tool result back to messages
            messages.append({
                "role": "tool",
                "tool_call_id": tool_id,
                "name": tool_name,
                "content": str(tool_result)
            })


def run_sample():
    urls = ["https://example.com"]
    res = web_fetch(urls)
    print(res)


if __name__ == '__main__':
    run_sample()
