import json
import asyncio
from typing import List, Dict, Optional, Tuple
from dataclasses import asdict

from scrapling.fetchers import Fetcher, DynamicFetcher, StealthyFetcher
from scrapling.core.shell import Convertor
from scrapling.core._types import extraction_types, SelectorWaitStates
from curl_cffi.requests import BrowserTypeLiteral

SYSTEM_PROMPT = """You are a web scraping assistant with access to powerful scrapling tools for extracting and processing web content:

- scrapling_get: Fast HTTP requests with browser fingerprint impersonation for basic scraping
- scrapling_bulk_get: Async bulk version of get for scraping multiple URLs simultaneously  
- scrapling_fetch: Browser-based fetching for dynamic content with JavaScript support
- scrapling_bulk_fetch: Async bulk browser fetching for multiple dynamic URLs
- scrapling_stealthy_fetch: Stealth browser fetching to bypass Cloudflare and anti-bot protections
- scrapling_bulk_stealthy_fetch: Async bulk stealth fetching for multiple protected URLs

All tools support markdown, HTML, or text extraction with CSS selector targeting and main content filtering."""


def scrapling_get(
    url: str,
    impersonate: Optional[str] = "chrome",
    extraction_type: str = "markdown",
    css_selector: Optional[str] = None,
    main_content_only: bool = True,
    timeout: Optional[int] = 30,
    retries: Optional[int] = 3,
    retry_delay: Optional[int] = 1,
) -> Dict:
    """Make GET HTTP request to a URL and return structured markdown/HTML/text output.
    
    Args:
        url: The URL to request
        extraction_type: "markdown", "html", or "text" 
        css_selector: CSS selector to extract specific elements
        main_content_only: Extract only main content (body tag)
        timeout: Request timeout in seconds
        retries: Number of retry attempts
        retry_delay: Delay between retries in seconds
    """
    try:
        page = Fetcher.get(
            url,
            timeout=timeout,
            retries=retries,
            retry_delay=retry_delay,
            impersonate=impersonate,
        )
        
        content = list(Convertor._extract_content(
            page,
            css_selector=css_selector,
            extraction_type=extraction_type,
            main_content_only=main_content_only,
        ))
        
        return {
            "status": page.status,
            "content": content,
            "url": page.url
        }
    except Exception as e:
        return {"status": 0, "content": [f"Error: {str(e)}"], "url": url}


async def scrapling_bulk_get(
    urls: Tuple[str, ...],
    impersonate: Optional[str] = "chrome", 
    extraction_type: str = "markdown",
    css_selector: Optional[str] = None,
    main_content_only: bool = True,
    timeout: Optional[int] = 30,
    retries: Optional[int] = 3,
    retry_delay: Optional[int] = 1,
) -> List[Dict]:
    """Bulk GET requests to multiple URLs with async processing.
    
    Args:
        urls: Tuple of URLs to request
        extraction_type: "markdown", "html", or "text"
        css_selector: CSS selector to extract specific elements  
        main_content_only: Extract only main content (body tag)
        timeout: Request timeout in seconds
        retries: Number of retry attempts
        retry_delay: Delay between retries in seconds
    """
    from scrapling.engines import AsyncFetcherClient
    
    async_client = AsyncFetcherClient()
    
    async def fetch_single(url: str):
        try:
            page = await async_client.get(
                url,
                timeout=timeout,
                retries=retries,
                retry_delay=retry_delay,
                impersonate=impersonate,
            )
            
            content = list(Convertor._extract_content(
                page,
                css_selector=css_selector,
                extraction_type=extraction_type,
                main_content_only=main_content_only,
            ))
            
            return {
                "status": page.status,
                "content": content,
                "url": page.url
            }
        except Exception as e:
            return {"status": 0, "content": [f"Error: {str(e)}"], "url": url}
    
    results = await asyncio.gather(*[fetch_single(url) for url in urls])
    return results


def scrapling_fetch(
    url: str,
    extraction_type: str = "markdown",
    css_selector: Optional[str] = None,
    main_content_only: bool = True,
    headless: bool = True,
    timeout: int = 30000,
    wait: int = 0,
    network_idle: bool = False,
    wait_selector: Optional[str] = None,
    disable_resources: bool = False,
) -> Dict:
    """Use browser to fetch dynamic content with JavaScript support.
    
    Args:
        url: The URL to fetch
        extraction_type: "markdown", "html", or "text"
        css_selector: CSS selector to extract specific elements
        main_content_only: Extract only main content (body tag)
        headless: Run browser in headless mode
        timeout: Timeout in milliseconds
        wait: Wait time in milliseconds after page loads
        network_idle: Wait for no network activity
        wait_selector: Wait for specific CSS selector
        disable_resources: Block images/fonts for speed
    """
    try:
        page = DynamicFetcher.fetch(
            url,
            headless=headless,
            timeout=timeout,
            wait=wait,
            network_idle=network_idle,
            wait_selector=wait_selector,
            disable_resources=disable_resources,
        )
        
        content = list(Convertor._extract_content(
            page,
            css_selector=css_selector,
            extraction_type=extraction_type,
            main_content_only=main_content_only,
        ))
        
        return {
            "status": page.status,
            "content": content,
            "url": page.url
        }
    except Exception as e:
        return {"status": 0, "content": [f"Error: {str(e)}"], "url": url}


async def scrapling_bulk_fetch(
    urls: Tuple[str, ...],
    extraction_type: str = "markdown",
    css_selector: Optional[str] = None,
    main_content_only: bool = True,
    headless: bool = True,
    timeout: int = 30000,
    wait: int = 0,
    network_idle: bool = False,
    disable_resources: bool = False,
) -> List[Dict]:
    """Bulk browser fetching for multiple dynamic URLs.
    
    Args:
        urls: Tuple of URLs to fetch
        extraction_type: "markdown", "html", or "text"
        css_selector: CSS selector to extract specific elements
        main_content_only: Extract only main content (body tag)
        headless: Run browser in headless mode
        timeout: Timeout in milliseconds
        wait: Wait time in milliseconds after page loads
        network_idle: Wait for no network activity
        disable_resources: Block images/fonts for speed
    """
    from scrapling.engines import AsyncDynamicSession
    
    async with AsyncDynamicSession() as session:
        async def fetch_single(url: str):
            try:
                page = await session.fetch(
                    url,
                    headless=headless,
                    timeout=timeout,
                    wait=wait,
                    network_idle=network_idle,
                    disable_resources=disable_resources,
                )
                
                content = list(Convertor._extract_content(
                    page,
                    css_selector=css_selector,
                    extraction_type=extraction_type,
                    main_content_only=main_content_only,
                ))
                
                return {
                    "status": page.status,
                    "content": content,
                    "url": page.url
                }
            except Exception as e:
                return {"status": 0, "content": [f"Error: {str(e)}"], "url": url}
        
        results = await asyncio.gather(*[fetch_single(url) for url in urls])
        return results


def scrapling_stealthy_fetch(
    url: str,
    extraction_type: str = "markdown",
    css_selector: Optional[str] = None,
    main_content_only: bool = True,
    headless: bool = True,
    timeout: int = 30000,
    wait: int = 0,
    network_idle: bool = False,
    solve_cloudflare: bool = True,
    humanize: bool = True,
    block_images: bool = False,
    disable_resources: bool = False,
) -> Dict:
    """Stealth browser fetching to bypass Cloudflare and anti-bot protections.
    
    Args:
        url: The URL to fetch
        extraction_type: "markdown", "html", or "text"
        css_selector: CSS selector to extract specific elements
        main_content_only: Extract only main content (body tag)
        headless: Run browser in headless mode
        timeout: Timeout in milliseconds
        wait: Wait time in milliseconds after page loads
        network_idle: Wait for no network activity
        solve_cloudflare: Automatically solve Cloudflare challenges
        humanize: Humanize cursor movements
        block_images: Block image loading for speed
        disable_resources: Block unnecessary resources for speed
    """
    try:
        page = StealthyFetcher.fetch(
            url,
            headless=headless,
            timeout=timeout,
            wait=wait,
            network_idle=network_idle,
            solve_cloudflare=solve_cloudflare,
            humanize=humanize,
            block_images=block_images,
            disable_resources=disable_resources,
        )
        
        content = list(Convertor._extract_content(
            page,
            css_selector=css_selector,
            extraction_type=extraction_type,
            main_content_only=main_content_only,
        ))
        
        return {
            "status": page.status,
            "content": content,
            "url": page.url
        }
    except Exception as e:
        return {"status": 0, "content": [f"Error: {str(e)}"], "url": url}


async def scrapling_bulk_stealthy_fetch(
    urls: Tuple[str, ...],
    extraction_type: str = "markdown",
    css_selector: Optional[str] = None,
    main_content_only: bool = True,
    headless: bool = True,
    timeout: int = 30000,
    wait: int = 0,
    network_idle: bool = False,
    solve_cloudflare: bool = True,
    humanize: bool = True,
    disable_resources: bool = False,
) -> List[Dict]:
    """Bulk stealth browser fetching for multiple protected URLs.
    
    Args:
        urls: Tuple of URLs to fetch
        extraction_type: "markdown", "html", or "text"
        css_selector: CSS selector to extract specific elements
        main_content_only: Extract only main content (body tag)
        headless: Run browser in headless mode
        timeout: Timeout in milliseconds
        wait: Wait time in milliseconds after page loads
        network_idle: Wait for no network activity
        solve_cloudflare: Automatically solve Cloudflare challenges
        humanize: Humanize cursor movements
        disable_resources: Block unnecessary resources for speed
    """
    from scrapling.engines import AsyncStealthySession
    
    async with AsyncStealthySession() as session:
        async def fetch_single(url: str):
            try:
                page = await session.fetch(
                    url,
                    headless=headless,
                    timeout=timeout,
                    wait=wait,
                    network_idle=network_idle,
                    solve_cloudflare=solve_cloudflare,
                    humanize=humanize,
                    disable_resources=disable_resources,
                )
                
                content = list(Convertor._extract_content(
                    page,
                    css_selector=css_selector,
                    extraction_type=extraction_type,
                    main_content_only=main_content_only,
                ))
                
                return {
                    "status": page.status,
                    "content": content,
                    "url": page.url
                }
            except Exception as e:
                return {"status": 0, "content": [f"Error: {str(e)}"], "url": url}
        
        results = await asyncio.gather(*[fetch_single(url) for url in urls])
        return results


# Sync wrappers for async functions
def scrapling_bulk_get_sync(urls: Tuple[str, ...], **kwargs) -> List[Dict]:
    """Sync wrapper for bulk_get"""
    return asyncio.run(scrapling_bulk_get(urls, **kwargs))

def scrapling_bulk_fetch_sync(urls: Tuple[str, ...], **kwargs) -> List[Dict]:
    """Sync wrapper for bulk_fetch"""
    return asyncio.run(scrapling_bulk_fetch(urls, **kwargs))

def scrapling_bulk_stealthy_fetch_sync(urls: Tuple[str, ...], **kwargs) -> List[Dict]:
    """Sync wrapper for bulk_stealthy_fetch"""
    return asyncio.run(scrapling_bulk_stealthy_fetch(urls, **kwargs))


# Legacy functions for backward compatibility
def extract_markdown_from_url(url: str):
    """Extract markdown content directly from URL using scrapling"""
    result = scrapling_get(url, extraction_type="markdown")
    return {"url": result["url"], "content": "".join(result["content"]) if result["content"] else None}


def web_fetch(url: str):
    """Fetch URL and save as markdown file in .claudecode directory"""
    import re
    import urllib.parse
    from datetime import datetime
    from pathlib import Path
    
    result = scrapling_get(url, extraction_type="markdown")
    
    if not result["content"] or result["status"] != 200:
        return f"Failed to fetch content from URL (status: {result['status']})"
    
    # Create .claudecode directory if it doesn't exist
    Path(".claudecode").mkdir(exist_ok=True)
    
    # Generate filename from URL
    parsed_url = urllib.parse.urlparse(url)
    domain = parsed_url.netloc.replace('www.', '')
    path = parsed_url.path.strip('/')
    
    if path:
        path_clean = re.sub(r'[^\w\-_.]', '_', path.replace('/', '_'))
        path_clean = re.sub(r'_+', '_', path_clean).strip('_')
        base_name = f"{domain}_{path_clean}"
    else:
        base_name = domain
    
    base_name = re.sub(r'\.(html?|php|asp|jsp)$', '', base_name, flags=re.IGNORECASE)
    base_name = base_name.replace('.', '_')
    base_name = re.sub(r'[^\w\-_]', '_', base_name)
    base_name = re.sub(r'_+', '_', base_name).strip('_')
    
    if len(base_name) > 50:
        base_name = base_name[:50]
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{base_name}_{timestamp}.md"
    filepath = Path(".claudecode") / filename
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("".join(result["content"]))
    
    return f"Markdown saved at {filepath}"


tool_functions = {
    "scrapling_get": scrapling_get,
    "scrapling_bulk_get": scrapling_bulk_get_sync,
    "scrapling_fetch": scrapling_fetch,
    "scrapling_bulk_fetch": scrapling_bulk_fetch_sync,
    "scrapling_stealthy_fetch": scrapling_stealthy_fetch,
    "scrapling_bulk_stealthy_fetch": scrapling_bulk_stealthy_fetch_sync,
    "extract_markdown_from_url": extract_markdown_from_url,
    "web_fetch": web_fetch,
}


def run_example():
    """Example usage of scrapling tools"""
    from flowgen.llm.gemini import Gemini
    llm = Gemini(tools=list(tool_functions.values()))
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Extract content from https://example.com using scrapling_get"}
    ]

    while True:
        response = llm(messages)
        if 'tools' not in response or not response['tools']:
            print("=== FINAL RESPONSE ===")
            print(response.get('content', 'No content'))
            break

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

        for i, tool_call in enumerate(response['tools']):
            tool_name = tool_call['name']
            tool_args = tool_call['arguments']
            tool_id = tool_call.get('id', f"call_{i}")

            print(f"Calling tool: {tool_name} with args: {tool_args}")
            tool_result = tool_functions[tool_name](**tool_args)
            print(f"Tool result: {str(tool_result)[:200]}...")

            messages.append({
                "role": "tool",
                "tool_call_id": tool_id,
                "name": tool_name,
                "content": str(tool_result)
            })


if __name__ == '__main__':
    # Test scrapling_get
    result = scrapling_get("https://en.wikipedia.org/wiki/Tiger")
    for x in result['content']:
        print(x)
        print('----')

    print(f"Status: {result['status']}")
    print(f"Content length: {len(''.join(result['content']))}")
